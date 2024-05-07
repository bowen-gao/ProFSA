import logging
from pathlib import Path
from typing import Any, Dict, List, Union

import numpy as np
import torch
import torch.nn.functional as F
from lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset

from src.dataset.components.lmdb import UniMolLMDBDataset

logger = logging.getLogger(__name__)

MAP_SIZE = 10 * 1024 * 1024 * 1024 * 1024  # 10T


def softmax(x):
    x -= np.max(x)
    x = np.exp(x) / np.sum(np.exp(x))
    return x


def stack_pad_tokens(tokens, maxlen):
    return torch.stack(
        [F.pad(token, (0, maxlen - len(token)), value=0) for token in tokens]
    )


def stack_pad_coords(coords, maxlen):
    return torch.stack(
        [
            F.pad(coord, (0, 0, 0, maxlen - len(coord)), value=0)
            for coord in coords
        ]
    )


def stack_pad_pairs(pairs, maxlen):
    return torch.stack(
        [
            F.pad(
                pair,
                (0, maxlen - len(pair), 0, maxlen - len(pair)),
                value=0,
            )
            for pair in pairs
        ]
    )


class AtomTokenizer:
    # fmt: off
    ATOMS = ["B", "C", "N", "O", "F", "Si", "P", "S", "Cl", "Se", "Br", "I", "Na", "Mg", "K", "Ca", "Fe", "Zn", "Cu", "Co", "Mn"]
    # fmt: on

    def __init__(self):
        self.pad_token = "[PAD]"
        self.bos_token = "[CLS]"
        self.eos_token = "[SEP]"
        self.unk_token = "[UNK]"
        self.tokens = (
            [self.pad_token, self.bos_token, self.eos_token]
            + [atom.upper() for atom in self.ATOMS]
            + [self.unk_token]
        )
        self.vocab = {t: i for i, t in enumerate(self.tokens)}
        self.ids_to_tokens = {i: t for t, i in self.vocab.items()}

    def get_vocab_size(self):
        return len(self.vocab)

    def __len__(self):
        return self.get_vocab_size()

    @property
    def pad_idx(self):
        return self.vocab[self.pad_token]

    @property
    def bos_idx(self):
        return self.vocab[self.bos_token]

    @property
    def eos_idx(self):
        return self.vocab[self.eos_token]

    @property
    def unk_idx(self):
        return self.vocab[self.unk_token]

    def tokenize(self, atoms: Union[str, List[str]]):
        if isinstance(atoms, list):
            return [
                self.vocab.get(atom.upper(), self.unk_idx) for atom in atoms
            ]
        else:
            return self.vocab.get(atoms.upper(), self.unk_idx)

    def __call__(self, atoms: Union[str, List[str]]):
        return self.tokenize(atoms)

    def convert_ids_to_tokens(self, ids: Union[int, List[int]]):
        if isinstance(ids, torch.Tensor):
            ids = ids.tolist()
        if isinstance(ids, list):
            return [self.ids_to_tokens.get(i, self.unk_token) for i in ids]
        else:
            return self.ids_to_tokens.get(ids, self.unk_token)


class PocketMolDataset(Dataset):
    def __init__(
        self,
        db_path: Path,
        max_pocket_atoms: int = 256,
        pocket_atom_key: str = None,
        pocket_coord_key: str = None,
        mol_atom_key: str = None,
        mol_coord_key: str = None,
        other_feat_keys: List[str] = None,
    ) -> None:
        super().__init__()

        self.db_path = db_path
        self.max_pocket_atoms = max_pocket_atoms
        self.pocket_atom_key = pocket_atom_key
        self.pocket_coord_key = pocket_coord_key
        self.mol_atom_key = mol_atom_key
        self.mol_coord_key = mol_coord_key
        self.other_feat_keys = other_feat_keys

        self.tokenizer = AtomTokenizer()
        self.lmdbdataset = UniMolLMDBDataset(self.db_path)

    def __len__(self):
        return len(self.lmdbdataset)

    def crop_atoms(self, coord, maxlen):
        """sample coordiantes based on distance to the center, with higher
        probability for atoms closer to the center."""
        if len(coord) <= maxlen:
            return torch.tensor(np.arange(len(coord))).long()
        center = coord.mean(axis=0)
        distance = np.linalg.norm(coord - center, axis=1)
        distance += 1
        weight = softmax(np.reciprocal(distance))
        index = np.random.choice(len(coord), maxlen, replace=False, p=weight)
        return torch.tensor(index).long()

    def read_pocket_mol(self, sample, atom_key, coord_key):
        atom = sample[atom_key]
        coord = sample[coord_key]

        atom_out = []
        coord_out = []
        # filter out H and D atoms
        for a, c in zip(atom, coord):
            if a == "H" or a == "D":
                continue
            else:
                t = self.tokenizer(a)
                atom_out.append(t)
                coord_out.append(c)

        # normalize coordinates
        coord_out = np.array(coord_out)
        coord_out -= coord_out.mean(axis=0)

        # crop atoms to max_pocket_atoms
        index = self.crop_atoms(coord_out, self.max_pocket_atoms)
        atom_out = torch.tensor(atom_out)[index].long()
        coord_out = torch.tensor(coord_out)[index].float()

        # add cls token to tokens, add mean of coordinates to coords
        atom_out = torch.cat(
            [
                torch.tensor([self.tokenizer.bos_idx]).long(),
                atom_out,
                torch.tensor([self.tokenizer.eos_idx]).long(),
            ],
            dim=0,
        )
        coord_out = torch.cat(
            [coord_out.mean(dim=0, keepdims=True), coord_out],
            dim=0,
        )

        # calculate distance matrix
        distance = torch.cdist(coord_out, coord_out)

        # calculate edge type
        edge_type = atom_out.unsqueeze(0) * len(
            self.tokenizer
        ) + atom_out.unsqueeze(1)

        return {
            "atom": atom_out,
            "coord": coord_out,
            "distance": distance,
            "edge_type": edge_type,
            "len": len(atom_out),
        }

    def read_other_feats(self, sample, other_feat_keys):
        if other_feat_keys is None or len(other_feat_keys) == 0:
            return {}
        else:
            other_feats = {}
            for key in other_feat_keys:
                other_feats[key] = sample[key]
            return other_feats

    def __getitem__(self, idx):
        sample = self.lmdbdataset[idx]
        # breakpoint()
        pocket = self.read_pocket_mol(
            sample, self.pocket_atom_key, self.pocket_coord_key
        )
        mol = self.read_pocket_mol(
            sample, self.mol_atom_key, self.mol_coord_key
        )
        other_feats = self.read_other_feats(sample, self.other_feat_keys)
        return {
            **{f"pocket_{k}": v for k, v in pocket.items()},
            **{f"mol_{k}": v for k, v in mol.items()},
            **other_feats,
        }

    def collate_fn(self, samples):
        max_pocket_len = max([s["pocket_len"] for s in samples])
        max_mol_len = max([s["mol_len"] for s in samples])
        batch = {
            "net_input": {
                "pocket_src_tokens": stack_pad_tokens(
                    [s["pocket_atom"] for s in samples], max_pocket_len
                ),
                "pocket_src_coord": stack_pad_coords(
                    [s["pocket_coord"] for s in samples], max_pocket_len
                ),
                "pocket_src_distance": stack_pad_pairs(
                    [s["pocket_distance"] for s in samples], max_pocket_len
                ),
                "pocket_src_edge_type": stack_pad_pairs(
                    [s["pocket_edge_type"] for s in samples], max_pocket_len
                ),
                "mol_src_tokens": stack_pad_tokens(
                    [s["mol_atom"] for s in samples], max_mol_len
                ),
                "mol_src_coord": stack_pad_coords(
                    [s["mol_coord"] for s in samples], max_mol_len
                ),
                "mol_src_distance": stack_pad_pairs(
                    [s["mol_distance"] for s in samples], max_mol_len
                ),
                "mol_src_edge_type": stack_pad_pairs(
                    [s["mol_edge_type"] for s in samples], max_mol_len
                ),
            }
        }
        for key in self.other_feat_keys:
            if type(samples[0][key]) is str:
                batch[key] = [s[key] for s in samples]
            else:
                batch[key] = torch.tensor(np.array([s[key] for s in samples]))
        return batch


class PocketMolDataModule(LightningDataModule):
    def __init__(
        self,
        num_workers: int = 0,
        pin_memory: bool = False,
        batch_size: Union[int, Dict[str, int]] = None,
        dataset_cfg: Dict[str, Any] = None,
    ):
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        self.save_hyperparameters(logger=False)

    def _dataloader(self, split):
        dataset_cfg = self.hparams["dataset_cfg"][split]
        dataset = PocketMolDataset(**dataset_cfg)
        if type(self.hparams["batch_size"]) == int:
            batch_size = self.hparams.batch_size
        else:
            batch_size = self.hparams.batch_size[split]
        dataloader = DataLoader(
            dataset,
            collate_fn=dataset.collate_fn,
            batch_size=batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=(split == "train"),
        )
        return dataloader

    def train_dataloader(self):
        return self._dataloader("train")

    def val_dataloader(self):
        return self._dataloader("val")


if __name__ == "__main__":
    frad_dataset = PocketMolDataset(
        db_path="/data/dataset/train/valid.lmdb",
        max_pocket_atoms=256,
        pocket_atom_key="pocket_atoms",
        pocket_coord_key="pocket_coordinates",
        mol_atom_key="lig_atoms_real",
        mol_coord_key="lig_coord_real",
        other_feat_keys=["feat", "pocket", "smi"],
    )
    frad_dataloader = DataLoader(
        frad_dataset,
        batch_size=4,
        shuffle=False,
        collate_fn=frad_dataset.collate_fn,
    )
    for idx, batch in enumerate(frad_dataloader):
        for key, val in batch.items():
            if type(val) is torch.Tensor:
                print(f"- {key}: {val.shape} {val.dtype}")
            else:
                print(f"- {key}: {len(val)}, e.g.: {val[0]}")
        break
