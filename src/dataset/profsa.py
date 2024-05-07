import logging
from pathlib import Path
from typing import Any, Dict, Union

import numpy as np
import torch
from lightning import LightningDataModule
from scipy.spatial import distance_matrix
from torch.utils.data import DataLoader, Dataset
from unicore.data import (
    AppendTokenDataset,
    Dictionary,
    FromNumpyDataset,
    LMDBDataset,
    NestedDictionaryDataset,
    PrependTokenDataset,
    RawArrayDataset,
    RawLabelDataset,
    RightPadDataset,
    RightPadDataset2D,
    SortDataset,
    TokenizeDataset,
    data_utils,
)

from src.utils.torchtool import is_rank_zero

from .components.lmdb import LMDBDataset as LMDBDataset2
from .components.lmdb import UniMolLMDBDataset
from .components.unimol import (
    AffinityDataset,
    CroppingPocketDockingPoseDataset,
    DistanceDataset,
    EdgeTypeDataset,
    KeyDataset,
    LengthDataset,
    NormalizeDataset,
    PrependAndAppend2DDataset,
    RemoveHydrogenDataset,
    RemoveHydrogenPocketDataset,
    ResamplingDataset,
    RightPadDatasetCoord,
)

logger = logging.getLogger(__name__)

DICT_MOL_PATH = (
    Path(__file__).parent
    / "components"
    / "unimol"
    / "dictionary"
    / "dict_mol.txt"
)
DICT_POCKET_PATH = (
    Path(__file__).parent
    / "components"
    / "unimol"
    / "dictionary"
    / "dict_pkt.txt"
)
mol_dict = Dictionary.load(str(DICT_MOL_PATH))
pocket_dict = Dictionary.load(str(DICT_POCKET_PATH))
mol_dict.add_symbol("[MASK]", is_special=True)
pocket_dict.add_symbol("[MASK]", is_special=True)


class ProFSADataset(Dataset):
    def __init__(
        self,
        data_dir: str,
        data_file: str = "valid.lmdb",
        mol_dict_file: str = DICT_MOL_PATH,
        pocket_dict_file: str = DICT_POCKET_PATH,
        max_pocket_atoms: int = 256,
        max_seq_len: int = 512,
        shuffle: bool = False,
        seed: int = 0,
        ligand_atoms_key="lig_atoms_real",
        ligand_coord_key="lig_coord_real",
        pocket_atoms_key="pocket_atoms",
        pocket_coord_key="pocket_coordinates",
        affinity_key="affinity",
    ):
        self.data_dir = Path(data_dir)
        self.data_path = str(self.data_dir / data_file)
        self.mol_dict_path = str(self.data_dir / mol_dict_file)
        self.pocket_dict_path = str(self.data_dir / pocket_dict_file)
        self.max_pocket_atoms = max_pocket_atoms
        self.max_seq_len = max_seq_len
        self.shuffle = shuffle
        self.seed = seed
        self.ligand_atoms_key = ligand_atoms_key
        self.ligand_coord_key = ligand_coord_key
        self.pocket_atoms_key = pocket_atoms_key
        self.pocket_coord_key = pocket_coord_key
        self.affinity_key = affinity_key

        self.mol_dict, self.pocket_dict = self.load_dictionary()
        self.dataset = self.load_dataset()

        if is_rank_zero():
            logger.info(
                f"{self.__class__.__name__}: {len(self)} samples in total."
            )

    def __getitem__(self, index: int):
        return self.dataset[index]

    def __len__(self):
        return len(self.dataset)

    def set_epoch(self, epoch: int):
        self.dataset.set_epoch(epoch)

    def load_dictionary(self):
        mol_dict = Dictionary.load(self.mol_dict_path)
        pocket_dict = Dictionary.load(self.pocket_dict_path)
        mol_dict.add_symbol("[MASK]", is_special=True)
        pocket_dict.add_symbol("[MASK]", is_special=True)
        if is_rank_zero():
            logger.info("mol dictionary: {} types".format(len(mol_dict)))
            logger.info("pocket dictionary: {} types".format(len(pocket_dict)))
        return mol_dict, pocket_dict

    def load_dataset(self):
        dataset = LMDBDataset(self.data_path)
        if self.shuffle:
            smi_dataset = KeyDataset(dataset, "smi")
            poc_dataset = KeyDataset(dataset, "pocket")
            dataset = AffinityDataset(
                dataset,
                self.seed,
                self.ligand_atoms_key,
                self.ligand_coord_key,
                self.pocket_atoms_key,
                self.pocket_coord_key,
                self.affinity_key,
                True,
            )
            tgt_dataset = KeyDataset(dataset, "affinity")

        else:
            dataset = AffinityDataset(
                dataset,
                self.seed,
                self.ligand_atoms_key,
                self.ligand_coord_key,
                self.pocket_atoms_key,
                self.pocket_coord_key,
                self.affinity_key,
            )
            tgt_dataset = KeyDataset(dataset, "affinity")
            smi_dataset = KeyDataset(dataset, "smi")
            poc_dataset = KeyDataset(dataset, "pocket")

        def PrependAndAppend(dataset, pre_token, app_token):
            dataset = PrependTokenDataset(dataset, pre_token)
            return AppendTokenDataset(dataset, app_token)

        dataset = RemoveHydrogenPocketDataset(
            dataset,
            "pocket_atoms",
            "pocket_coordinates",
            "holo_pocket_coordinates",
            True,
            True,
        )
        dataset = CroppingPocketDockingPoseDataset(
            dataset,
            self.seed,
            "pocket_atoms",
            "pocket_coordinates",
            "holo_pocket_coordinates",
            self.max_pocket_atoms,
        )

        dataset = RemoveHydrogenDataset(
            dataset, "atoms", "coordinates", True, True
        )

        apo_dataset = NormalizeDataset(dataset, "coordinates")
        apo_dataset = NormalizeDataset(apo_dataset, "pocket_coordinates")

        src_dataset = KeyDataset(apo_dataset, "atoms")
        mol_len_dataset = LengthDataset(src_dataset)
        src_dataset = TokenizeDataset(
            src_dataset, self.mol_dict, max_seq_len=self.max_seq_len
        )
        coord_dataset = KeyDataset(apo_dataset, "coordinates")
        src_dataset = PrependAndAppend(
            src_dataset, self.mol_dict.bos(), self.mol_dict.eos()
        )
        edge_type = EdgeTypeDataset(src_dataset, len(self.mol_dict))
        coord_dataset = FromNumpyDataset(coord_dataset)
        distance_dataset = DistanceDataset(coord_dataset)
        coord_dataset = PrependAndAppend(coord_dataset, 0.0, 0.0)
        distance_dataset = PrependAndAppend2DDataset(distance_dataset, 0.0)

        src_pocket_dataset = KeyDataset(apo_dataset, "pocket_atoms")
        pocket_len_dataset = LengthDataset(src_pocket_dataset)
        src_pocket_dataset = TokenizeDataset(
            src_pocket_dataset,
            self.pocket_dict,
            max_seq_len=self.max_seq_len,
        )
        coord_pocket_dataset = KeyDataset(apo_dataset, "pocket_coordinates")
        src_pocket_dataset = PrependAndAppend(
            src_pocket_dataset,
            self.pocket_dict.bos(),
            self.pocket_dict.eos(),
        )
        pocket_edge_type = EdgeTypeDataset(
            src_pocket_dataset, len(self.pocket_dict)
        )
        coord_pocket_dataset = FromNumpyDataset(coord_pocket_dataset)
        distance_pocket_dataset = DistanceDataset(coord_pocket_dataset)
        coord_pocket_dataset = PrependAndAppend(coord_pocket_dataset, 0.0, 0.0)
        distance_pocket_dataset = PrependAndAppend2DDataset(
            distance_pocket_dataset, 0.0
        )

        dataset = NestedDictionaryDataset(
            {
                "net_input": {
                    "mol_src_tokens": RightPadDataset(
                        src_dataset,
                        pad_idx=self.mol_dict.pad(),
                    ),
                    "mol_src_distance": RightPadDataset2D(
                        distance_dataset,
                        pad_idx=0,
                    ),
                    "mol_src_edge_type": RightPadDataset2D(
                        edge_type,
                        pad_idx=0,
                    ),
                    "pocket_src_tokens": RightPadDataset(
                        src_pocket_dataset,
                        pad_idx=self.pocket_dict.pad(),
                    ),
                    "pocket_src_distance": RightPadDataset2D(
                        distance_pocket_dataset,
                        pad_idx=0,
                    ),
                    "pocket_src_edge_type": RightPadDataset2D(
                        pocket_edge_type,
                        pad_idx=0,
                    ),
                    "pocket_src_coord": RightPadDatasetCoord(
                        coord_pocket_dataset,
                        pad_idx=0,
                    ),
                    "mol_len": RawArrayDataset(mol_len_dataset),
                    "pocket_len": RawArrayDataset(pocket_len_dataset),
                },
                "target": {
                    "finetune_target": RawLabelDataset(tgt_dataset),
                },
                "smi_name": RawArrayDataset(smi_dataset),
                "pocket_name": RawArrayDataset(poc_dataset),
            },
        )
        if self.shuffle:
            with data_utils.numpy_seed(self.seed):
                shuffle = np.random.permutation(len(src_dataset))

            dataset = SortDataset(
                dataset,
                sort_order=[shuffle],
            )
            dataset = ResamplingDataset(dataset)
        return dataset


class ProFSADataModule(LightningDataModule):
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
        dataset = ProFSADataset(**dataset_cfg)
        if type(self.hparams["batch_size"]) == int:
            batch_size = self.hparams.batch_size
        else:
            batch_size = self.hparams.batch_size[split]
        dataloader = DataLoader(
            dataset,
            collate_fn=dataset.dataset.collater,
            batch_size=batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
        )
        return dataloader

    def train_dataloader(self):
        return self._dataloader("train")

    def val_dataloader(self):
        return self._dataloader("val")

    def test_dataloader(self):
        if "test" not in self.hparams.dataset_cfg:
            super().test_dataloader()
        return self._dataloader("test")

    def predict_dataloader(self) -> Any:
        return self._dataloader("train")


def process_pocket_mol(
    mode: str,  # pocket or mol
    atoms: np.ndarray,
    coordinates: np.ndarray,
    max_pocket_atoms: int = 256,
    max_seq_len: int = 512,
):

    if mode == "pocket":
        token_dict = pocket_dict
    elif mode == "mol":
        token_dict = mol_dict
    else:
        raise ValueError(f"Unsupported mode: {mode}")

    # remove hydrogen atoms
    atoms = np.array(atoms)
    mask_hydrogen = atoms != "H"
    atoms = atoms[mask_hydrogen]
    coordinates = coordinates[mask_hydrogen]

    # crop pocket
    if mode == "pocket" and len(atoms) > max_pocket_atoms:
        distance = np.linalg.norm(
            coordinates - coordinates.mean(axis=0), axis=1
        )

        def softmax(x):
            x -= np.max(x)
            x = np.exp(x) / np.sum(np.exp(x))
            return x

        distance += 1  # prevent inf
        weight = softmax(np.reciprocal(distance))
        index = np.random.choice(
            len(atoms), max_pocket_atoms, replace=False, p=weight
        )
        atoms = atoms[index]
        coordinates = coordinates[index]

    # normalize
    if len(coordinates) == 0:  # throw away empty pocket
        raise ValueError("Detected empty pocket after cropping")
    coordinates = coordinates - coordinates.mean(axis=0)
    coordinates = torch.from_numpy(coordinates)

    # tokenize
    assert len(atoms) < max_seq_len and len(atoms) > 0
    atoms = torch.from_numpy(token_dict.vec_index(atoms)).long()

    # prepend and append atoms
    atoms = torch.cat(
        [torch.full_like(atoms[0], token_dict.bos()).unsqueeze(0), atoms], dim=0
    )
    atoms = torch.cat(
        [atoms, torch.full_like(atoms[0], token_dict.eos()).unsqueeze(0)], dim=0
    )

    # edge type
    edge_type = atoms.view(-1, 1) * len(token_dict) + atoms.view(1, -1)

    # distance
    pos = coordinates.view(-1, 3).numpy()
    distance = torch.from_numpy(distance_matrix(pos, pos).astype(np.float32))

    # prepend and append coordiantes
    coordinates = torch.cat(
        [torch.zeros_like(coordinates[0]).unsqueeze(0), coordinates], dim=0
    )
    coordinates = torch.cat(
        [coordinates, torch.zeros_like(coordinates[0]).unsqueeze(0)], dim=0
    )

    # prepend and append distance
    h, w = distance.size(-2), distance.size(-1)
    new_distance = torch.full((h + 2, w + 2), 0.0).type_as(distance)
    new_distance[1:-1, 1:-1] = distance
    distance = new_distance

    return {
        "atoms": atoms,
        "distance": distance,
        "edge_type": edge_type,
    }


def process_pocket(
    atoms: np.ndarray,
    coordinates: np.ndarray,
    max_atoms: int = 256,
    max_seq_len: int = 512,
):
    return process_pocket_mol(
        mode="pocket",
        atoms=atoms,
        coordinates=coordinates,
        max_pocket_atoms=max_atoms,
        max_seq_len=max_seq_len,
    )


def process_mol(
    atoms: np.ndarray,
    coordinates: np.ndarray,
    max_seq_len: int = 512,
):
    return process_pocket_mol(
        mode="mol",
        atoms=atoms,
        coordinates=coordinates,
        max_pocket_atoms=None,
        max_seq_len=max_seq_len,
    )


class PocketMolDataset(Dataset):

    POCKET_ATOM_KEY = "pocket_src_tokens"
    POCKET_DISTANCE_KEY = "pocket_src_distance"
    POCKET_EDGE_TYPE_KEY = "pocket_src_edge_type"
    MOL_ATOM_KEY = "mol_src_tokens"
    MOL_DISTANCE_KEY = "mol_src_distance"
    MOL_EDGE_TYPE_KEY = "mol_src_edge_type"

    def __init__(
        self,
        lmdb_path,
        pocket_atoms_key="pocket_atoms",
        pocket_coord_key="pocket_coordinates",
        mol_atoms_key="lig_atoms_real",
        mol_coord_key="lig_coord_real",
        max_pocket_atoms: int = 256,
        max_seq_len: int = 512,
    ):
        self.lmdb_path = lmdb_path
        self.pocket_atoms_key = pocket_atoms_key
        self.pocket_coord_key = pocket_coord_key
        self.mol_atoms_key = mol_atoms_key
        self.mol_coord_key = mol_coord_key
        self.max_pocket_atoms = max_pocket_atoms
        self.max_seq_len = max_seq_len
        self.dataset = UniMolLMDBDataset(self.lmdb_path)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        sample = self.dataset[index]
        pocket_atoms = sample[self.pocket_atoms_key]
        pocket_coords = sample[self.pocket_coord_key]
        mol_atoms = sample[self.mol_atoms_key]
        mol_coords = sample[self.mol_coord_key]
        try:
            pocket_result = process_pocket(
                pocket_atoms,
                pocket_coords,
                self.max_pocket_atoms,
                self.max_seq_len,
            )
            mol_result = process_mol(mol_atoms, mol_coords, self.max_seq_len)
        except Exception:
            return None
        return {
            self.POCKET_ATOM_KEY: pocket_result["atoms"],
            self.POCKET_DISTANCE_KEY: pocket_result["distance"],
            self.POCKET_EDGE_TYPE_KEY: pocket_result["edge_type"],
            self.MOL_ATOM_KEY: mol_result["atoms"],
            self.MOL_DISTANCE_KEY: mol_result["distance"],
            self.MOL_EDGE_TYPE_KEY: mol_result["edge_type"],
        }

    def collater(self, samples):
        samples = [s for s in samples if s is not None]
        pocket_atoms_list = [s[self.POCKET_ATOM_KEY] for s in samples]
        pocket_atoms_list = data_utils.collate_tokens(
            pocket_atoms_list,
            pocket_dict.pad(),
            left_pad=False,
            pad_to_multiple=8,
        )
        pocket_distance_list = [s[self.POCKET_DISTANCE_KEY] for s in samples]
        pocket_distance_list = data_utils.collate_tokens_2d(
            pocket_distance_list, 0, left_pad=False, pad_to_multiple=8
        )
        pocket_edge_type_list = [s[self.POCKET_EDGE_TYPE_KEY] for s in samples]
        pocket_edge_type_list = data_utils.collate_tokens_2d(
            pocket_edge_type_list, 0, left_pad=False, pad_to_multiple=8
        )
        mol_atoms_list = [s[self.MOL_ATOM_KEY] for s in samples]
        mol_atoms_list = data_utils.collate_tokens(
            mol_atoms_list, mol_dict.pad(), left_pad=False, pad_to_multiple=8
        )
        mol_distance_list = [s[self.MOL_DISTANCE_KEY] for s in samples]
        mol_distance_list = data_utils.collate_tokens_2d(
            mol_distance_list, 0, left_pad=False, pad_to_multiple=8
        )
        mol_edge_type_list = [s[self.MOL_EDGE_TYPE_KEY] for s in samples]
        mol_edge_type_list = data_utils.collate_tokens_2d(
            mol_edge_type_list, 0, left_pad=False, pad_to_multiple=8
        )
        return {
            self.POCKET_ATOM_KEY: pocket_atoms_list,
            self.POCKET_DISTANCE_KEY: pocket_distance_list,
            self.POCKET_EDGE_TYPE_KEY: pocket_edge_type_list,
            self.MOL_ATOM_KEY: mol_atoms_list,
            self.MOL_DISTANCE_KEY: mol_distance_list,
            self.MOL_EDGE_TYPE_KEY: mol_edge_type_list,
        }


class MolDataset(Dataset):

    ATOM_KEY = "mol_src_tokens"
    DISTANCE_KEY = "mol_src_distance"
    EDGE_TYPE_KEY = "mol_src_edge_type"

    def __init__(
        self,
        lmdb_path,
        atoms_key="lig_atoms_real",
        coord_key="lig_coord_real",
        max_seq_len: int = 512,
    ):
        self.lmdb_path = lmdb_path
        self.atoms_key = atoms_key
        self.coord_key = coord_key
        self.max_seq_len = max_seq_len
        self.dataset = UniMolLMDBDataset(self.lmdb_path)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        sample = self.dataset[index]
        atoms = sample[self.atoms_key]
        coords = sample[self.coord_key]
        try:
            result = process_mol(atoms, coords, self.max_seq_len)
        except Exception:
            return None
        return {
            self.ATOM_KEY: result["atoms"],
            self.DISTANCE_KEY: result["distance"],
            self.EDGE_TYPE_KEY: result["edge_type"],
        }

    def collater(self, samples):
        samples = [s for s in samples if s is not None]
        atoms_list = [s[self.ATOM_KEY] for s in samples]
        atoms_list = data_utils.collate_tokens(
            atoms_list, mol_dict.pad(), left_pad=False, pad_to_multiple=8
        )
        distance_list = [s[self.DISTANCE_KEY] for s in samples]
        distance_list = data_utils.collate_tokens_2d(
            distance_list, 0, left_pad=False, pad_to_multiple=8
        )
        edge_type_list = [s[self.EDGE_TYPE_KEY] for s in samples]
        edge_type_list = data_utils.collate_tokens_2d(
            edge_type_list, 0, left_pad=False, pad_to_multiple=8
        )
        return {
            self.ATOM_KEY: atoms_list,
            self.DISTANCE_KEY: distance_list,
            self.EDGE_TYPE_KEY: edge_type_list,
        }


class NextMolDataset(Dataset):

    ATOM_KEY = "mol_src_tokens"
    DISTANCE_KEY = "mol_src_distance"
    EDGE_TYPE_KEY = "mol_src_edge_type"

    def __init__(
        self,
        lmdb_path,
        split: str = "full",
        atoms_key="atom_types",
        coord_key="coords",
        max_seq_len: int = 512,
        return_key: bool = False,
    ):
        self.lmdb_path = lmdb_path
        self.max_seq_len = max_seq_len
        self.split = split
        self.atom_key = atoms_key
        self.coord_key = coord_key
        self.return_key = return_key

        self.dataset = LMDBDataset2(self.lmdb_path)
        self.dataset.set_default_split(split)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        sample = self.dataset[index]
        if sample is None:
            return None
        atoms = sample[self.atom_key]
        if type(sample[self.coord_key]) == list:
            coords = sample[self.coord_key][0]
        else:
            coords = sample[self.coord_key]
        try:
            result = process_mol(atoms, coords, self.max_seq_len)
        except Exception:
            return None
        sample = {
            self.ATOM_KEY: result["atoms"],
            self.DISTANCE_KEY: result["distance"],
            self.EDGE_TYPE_KEY: result["edge_type"],
        }
        if self.return_key:
            sample["key"] = self.dataset.get_split(self.split)[index]
        return sample

    def collater(self, samples):
        samples = [s for s in samples if s is not None]
        atoms_list = [s[self.ATOM_KEY] for s in samples]
        atoms_list = data_utils.collate_tokens(
            atoms_list, mol_dict.pad(), left_pad=False, pad_to_multiple=8
        )
        distance_list = [s[self.DISTANCE_KEY] for s in samples]
        distance_list = data_utils.collate_tokens_2d(
            distance_list, 0, left_pad=False, pad_to_multiple=8
        )
        edge_type_list = [s[self.EDGE_TYPE_KEY] for s in samples]
        edge_type_list = data_utils.collate_tokens_2d(
            edge_type_list, 0, left_pad=False, pad_to_multiple=8
        )
        batch = {
            self.ATOM_KEY: atoms_list,
            self.DISTANCE_KEY: distance_list,
            self.EDGE_TYPE_KEY: edge_type_list,
        }
        if self.return_key:
            batch["key"] = [s["key"] for s in samples]
        return batch
