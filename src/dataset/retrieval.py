import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch
from lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from src.utils.datatool import (
    read_json,
    read_jsonlines,
    write_json,
    write_jsonlines,
)

from .components.lmdb import LMDBDataset
from .components.quantize import Quantizer
from .components.unimol import data_utils
from .profsa import mol_dict, pocket_dict, process_mol, process_pocket

logger = logging.getLogger(__name__)


class IdMap:
    """Mapping from docid to inchikey."""

    def __init__(self, id_mapping_path: str):
        self.id_mapping_path = id_mapping_path
        self.mapping = read_json(id_mapping_path, not_found_ok=True)

    def __contains__(self, key):
        return key in self.mapping

    def __getitem__(self, key):
        return self.mapping[key]

    def __setitem__(self, key, value):
        if key in self.mapping and self.mapping[key] != value:
            raise ValueError(
                f"Key {key} already exists with value {self.mapping[key]}"
            )
        self.mapping[key] = value

    def save(self):
        write_json(self.id_mapping_path, self.mapping)


class QuantizeTokenizer:
    def __init__(
        self,
        codesize: int = 32,
        codewords: int = 256,
    ):
        self.codesize = codesize
        self.codewords = codewords
        self.total_codes = codesize * codewords

        self.pad_idx = 0
        self.eos_idx = 1
        self.total_special_tokens = 2

        self.total_tokens = self.total_codes + self.total_special_tokens

    def encode(
        self,
        codes: List[int],
        prepend_pad: bool = False,
        append_eos: bool = False,
    ) -> torch.tensor:
        tokens = torch.tensor(codes)
        place = torch.arange(len(codes))
        tokens = (
            torch.tensor(codes)
            + place * self.codewords
            + self.total_special_tokens
        )
        if prepend_pad:
            tokens = torch.cat([torch.tensor([self.pad_idx]), tokens])
        if append_eos:
            tokens = torch.cat([tokens, torch.tensor([self.eos_idx])])
        return tokens

    def decode(
        self,
        tokens: torch.tensor,
        rm_prepend_pad: bool = False,
        rm_append_eos: bool = False,
    ) -> List[int]:
        if rm_prepend_pad:
            tokens = tokens[1:]
        if rm_append_eos:
            tokens = tokens[:-1]
        place = torch.arange(len(tokens))
        codes = tokens % (place * self.codewords)
        return codes.tolist()


def build_index(
    pocketdb_path: str,
    moldb_path: str,
    retrieval_split: str = None,
    indexing_split: str = None,
    quantizer_path: str = None,
    quantizer_target: str = None,
    index_path: str = None,
    id_mapping_path: str = None,
):
    pocketdb = LMDBDataset(pocketdb_path)
    moldb = LMDBDataset(moldb_path)
    quantizer = Quantizer(pretrain=quantizer_path)

    samples = []
    if id_mapping_path is not None:
        id_map = IdMap(id_mapping_path)

    # write retrieval samples
    if retrieval_split is not None:
        logger.info(f"Building retrieval samples for {retrieval_split}")
        keys = pocketdb.get_split(retrieval_split)
        for key in tqdm(keys, ncols=80):
            pocket = pocketdb[key]
            ligand_inchikey = pocket["inchikey"]
            if ligand_inchikey not in moldb:
                continue
            mol = moldb[ligand_inchikey]
            if quantizer_target not in mol:
                continue
            vector = mol[quantizer_target]
            quantized_vector = quantizer.encode(vector).tolist()
            quantized_str = ",".join([str(x) for x in quantized_vector])
            if id_mapping_path is not None:
                id_map[quantized_str] = ligand_inchikey
            samples.append({"p": key, "i": quantized_str})

    # write indexing samples
    if indexing_split is not None:
        logger.info(f"Building indexing samples for {indexing_split}")
        keys = moldb.get_split(indexing_split)
        for key in tqdm(keys, ncols=80):
            mol = moldb[key]
            if quantizer_target not in mol:
                continue
            vector = mol[quantizer_target]
            quantized_vector = quantizer.encode(vector).tolist()
            quantized_str = ",".join([str(x) for x in quantized_vector])
            if id_mapping_path is not None:
                id_map[quantized_str] = key
            samples.append({"m": key, "i": quantized_str})

    if id_mapping_path is not None:
        logger.info(f"Saving id mapping to {id_mapping_path}")
        id_map.save()

    logger.info(f"Writing index to {index_path}")
    Path(index_path).parent.mkdir(parents=True, exist_ok=True)
    write_jsonlines(index_path, samples)


def build_index_downstream(
    pocketdb_path: str,
    retrieval_split: str,
    index_dir: str,
    mode: str,
    index_mapping_path: str,
):
    pocketdb = LMDBDataset(pocketdb_path)

    logger.info("Loading inchikey-index mapping")
    id_mapping = {}
    with open(index_mapping_path, "r") as f:
        for line in tqdm(f):
            record = json.loads(line.strip())
            if "m" in record:
                id_mapping[record["m"]] = record["i"]

    logger.info(f"Building samples for {retrieval_split}")
    keys = pocketdb.get_split(retrieval_split)
    key_count = 0
    for key in keys:
        samples = []
        pocket = pocketdb[key]
        actives_inchikeys = pocket["actives"]
        decoys_inchikeys = pocket["decoys"]
        if retrieval_split == "litpcba":
            pocket_name = pocket["pocket_name"]
        for active_inchikey in tqdm(
            actives_inchikeys, desc=f"Processing {key} actives"
        ):
            if active_inchikey not in id_mapping:
                continue
            quantized_str = id_mapping[active_inchikey]
            samples.append({"p": key, "i": quantized_str})
            samples.append({"m": active_inchikey, "i": quantized_str})

        for decoy_inchikey in tqdm(
            decoys_inchikeys, desc=f"Processing {key} decoys"
        ):
            if decoy_inchikey not in id_mapping:
                continue
            quantized_str = id_mapping[decoy_inchikey]
            samples.append({"m": decoy_inchikey, "i": quantized_str})
        if retrieval_split == "litpcba":
            index_path = os.path.join(
                index_dir, f"{pocket_name}_{key[8:]}", f"{mode}.jsonl"
            )
        else:
            index_path = os.path.join(index_dir, key[5:], f"{mode}.jsonl")
        Path(index_path).parent.mkdir(parents=True, exist_ok=True)
        logger.info(f"Writing index to {index_path}")
        write_jsonlines(index_path, samples)
        key_count += 1
        print("-" * 50, f"[{key_count}/{len(keys)}]", "-" * 50)


class MolRetrievalDataset(Dataset):
    def __init__(
        self,
        index_path: str,
        pocketdb_path: str,
        moldb_path: str,
        duplicate_retrieval_ratio: int = 1,
        pocket_key: str = "pocket.r6",
        max_pocket_atoms: int = 256,
        max_seq_len: int = 512,
        limit_samples: int = None,
    ):
        self.pocketdb = LMDBDataset(pocketdb_path)
        self.molodb = LMDBDataset(moldb_path)
        self.duplicate_retrieval_ratio = duplicate_retrieval_ratio
        self.pocket_key = pocket_key
        self.max_pocket_atoms = max_pocket_atoms
        self.max_seq_len = max_seq_len
        self.limit_samples = limit_samples

        self.data = self.setup_data(index_path)
        self.tokenizer = QuantizeTokenizer()

    def setup_data(self, index_path):
        samples = read_jsonlines(index_path)
        retrieval_samples = []
        indexing_samples = []
        for sample in samples:
            if "p" in sample:
                retrieval_samples.append(sample)
            elif "m" in sample:
                indexing_samples.append(sample)
        data = {
            "retrieval": retrieval_samples,
            "indexing": indexing_samples,
        }
        data["full"] = (
            data["retrieval"] * self.duplicate_retrieval_ratio
            + data["indexing"]
        )
        return data

    def __len__(self):
        if self.limit_samples is not None:
            return min(self.limit_samples, len(self.data["full"]))
        return len(self.data["full"])

    def __getitem__(self, index):
        try:
            sample = self.data["full"][index]
            data = {}
            if "p" in sample:
                data["type"] = 0
                protein = self.pocketdb[sample["p"]]
                pocket = protein[self.pocket_key]
                data.update(
                    process_pocket(
                        pocket["atoms"],
                        pocket["coords"],
                        max_atoms=self.max_pocket_atoms,
                        max_seq_len=self.max_seq_len,
                    )
                )
            elif "m" in sample:
                data["type"] = 1
                mol = self.molodb[sample["m"]]
                if type(mol["coords"]) == list:
                    coords = mol["coords"][0]
                else:
                    coords = mol["coords"]
                data.update(
                    process_mol(
                        mol["atom_types"],
                        coords,
                        max_seq_len=self.max_seq_len,
                    )
                )

            data["labels"] = self.tokenizer.encode(
                [int(x) for x in sample["i"].split(",")]
            )
        except Exception:
            return None
        return data

    def collate_fn(self, batch):
        samples = [x for x in batch if x is not None]
        atoms_list = [x["atoms"] for x in samples]
        atoms_list = data_utils.collate_tokens(
            atoms_list, 0, left_pad=False, pad_to_multiple=8
        )  # pad is 0 in both mol_dict and pocket_dict
        distance_list = [x["distance"] for x in samples]
        distance_list = data_utils.collate_tokens_2d(
            distance_list, 0, left_pad=False, pad_to_multiple=8
        )
        edge_type_list = [x["edge_type"] for x in samples]
        edge_type_list = data_utils.collate_tokens_2d(
            edge_type_list, 0, left_pad=False, pad_to_multiple=8
        )
        batch = {
            "atoms": atoms_list,
            "distance": distance_list,
            "edge_type": edge_type_list,
            "type": torch.tensor([x["type"] for x in samples]),
            "labels": torch.stack([x["labels"] for x in samples]),
        }
        return batch


class MolRetrievalDataModule(LightningDataModule):
    def __init__(
        self,
        num_workers: int = 0,
        pin_memory: bool = False,
        batch_size: int = 32,
        dataset_cfg: Dict[str, Any] = None,
    ):
        super().__init__()

        self.save_hyperparameters(logger=False)

    def _dataloader(self, split: str):
        dataset_cfg = self.hparams["dataset_cfg"][split]
        dataset = MolRetrievalDataset(**dataset_cfg)
        if type(self.hparams["batch_size"]) is int:
            batch_size = self.hparams["batch_size"]
        else:
            batch_size = self.hparams["batch_size"][split]
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            num_workers=self.hparams["num_workers"],
            pin_memory=self.hparams["pin_memory"],
            collate_fn=dataset.collate_fn,
            shuffle=split == "train",
        )
        return dataloader

    def train_dataloader(self):
        return self._dataloader("train")

    def val_dataloader(self):
        return self._dataloader("val")


class MolRetrievalTestDataset(Dataset):
    def __init__(
        self,
        index_root: str,
        index_file: str,
        pocketdb_path: str,
        trie_root: str,
        pocket_key: str = "pocket.r6",
        max_pocket_atoms: int = 256,
        max_seq_len: int = 512,
        limit_samples: int = None,
        has_pocket_name: bool = False,
    ):
        self.pocketdb = LMDBDataset(pocketdb_path)
        self.pocket_key = pocket_key
        self.max_pocket_atoms = max_pocket_atoms
        self.max_seq_len = max_seq_len
        self.limit_samples = limit_samples
        self.has_pocket_name = has_pocket_name

        self.trie_root = trie_root

        self.index_root = Path(index_root)
        self.indices = list(self.index_root.glob(f"*/{index_file}"))

        self.tokenizer = QuantizeTokenizer()

    def __len__(self):
        if self.limit_samples is not None:
            return min(self.limit_samples, len(self.indices))
        return len(self.indices)

    def read_index(self, index_path):
        actives = []
        protein = None
        last_is_mol = False
        with open(index_path, "r") as f:
            for line in f:
                record = json.loads(line.strip())
                if "p" in record:
                    last_is_mol = False
                    if protein is None:
                        protein = self.pocketdb[record["p"]]
                    actives.append(record["i"])
                else:
                    if last_is_mol:  # skip inactives
                        break
                    last_is_mol = True
        return actives, protein

    def __getitem__(self, index):
        index_path = self.indices[index]
        actives, protein = self.read_index(index_path)
        pocket = protein[self.pocket_key]
        data = process_pocket(
            pocket["atoms"],
            pocket["coords"],
            max_atoms=self.max_pocket_atoms,
            max_seq_len=self.max_seq_len,
        )
        actives = np.array(
            [
                self.tokenizer.encode([int(x) for x in active.split(",")])
                for active in actives
            ]
        )
        data["type"] = 0  # pocket
        data["labels"] = actives
        if self.has_pocket_name:
            data["pocket_name"] = protein["pocket_name"]
            data["trie_path"] = os.path.join(
                self.trie_root, f'{protein["pocket_name"]}.pkl'
            )
        else:
            data["trie_path"] = os.path.join(
                self.trie_root, f'{str(index_path).split("/")[-2]}.pkl'
            )
        return data

    def collate_fn(self, batch):
        samples = [x for x in batch if x is not None]
        atoms_list = [x["atoms"] for x in samples]
        atoms_list = data_utils.collate_tokens(
            atoms_list, 0, left_pad=False, pad_to_multiple=8
        )  # pad is 0 in both mol_dict and pocket_dict
        distance_list = [x["distance"] for x in samples]
        distance_list = data_utils.collate_tokens_2d(
            distance_list, 0, left_pad=False, pad_to_multiple=8
        )
        edge_type_list = [x["edge_type"] for x in samples]
        edge_type_list = data_utils.collate_tokens_2d(
            edge_type_list, 0, left_pad=False, pad_to_multiple=8
        )
        max_n_labels = max([x["labels"].shape[0] for x in samples])
        labels = torch.full(
            (len(samples), max_n_labels, samples[0]["labels"].shape[1]),
            fill_value=-1,
        )
        for i, x in enumerate(samples):
            labels[i, : x["labels"].shape[0]] = torch.tensor(x["labels"])
        batch = {
            "atoms": atoms_list,
            "distance": distance_list,
            "edge_type": edge_type_list,
            "type": torch.tensor([x["type"] for x in samples]),
            "trie_path": [x["trie_path"] for x in samples],
            "labels": labels,
        }
        if self.has_pocket_name:
            batch["pocket_name"] = [x["pocket_name"] for x in samples]
        return batch


class MolRetrievalTestDataModule(LightningDataModule):
    def __init__(
        self,
        num_workers: int = 0,
        pin_memory: bool = False,
        batch_size: int = 32,
        dataset_cfg: Dict[str, Any] = None,
    ):
        super().__init__()

        self.save_hyperparameters(logger=False)

    def test_dataloader(self):
        dataset = MolRetrievalTestDataset(**self.hparams["dataset_cfg"])
        dataloader = DataLoader(
            dataset,
            batch_size=self.hparams["batch_size"],
            num_workers=self.hparams["num_workers"],
            pin_memory=self.hparams["pin_memory"],
            collate_fn=dataset.collate_fn,
            shuffle=False,
        )
        return dataloader
