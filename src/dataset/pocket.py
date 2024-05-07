import logging
from pathlib import Path
from typing import Any, Dict, Union

import numpy as np
from lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset
from unicore.data import (
    AppendTokenDataset,
    Dictionary,
    FromNumpyDataset,
    LMDBDataset,
    NestedDictionaryDataset,
    PrependTokenDataset,
    RawArrayDataset,
    RightPadDataset,
    RightPadDataset2D,
    TokenizeDataset,
)

from src.utils.torchtool import is_rank_zero

from .components.unimol import (
    AffinityPocketDataset,
    CroppingPocketDockingPoseTestDataset,
    DistanceDataset,
    EdgeTypeDataset,
    KeyDataset,
    LengthDataset,
    NormalizeDataset,
    PrependAndAppend2DDataset,
    RemoveHydrogenPocketDataset,
    RightPadDatasetCoord,
)

logger = logging.getLogger(__name__)


class PocketDataset(Dataset):
    def __init__(
        self,
        data_path: str,
        dict_dir: str,
        pocket_dict_file: str = "dict_pocket.txt",
        label_path: str = None,
        max_pocket_atoms: int = 256,
        max_seq_len: int = 512,
        seed: int = 0,
    ):
        self.data_path = data_path
        self.dict_dir = Path(dict_dir)
        self.pocket_dict_path = str(self.dict_dir / pocket_dict_file)
        self.label_path = label_path
        self.max_pocket_atoms = max_pocket_atoms
        self.max_seq_len = max_seq_len
        self.seed = seed

        self.pocket_dict = self.load_dictionary()
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
        pocket_dict = Dictionary.load(self.pocket_dict_path)
        pocket_dict.add_symbol("[MASK]", is_special=True)
        if is_rank_zero():
            logger.info("pocket dictionary: {} types".format(len(pocket_dict)))
        return pocket_dict

    def lmdb2dict(self, lmdb_dataset, atoms_key, coordinates_key):
        dataset = AffinityPocketDataset(
            lmdb_dataset,
            self.seed,
            atoms_key,
            coordinates_key,
            False,
            "pocket",
        )

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
        dataset = CroppingPocketDockingPoseTestDataset(
            dataset,
            self.seed,
            "pocket_atoms",
            "pocket_coordinates",
            "holo_pocket_coordinates",
            self.max_pocket_atoms,
        )

        apo_dataset = NormalizeDataset(dataset, "pocket_coordinates")
        src_pocket_dataset = KeyDataset(apo_dataset, "pocket_atoms")
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

        return {
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
        }

    def load_dataset(self):
        dataset = LMDBDataset(self.data_path)
        dataset_dict = self.lmdb2dict(
            dataset, "pocket_atoms", "pocket_coordinates"
        )
        poc_dataset = KeyDataset(dataset, "pocket")
        nest_dataset = NestedDictionaryDataset(
            {
                "net_input": dataset_dict,
                "pocket_name": RawArrayDataset(poc_dataset),
            },
        )
        return nest_dataset


class PairPocketDataset(PocketDataset):
    def load_dataset(self):
        dataset = LMDBDataset(self.data_path)
        label_dataset = KeyDataset(dataset, "label")
        pocket_a_dataset_dict = self.lmdb2dict(
            dataset, "atoms_a", "coordinates_a"
        )
        pocket_b_dataset_dict = self.lmdb2dict(
            dataset, "atoms_b", "coordinates_b"
        )
        return NestedDictionaryDataset(
            {
                "pocket_a": pocket_a_dataset_dict,
                "pocket_b": pocket_b_dataset_dict,
                "label": RawArrayDataset(label_dataset),
            },
        )


class PocketDataModule(LightningDataModule):
    def __init__(
        self,
        num_workers: int = 0,
        pin_memory: bool = False,
        batch_size: Union[int, Dict[str, int]] = None,
        dataset_cfg: Dict[str, Any] = None,
        dataset_type: str = "pocket",
    ):
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        self.save_hyperparameters(logger=False)

    def test_dataloader(self):
        dataset_cfg = self.hparams["dataset_cfg"]
        if self.hparams["dataset_type"] == "pocket":
            dataset = PocketDataset(**dataset_cfg)
        elif self.hparams["dataset_type"] == "pair_pocket":
            dataset = PairPocketDataset(**dataset_cfg)
        else:
            raise NotImplementedError(
                f"dataset_type: {self.hparams['dataset_type']} "
                "is not implemented."
            )
        dataloader = DataLoader(
            dataset,
            collate_fn=dataset.dataset.collater,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
        )
        return dataloader
