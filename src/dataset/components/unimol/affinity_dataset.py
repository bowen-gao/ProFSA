# Copyright (c) DP Technology.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


from functools import lru_cache

import numpy as np
from unicore.data import BaseWrapperDataset

from . import data_utils


class AffinityDataset(BaseWrapperDataset):
    def __init__(
        self,
        dataset,
        seed,
        atoms,
        coordinates,
        pocket_atoms,
        pocket_coordinates,
        affinity,
        is_train=False,
        pocket="pocket",
    ):
        self.dataset = dataset
        self.seed = seed
        self.atoms = atoms
        self.coordinates = coordinates
        self.pocket_atoms = pocket_atoms
        self.pocket_coordinates = pocket_coordinates
        self.affinity = affinity
        self.is_train = is_train
        self.pocket = pocket
        self.set_epoch(None)

    def set_epoch(self, epoch, **unused):
        super().set_epoch(epoch)
        self.epoch = epoch

    def pocket_atom(self, atom):
        if atom[0] in ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]:
            return atom[1]
        else:
            return atom[0]

    @lru_cache(maxsize=16)
    def __cached_item__(self, index: int, epoch: int):
        atoms = np.array(self.dataset[index][self.atoms])
        ori_mol_length = len(atoms)
        # coordinates = self.dataset[index][self.coordinates]
        size = len(self.dataset[index][self.coordinates])
        if self.is_train:
            with data_utils.numpy_seed(self.seed, epoch, index):
                sample_idx = np.random.randint(size)
        else:
            with data_utils.numpy_seed(self.seed, 1, index):
                sample_idx = np.random.randint(size)
        # print(len(self.dataset[index][self.coordinates][sample_idx]))
        # check if the coordinates is 2D or 3D

        if len(np.array(self.dataset[index][self.coordinates]).shape) > 2:
            coordinates = self.dataset[index][self.coordinates][sample_idx]
        else:
            coordinates = self.dataset[index][self.coordinates]
        # coordinates = self.dataset[index][self.coordinates]
        pocket_atoms = np.array(
            [
                self.pocket_atom(item)
                for item in self.dataset[index][self.pocket_atoms]
            ]
        )
        ori_pocket_length = len(pocket_atoms)
        # print(len(self.dataset[index][self.pocket_coordinates]))
        pocket_coordinates = np.stack(
            self.dataset[index][self.pocket_coordinates]
        )
        if "smi" in self.dataset[index]:
            smi = self.dataset[index]["smi"]
        else:
            smi = ""
        if self.pocket not in self.dataset[index]:
            pocket = ""
        else:
            pocket = self.dataset[index][self.pocket]
        if self.affinity in self.dataset[index]:
            affinity = float(self.dataset[index][self.affinity])
        else:
            affinity = 1
        return {
            "atoms": atoms,
            "coordinates": coordinates.astype(np.float32),
            "holo_coordinates": coordinates.astype(np.float32),  # placeholder
            "pocket_atoms": pocket_atoms,
            "pocket_coordinates": pocket_coordinates.astype(np.float32),
            "holo_pocket_coordinates": pocket_coordinates.astype(
                np.float32
            ),  # placeholder
            "smi": smi,
            "pocket": pocket,
            "affinity": affinity,
            "ori_mol_length": ori_mol_length,
            "ori_pocket_length": ori_pocket_length,
        }

    def __getitem__(self, index: int):
        return self.__cached_item__(index, self.epoch)


class AffinityHNSDataset(BaseWrapperDataset):
    def __init__(
        self,
        dataset,
        seed,
        atoms,
        coordinates,
        atoms_hns,
        coordinates_hns,
        pocket_atoms,
        pocket_coordinates,
        affinity,
        is_train=False,
        pocket="pocket",
    ):
        self.dataset = dataset
        self.seed = seed
        self.atoms = atoms
        self.coordinates = coordinates
        self.atoms_hns = atoms_hns
        self.coordinates_hns = coordinates_hns
        self.pocket_atoms = pocket_atoms
        self.pocket_coordinates = pocket_coordinates
        self.affinity = affinity
        self.is_train = is_train
        self.pocket = pocket
        self.set_epoch(None)

    def set_epoch(self, epoch, **unused):
        super().set_epoch(epoch)
        self.epoch = epoch

    def pocket_atom(self, atom):
        if atom[0] in ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]:
            return atom[1]
        else:
            return atom[0]

    @lru_cache(maxsize=16)
    def __cached_item__(self, index: int, epoch: int):
        atoms = np.array(self.dataset[index][self.atoms])
        ori_mol_length = len(atoms)
        # coordinates = self.dataset[index][self.coordinates]
        size = len(self.dataset[index][self.coordinates])
        if self.is_train:
            with data_utils.numpy_seed(self.seed, epoch, index):
                sample_idx = np.random.randint(size)
        else:
            with data_utils.numpy_seed(self.seed, 1, index):
                sample_idx = np.random.randint(size)
        # print(len(self.dataset[index][self.coordinates][sample_idx]))
        coordinates = self.dataset[index][self.coordinates][sample_idx]
        atoms_hns = np.array(self.dataset[index][self.atoms_hns])
        coordinates_hns = self.dataset[index][self.coordinates_hns][0]

        pocket_atoms = np.array(
            [
                self.pocket_atom(item)
                for item in self.dataset[index][self.pocket_atoms]
            ]
        )
        ori_pocket_length = len(pocket_atoms)
        pocket_coordinates = np.stack(
            self.dataset[index][self.pocket_coordinates]
        )

        smi = self.dataset[index]["smi"]
        pocket = self.dataset[index][self.pocket]
        if self.affinity in self.dataset[index]:
            affinity = float(self.dataset[index][self.affinity])
        else:
            affinity = 1
        return {
            "atoms": atoms,
            "coordinates": coordinates.astype(np.float32),
            "atoms_hns": atoms_hns,
            "coordinates_hns": coordinates_hns.astype(np.float32),
            "holo_coordinates": coordinates.astype(np.float32),  # placeholder
            "pocket_atoms": pocket_atoms,
            "pocket_coordinates": pocket_coordinates.astype(np.float32),
            "holo_pocket_coordinates": pocket_coordinates.astype(
                np.float32
            ),  # placeholder
            "smi": smi,
            "pocket": pocket,
            "affinity": affinity,
            "ori_mol_length": ori_mol_length,
            "ori_pocket_length": ori_pocket_length,
        }

    def __getitem__(self, index: int):
        return self.__cached_item__(index, self.epoch)


class AffinityTestDataset(BaseWrapperDataset):
    def __init__(
        self,
        dataset,
        seed,
        atoms,
        coordinates,
        pocket_atoms,
        pocket_coordinates,
        affinity=None,
        is_train=False,
        pocket="pocket",
    ):
        self.dataset = dataset
        self.seed = seed
        self.atoms = atoms
        self.coordinates = coordinates
        self.pocket_atoms = pocket_atoms
        self.pocket_coordinates = pocket_coordinates
        self.affinity = affinity
        self.is_train = is_train
        self.pocket = pocket
        self.set_epoch(None)

    def set_epoch(self, epoch, **unused):
        super().set_epoch(epoch)
        self.epoch = epoch

    def pocket_atom(self, atom):
        if atom[0] in ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]:
            return atom[1]
        else:
            return atom[0]

    @lru_cache(maxsize=16)
    def __cached_item__(self, index: int, epoch: int):
        atoms = np.array(self.dataset[index][self.atoms])
        ori_length = len(atoms)
        # coordinates = self.dataset[index][self.coordinates]
        size = len(self.dataset[index][self.coordinates])
        with data_utils.numpy_seed(self.seed, epoch, index):
            sample_idx = np.random.randint(size)
        coordinates = self.dataset[index][self.coordinates][sample_idx]
        pocket_atoms = np.array(
            [
                self.pocket_atom(item)
                for item in self.dataset[index][self.pocket_atoms]
            ]
        )
        # print(len(self.dataset[index][self.pocket_coordinates]))
        pocket_coordinates = np.stack(
            self.dataset[index][self.pocket_coordinates]
        )

        smi = self.dataset[index]["smi"]
        pocket = self.dataset[index][self.pocket]
        affinity = self.dataset[index][self.affinity]
        return {
            "atoms": atoms,
            "coordinates": coordinates.astype(np.float32),
            "holo_coordinates": coordinates.astype(np.float32),  # placeholder
            "pocket_atoms": pocket_atoms,
            "pocket_coordinates": pocket_coordinates.astype(np.float32),
            "holo_pocket_coordinates": pocket_coordinates.astype(
                np.float32
            ),  # placeholder
            "smi": smi,
            "pocket": pocket,
            "affinity": affinity.astype(np.float32),
            "ori_length": ori_length,
        }

    def __getitem__(self, index: int):
        return self.__cached_item__(index, self.epoch)


class AffinityMolDataset(BaseWrapperDataset):
    def __init__(
        self,
        dataset,
        seed,
        atoms,
        coordinates,
        is_train=False,
    ):
        self.dataset = dataset
        self.seed = seed
        self.atoms = atoms
        self.coordinates = coordinates
        self.is_train = is_train
        self.set_epoch(None)

    def set_epoch(self, epoch, **unused):
        super().set_epoch(epoch)
        self.epoch = epoch

    def pocket_atom(self, atom):
        if atom[0] in ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]:
            return atom[1]
        else:
            return atom[0]

    @lru_cache(maxsize=16)
    def __cached_item__(self, index: int, epoch: int):
        # print(self.dataset[index].keys())
        atoms = np.array(self.dataset[index][self.atoms])
        ori_length = len(atoms)
        # coordinates = self.dataset[index][self.coordinates]
        size = len(self.dataset[index][self.coordinates])
        # check if size is 2d or 3d
        coordinates = self.dataset[index][self.coordinates]
        with data_utils.numpy_seed(self.seed, epoch, index):
            sample_idx = np.random.randint(size)
        if len(np.array(self.dataset[index][self.coordinates]).shape) > 2:
            coordinates = coordinates[sample_idx]

        # print(coordinates.shape)
        if "seq" in self.dataset[index]:
            seq = self.dataset[index]["seq"]
            seq = "".join(seq)
        else:
            seq = ""
        if "smi" in self.dataset[index]:
            smi = self.dataset[index]["smi"]
        else:
            smi = ""
        return {
            "atoms": atoms,
            "coordinates": coordinates.astype(np.float32),
            "holo_coordinates": coordinates.astype(np.float32),  # placeholder
            "smi": smi,
            "seq": seq,
            "ori_length": ori_length,
        }

    def __getitem__(self, index: int):
        return self.__cached_item__(index, self.epoch)


class AffinityPocketDataset(BaseWrapperDataset):
    def __init__(
        self,
        dataset,
        seed,
        pocket_atoms,
        pocket_coordinates,
        is_train=False,
        pocket="pocket",
    ):
        self.dataset = dataset
        self.seed = seed
        self.pocket_atoms = pocket_atoms
        self.pocket_coordinates = pocket_coordinates
        self.is_train = is_train
        self.pocket = pocket
        self.set_epoch(None)

    def set_epoch(self, epoch, **unused):
        super().set_epoch(epoch)
        self.epoch = epoch

    def pocket_atom(self, atom):
        if atom[0] in ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]:
            return atom[1]
        else:
            return atom[0]

    @lru_cache(maxsize=16)
    def __cached_item__(self, index: int, epoch: int):
        pocket_atoms = np.array(
            [
                self.pocket_atom(item)
                for item in self.dataset[index][self.pocket_atoms]
            ]
        )
        ori_length = len(pocket_atoms)
        pocket_coordinates = np.stack(
            self.dataset[index][self.pocket_coordinates]
        )
        if self.pocket not in self.dataset[index]:
            pocket = ""
        else:
            pocket = self.dataset[index][self.pocket]
        if "seq" in self.dataset[index]:
            seq = self.dataset[index]["seq"]
            seq = "".join(seq)
        else:
            seq = ""
        return {
            "pocket_atoms": pocket_atoms,
            "pocket_coordinates": pocket_coordinates.astype(np.float32),
            "holo_pocket_coordinates": pocket_coordinates.astype(
                np.float32
            ),  # placeholder
            "pocket": pocket,
            "seq": seq,
            "ori_length": ori_length,
        }

    def __getitem__(self, index: int):
        return self.__cached_item__(index, self.epoch)


class AffinityPocketMatchingDataset(BaseWrapperDataset):
    def __init__(
        self,
        dataset,
        seed,
        atoms_a,
        coordinates_a,
        atoms_b,
        coordinates_b,
        label,
        type_a,
        type_b,
    ):
        self.dataset = dataset
        self.seed = seed
        self.atoms_a = atoms_a
        self.coordinates_a = coordinates_a
        self.atoms_b = atoms_b
        self.coordinates_b = coordinates_b
        # self.pocket=pocket
        self.label = label
        self.set_epoch(None)
        self.type_a = type_a
        self.type_b = type_b

    def set_epoch(self, epoch, **unused):
        super().set_epoch(epoch)
        self.epoch = epoch

    def pocket_atom(self, atom):
        if atom[0] in ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]:
            return atom[1]
        else:
            return atom[0]

    @lru_cache(maxsize=16)
    def __cached_item__(self, index: int, epoch: int):
        atoms_a = np.array(
            [
                self.pocket_atom(item)
                for item in self.dataset[index][self.atoms_a]
            ]
        )
        # print(self.dataset[index][self.coordinates_a])
        coordinates_a = np.stack(self.dataset[index][self.coordinates_a])
        atoms_b = np.array(
            [
                self.pocket_atom(item)
                for item in self.dataset[index][self.atoms_b]
            ]
        )
        coordinates_b = np.stack(self.dataset[index][self.coordinates_b])
        label = 1 * self.dataset[index][self.label]
        if self.type_a in self.dataset[index]:
            type_a = self.dataset[index][self.type_a]
        else:
            type_a = ""
        if self.type_b in self.dataset[index]:
            type_b = self.dataset[index][self.type_b]
        else:
            type_b = ""
        return {
            "atoms_a": atoms_a,
            "coordinates_a": coordinates_a.astype(np.float32),
            "holo_pocket_coordinates_a": coordinates_a.astype(
                np.float32
            ),  # placeholder
            "atoms_b": atoms_b,
            "coordinates_b": coordinates_b.astype(np.float32),
            "holo_pocket_coordinates_b": coordinates_b.astype(
                np.float32
            ),  # placeholder
            "label": label,
            "type_a": type_a,
            "type_b": type_b,
        }

    def __getitem__(self, index: int):
        return self.__cached_item__(index, self.epoch)


class AffinityValidDataset(BaseWrapperDataset):
    def __init__(
        self,
        dataset,
        seed,
        atoms,
        coordinates,
        pocket_atoms,
        pocket_coordinates,
        pocket="pocket",
    ):
        self.dataset = dataset
        self.seed = seed
        self.atoms = atoms
        self.coordinates = coordinates
        self.pocket_atoms = pocket_atoms
        self.pocket_coordinates = pocket_coordinates
        self.pocket = pocket
        self.set_epoch(None)

    def set_epoch(self, epoch, **unused):
        super().set_epoch(epoch)
        self.epoch = epoch

    def pocket_atom(self, atom):
        if atom[0] in ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]:
            return atom[1]
        else:
            return atom[0]

    @lru_cache(maxsize=16)
    def __cached_item__(self, index: int, epoch: int):
        atoms = np.array(self.dataset[index][self.atoms])
        ori_mol_length = len(atoms)
        # coordinates = self.dataset[index][self.coordinates]

        size = len(self.dataset[index][self.coordinates])
        with data_utils.numpy_seed(self.seed, epoch, index):
            sample_idx = np.random.randint(size)
        coordinates = self.dataset[index][self.coordinates][sample_idx]
        pocket_atoms = np.array(
            [
                self.pocket_atom(item)
                for item in self.dataset[index][self.pocket_atoms]
            ]
        )
        ori_pocket_length = len(pocket_atoms)
        pocket_coordinates = np.stack(
            self.dataset[index][self.pocket_coordinates]
        )

        smi = self.dataset[index]["smi"]
        pocket = self.dataset[index][self.pocket]
        return {
            "atoms": atoms,
            "coordinates": coordinates.astype(np.float32),
            "holo_coordinates": coordinates.astype(np.float32),  # placeholder
            "pocket_atoms": pocket_atoms,
            "pocket_coordinates": pocket_coordinates.astype(np.float32),
            "holo_pocket_coordinates": pocket_coordinates.astype(
                np.float32
            ),  # placeholder
            "smi": smi,
            "pocket": pocket,
            "ori_mol_length": ori_mol_length,
            "ori_pocket_length": ori_pocket_length,
        }

    def __getitem__(self, index: int):
        return self.__cached_item__(index, self.epoch)


class PocketFTDataset(BaseWrapperDataset):
    def __init__(
        self,
        dataset,
        seed,
        pocket_atoms,
        pocket_coordinates,
        target,
        pocket="pdbid",
        is_train=False,
    ):
        self.dataset = dataset
        self.seed = seed
        self.pocket_atoms = pocket_atoms
        self.pocket_coordinates = pocket_coordinates
        self.target = target
        self.is_train = is_train
        self.pocket = pocket
        self.set_epoch(None)

    def set_epoch(self, epoch, **unused):
        super().set_epoch(epoch)
        self.epoch = epoch

    def pocket_atom(self, atom):
        if atom[0] in ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]:
            return atom[1]
        else:
            return atom[0]

    @lru_cache(maxsize=16)
    def __cached_item__(self, index: int, epoch: int):

        pocket_atoms = np.array(
            [
                self.pocket_atom(item)
                for item in self.dataset[index][self.pocket_atoms]
            ]
        )
        ori_pocket_length = len(pocket_atoms)
        pocket_coordinates = np.array(
            self.dataset[index][self.pocket_coordinates][0]
        )
        pocket = self.dataset[index][self.pocket]
        # target = float(self.dataset[index][self.target])
        # target = self.dataset[index][self.target]
        return {
            "pocket_atoms": pocket_atoms,
            "pocket_coordinates": pocket_coordinates.astype(np.float32),
            "holo_pocket_coordinates": pocket_coordinates.astype(np.float32),
            "pocket": pocket,
            # "target": target,
            "ori_pocket_length": ori_pocket_length,
        }

    def __getitem__(self, index: int):
        return self.__cached_item__(index, self.epoch)
