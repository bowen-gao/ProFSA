import logging
import pickle as pkl
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Union

import lmdb
import zstandard as zstd
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)

# 1T map_size, ref: https://lmdb.readthedocs.io/en/release/#environment-class
MAP_SIZE = 10 * 1024 * 1024 * 1024 * 1024  # 10T
BENCHMARK_SIZE = 2000


class UniMolLMDBDataset(Dataset):
    def __init__(self, db_path: Path, readonly: bool = True) -> None:
        super().__init__()

        self.db_path = db_path
        self.readonly = readonly
        self.keys = self.load_data(self.db_path)

    def load_data(self, db_path):
        assert Path(db_path).is_file(), "{} not found".format(db_path)
        self.env = lmdb.open(
            str(db_path),
            subdir=False,
            readonly=self.readonly,
            lock=not self.readonly,
            readahead=False,
            meminit=False,
            max_readers=256,
            map_size=MAP_SIZE,
        )
        with self.env.begin() as txn:
            keys = list(txn.cursor().iternext(values=False))
        return keys

    def close(self):
        self.env.close()

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, index) -> Any:
        sample = pkl.loads(self.env.begin().get(self.keys[index]))
        return sample

    def __setitem__(self, index, value) -> None:
        with self.env.begin(write=True) as txn:
            txn.put(self.keys[index], pkl.dumps(value))


class LMDBDataset:
    """
    split:
        full: "key1,key2,..."
        dataset1: "key1,key2,..."
        dataset2: "key2,key3,..."
        train: "key1,key2,..."
        val: "key3,key4,..."
    data:
        key1: pkl.dump({k1: v11, k2: v12, ...})
        key2: pkl.dump({k1: v21, k2: v22, ...})
    """

    SPLIT_DB = "split"
    DATA_DB = "data"
    MAX_CACHE_SIZE = 16

    def __init__(
        self,
        lmdb_path: Union[str, Path],
        compresed: bool = True,
        readonly: bool = True,
    ):
        self.lmdb_path = Path(lmdb_path).resolve()
        self.compresed = compresed
        self.readonly = readonly
        self.env = lmdb.open(
            str(self.lmdb_path),
            max_dbs=2,
            map_size=MAP_SIZE,
            readonly=readonly,
            lock=not readonly,
            create=not readonly,
        )
        self.db = {
            db_name: self.env.open_db(db_name.encode())
            for db_name in [self.SPLIT_DB, self.DATA_DB]
        }
        self._splits = dict()
        self.compresed = compresed
        self.default_split = "full"

    def compress(self, content: bytes, compress_level: int = 3) -> bytes:
        content = zstd.ZstdCompressor(level=compress_level).compress(content)
        return content

    def decompress(self, content: bytes) -> bytes:
        content = zstd.ZstdDecompressor().decompress(content)
        return content

    def _get_value(self, db: str, key: str, default: Any = None) -> Any:
        with self.env.begin(db=self.db[db], write=False) as txn:
            value = txn.get(key.encode())
            if value is not None:
                if self.compresed:
                    value = self.decompress(value)
            else:
                logger.warning(f"Key {key} not found in {db}")
                value = default
            return value

    def _set_value(self, db: str, key: str, value: bytes):
        with self.env.begin(db=self.db[db], write=True) as txn:
            if self.compresed:
                value = self.compress(value)
            txn.put(key.encode(), value)

    def get_split(self, key: str) -> List[str]:
        if key not in self._splits:
            split_str = self._get_value(self.SPLIT_DB, key)
            if split_str is None:
                return []
            else:
                self._splits[key] = split_str.decode().split(",")
        return self._splits[key]

    def set_split(
        self,
        split: str,
        keys: List[str],
        append: bool = False,
        deduplicate: bool = True,
        update_full: bool = True,
    ):
        if append:
            keys = self.get_split(split) + keys

        if deduplicate:
            keys = list(sorted(list(set(keys))))
        keys_str = ",".join(keys)
        self._set_value(self.SPLIT_DB, split, keys_str.encode())
        self._splits[split] = keys
        if update_full:
            self.update_full_split()

    def update_full_split(self):
        keys = []
        with self.env.begin(db=self.db[self.SPLIT_DB], write=False) as txn:
            splits = [
                key.decode() for key in txn.cursor().iternext(values=False)
            ]
        for split in splits:
            if split != "full":
                keys.extend(self.get_split(split))
                keys = list(sorted(list(set(keys))))
        if not self.readonly:
            self.set_split("full", keys, update_full=False)
        else:
            self._splits["full"] = keys

    def check_keys(self):
        with self.env.begin(db=self.db[self.DATA_DB], write=False) as txn:
            keys = [key.decode() for key in txn.cursor().iternext(values=False)]
        full_keys = self.get_split("full")

        missing_keys = list(set(full_keys) - set(keys))
        if len(missing_keys) > 0:
            print(f"Missing keys: {missing_keys}")

        orphan_keys = list(set(keys) - set(full_keys))
        if len(orphan_keys) > 0:
            print(f"Orphan keys: {orphan_keys}")

        if len(missing_keys) + len(orphan_keys) == 0:
            print("All keys are in place")

    @lru_cache(maxsize=MAX_CACHE_SIZE)
    def __getitem__(self, key: Union[str, int]) -> Dict[str, Any]:
        if type(key) == int:
            key = self.get_split(self.default_split)[key]
        data = self._get_value(self.DATA_DB, key)
        if data is not None:
            data = pkl.loads(data)
        return data

    def __setitem__(self, key: str, value: Dict[str, Any]) -> None:
        self._set_value(self.DATA_DB, key, pkl.dumps(value))

    def __contains__(self, key: str) -> bool:
        with self.env.begin(db=self.db[self.DATA_DB], write=False) as txn:
            return txn.get(key.encode()) is not None

    def set_default_split(self, split: str) -> None:
        self.default_split = split

    def __len__(self) -> int:
        keys = self.get_split(self.default_split)
        return len(keys)

    @property
    def summary(self) -> Dict[str, int]:
        self.update_full_split()
        return {split: len(keys) for split, keys in self._splits.items()}

    def __repr__(self) -> str:
        return f"LMDBDataset({self.lmdb_path})"

    def __iter__(self):
        self._iter_index = 0
        return self

    def __next__(self):
        if self._iter_index < len(self):
            result = self[self._iter_index]
            self._iter_index += 1
            return result
        else:
            raise StopIteration
