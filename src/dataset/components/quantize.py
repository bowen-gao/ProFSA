import logging
import time
from typing import List

import faiss
import nanopq
import numpy as np
import pandas as pd
from tqdm import tqdm

logger = logging.getLogger(__name__)


class Quantizer:
    """
    - d: dimensionality of the input vectors
    - M: number of subquantizers
    - nbits: number of bit per subvector index, faiss only supprot 8 bits (256)
    """

    nbits = 8

    def __init__(self, pretrain: str = None):
        self.pretrain = pretrain
        self.index = None
        self.nanopq = None
        self.type = None
        if self.pretrain is not None:
            self.load(self.pretrain)

    def init_pq(self, d: int, M: int):
        self.index = faiss.IndexPQ(d, M, self.nbits)
        self.type = "pq"
        return self

    def init_nanopq(self, d: int, M: int, opq: bool = False):
        if opq:
            self.nanopq = nanopq.OPQ(M, Ks=2**self.nbits, verbose=False)
            self.type = "nanoopq"
            self.nanopq.pq.Ds = d * M
        else:
            self.nanopq = nanopq.PQ(M, Ks=2**self.nbits, verbose=False)
            self.type = "nanopq"
            self.nanopq.Ds = d * M
        return self

    def init_rq(self, d: int, M: int):
        self.index = faiss.IndexResidualQuantizer(d, M, self.nbits)
        self.type = "rq"
        return self

    @property
    def quantizer(self):
        assert self.index is not None, "Index is not initialized"
        if self.type == "pq":
            return self.index.pq
        elif self.type == "rq":
            return self.index.rq
        else:
            raise ValueError(f"Invalid quantizer type: {self.type}")

    @property
    def input_dim(self):
        return self.index.d

    @property
    def code_size(self):
        return self.index.code_size

    @property
    def code_bits(self):
        return self.quantizer.nbits

    def __repr__(self):
        return (
            f"Quantizer({self.type}) "
            f"<{self.input_dim} (fp32) -> {self.code_size} (uint8)>"
        )

    def to_gpu(self, ngpu: int = 4):
        self.index = faiss.index_cpu_to_gpus_list(self.index, ngpu=ngpu)
        return self

    def to_cpu(self):
        self.index = faiss.index_gpu_to_cpu(self.index)
        return self

    def _check(self, data: np.ndarray, encode: bool = True):
        if "nano" in self.type:
            assert (
                self.index is not None
            ), "You must train nanopq first to get transformed index"
        else:
            assert self.index is not None, "Index is not initialized"
        if encode:
            assert data.shape[-1] == self.input_dim, (
                "Data dimension does not match input dimension: "
                f"{data.shape[-1]} != {self.input_dim}"
            )
        else:
            assert data.shape[-1] == self.code_size, (
                "Data dimension does not match code size: "
                f"{data.shape[-1]} != {self.code_size}"
            )

    def train(self, data: np.ndarray):
        if "nano" in self.type:
            self.nanopq.fit(data)
            if self.type == "nanoopq":
                self.index = nanopq.nanopq_to_faiss(self.nanopq.pq)
            else:
                self.index = nanopq.nanopq_to_faiss(self.nanopq)
        else:
            self._check(data)
            self.index.train(data)
        return self

    def encode(self, data: np.ndarray):
        self._check(data)
        if data.ndim == 1:
            data = data[None, ...]
        return self.index.sa_encode(data).squeeze()

    def decode(self, data: np.ndarray):
        self._check(data, encode=False)
        if data.ndim == 1:
            data = data[None, ...]
        return self.index.sa_decode(data).squeeze()

    def save(self, path: str):
        self.to_cpu()
        faiss.write_index(self.index, str(path))

    def load(self, path: str):
        self.index = faiss.read_index(path)
        index_name = self.index.__class__.__name__
        if "PQ" in index_name:
            self.type = "pq"
        elif "ResidualQuantizer" in index_name:
            self.type = "rq"
        else:
            raise ValueError(f"Unsupported index type: {type(self.index)}")
        logger.info(f"Load {index_name} from {path}.")
        return self

    def renconstrcut(self, data: np.ndarray):
        codes = self.encode(data)
        recon = self.decode(codes)
        return recon

    def reconstruct_rmse(self, data: np.ndarray):
        recon = self.renconstrcut(data)
        return np.sqrt(np.mean((data - recon) ** 2))


def benchmark_quantizer(
    quantizer: Quantizer, data: np.ndarray, test_data: np.ndarray = None
):
    start = time.time()
    quantizer.train(data)
    train_time = time.time() - start

    if test_data is None:
        test_data = data

    start = time.time()
    codes = quantizer.encode(test_data)
    encode_time = time.time() - start

    start = time.time()
    recon = quantizer.decode(codes)
    decode_time = time.time() - start

    rmse = np.sqrt(np.mean((test_data - recon) ** 2))

    return {
        "type": quantizer.type,
        "input_dim": quantizer.input_dim,
        "code_size": quantizer.code_size,
        "train_time": train_time,
        "encode_time": encode_time,
        "decode_time": decode_time,
        "rmse": rmse,
    }


def benchmark_quantizers(
    quantizers: List[Quantizer], data: np.ndarray, plot: bool = False
):
    results = []
    for quantizer in tqdm(quantizers, ncols=80):
        results.append(benchmark_quantizer(quantizer, data))
    df = pd.DataFrame(results)
    return df
