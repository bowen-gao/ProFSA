import gzip
import itertools
import json
import logging
import multiprocessing as mp
import sys
import time
from pathlib import Path, PosixPath
from typing import Any, Dict, List, Union

import numpy as np
import ray
from biopandas.mol2 import PandasMol2
from mordred import Calculator, descriptors
from ray.experimental.tqdm_ray import tqdm as tqdm_ray
from ray.util.queue import Queue
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem.rdchem import Mol
from tqdm import tqdm

sys.path.append(".")  # noqa: E402
from src.dataset.components.lmdb import LMDBDataset  # noqa: E402

logger = logging.getLogger(__name__)


ERROR_LOG = "error.log"
DEFAULT_THREADS = mp.cpu_count()
DEFAULT_CHUNK_SIZE = DEFAULT_THREADS * 4


def record_error(error_log, message, mol=None, src=None, **kwargs):
    message = str(message)
    with open(error_log, "a") as f:
        record = {
            "error": f"{message}",
        }
        if mol is not None:
            record["smi"] = Chem.MolToSmiles(mol)
        if src is not None:
            record["src"] = str(src)
        f.write(json.dumps(record) + "\n")
    if src is not None:
        message = f"{message} (source: {src})"
    return message


def parse_mol(
    input_mol: Union[str, PosixPath, Mol], rm_conformer: bool = False
) -> Union[Mol, List[Mol]]:
    if type(input_mol) == Mol:
        return input_mol
    elif type(input_mol) == str:
        if Path(input_mol).exists():
            return parse_mol(Path(input_mol))
        else:
            mol = Chem.MolFromSmiles(input_mol)
    elif type(input_mol) == PosixPath:
        if input_mol.suffix == ".sdf":
            suppl = Chem.SDMolSupplier(str(input_mol))
            mol = [m for m in suppl]
        elif input_mol.suffix == ".sdf.gz":
            data = gzip.open(input_mol, "rb")
            with Chem.ForwardSDMolSupplier(data) as suppl:
                mol = [m for m in suppl]
        elif input_mol.suffix == ".mol2":
            mol = Chem.MolFromMol2File(str(input_mol))
        elif input_mol.suffix == ".smi" or input_mol.suffix == ".ism":
            lines = input_mol.read_text().strip().split("\n")
            mol = [Chem.MolFromSmiles(line) for line in lines]
        elif input_mol.suffix == ".pdb":
            # only support specillay formatted BioLip pdb file now
            het_id = input_mol.stem.split("_")[-1]
            with open(input_mol) as f:
                lines = f.readlines()
                lines = [line for line in lines if het_id in line]
            pdb_block = "".join(lines)
            mol = Chem.MolFromPDBBlock(pdb_block)
        else:
            raise ValueError(f"Unsupported file type: {input_mol.suffix}")
    else:
        raise ValueError(f"Unsupported input type: {type(input_mol)}")
    if type(mol) == list:
        mol = [m for m in mol if m is not None]
        if rm_conformer:
            mol = [m.RemoveAllConformers() for m in mol]
        if len(mol) == 1:
            mol = mol[0]
        if len(mol) == 0:
            mol = None
        mol = [Chem.RemoveHs(m) for m in mol]
    else:
        if rm_conformer:
            mol.RemoveAllConformers()
        mol = Chem.RemoveHs(mol)
    return mol


def get_inchikey(mol: Union[str, Mol]):
    try:
        if type(mol) == str:
            mol = Chem.MolFromSmiles(mol)
        inchi_key = Chem.MolToInchiKey(mol)
    except Exception as e:
        raise Exception(f"Failed to get InChI key: {e}")

    if inchi_key is None or inchi_key == "":
        raise Exception("Empty InChI key")
    return inchi_key


def gen_conformation(mol, num_conf=1, num_worker=5):
    try:
        mol = Chem.AddHs(mol)
        AllChem.EmbedMultipleConfs(
            mol,
            numConfs=num_conf,
            numThreads=num_worker,
            pruneRmsThresh=1,
            maxAttempts=10000,
            useRandomCoords=False,
        )
        try:
            AllChem.MMFFOptimizeMoleculeConfs(mol, numThreads=num_worker)
        except Exception:
            pass
        mol = Chem.RemoveHs(mol)
    except Exception as e:
        raise Exception(f"Failed to generate conformation: {e}")
    if mol.GetNumConformers() == 0:
        raise Exception("Cannot generate conformation")
    return mol


def get_3d_data(mol: Mol):
    if mol.GetNumConformers() == 0:
        mol = gen_conformation(mol)
    try:
        coords = np.array(mol.GetConformer(0).GetPositions())
        atom_types = [a.GetSymbol() for a in mol.GetAtoms()]
        res = {"coords": coords, "atom_types": atom_types}
    except Exception as e:
        raise Exception(f"Failed to get 3D data: {e}")
    return res


def get_3d_data_from_mol2(path: str):
    try:
        mol2_df = PandasMol2().read_mol2(str(path))
        coords = mol2_df.df[["x", "y", "z"]]
    except Exception:
        mol = Chem.MolFromMol2File(str(path), removeHs=True, sanitize=False)
        coords = mol.GetConformer().GetPositions()
    return {"coords": np.array(coords)}


def get_morgan_fingerprint(mol: Mol, radius: int = 3, n_bits: int = 2048):
    try:
        fp = AllChem.GetHashedMorganFingerprint(mol, radius, nBits=n_bits)
        res = np.array(fp.ToList(), dtype=np.uint8)
    except Exception as e:
        raise Exception(f"Failed to get fingerprint: {e}")
    return res


def get_descriptor(mol: Mol):
    try:
        calc = Calculator(descriptors)
        desc = calc(mol)
        res = np.array(list(desc.fill_missing()), dtype=np.float32)
    except Exception as e:
        raise Exception(f"Failed to get descriptor: {e}")
    return res


def inchi_key_exists(
    inchi_key: str, dataset: LMDBDataset, smi: str, raise_error: bool = True
):
    if inchi_key in dataset:
        if raise_error and dataset[inchi_key]["smi"] != smi:
            raise ValueError(
                f"Duplicate InChI key ({inchi_key}) with different SMILES: "
                f"{dataset[inchi_key]['smi']}"
            )
        return True
    return False


def process_sample(sample, dataset, error_log):
    try:
        mol = sample["mol"]
        if mol is None:
            return

        smi = Chem.MolToSmiles(mol)
        inchi_key = get_inchikey(mol)
        if inchi_key_exists(inchi_key, dataset, smi, raise_error=False):
            return (inchi_key, None)
        three_d_data = get_3d_data(mol)
        fp = get_morgan_fingerprint(mol)
        # desc = get_descriptor(mol)
        return (
            inchi_key,
            {
                "smi": smi,
                "fp": fp,
                # "desc": desc,
                **three_d_data,
                **{k: v for k, v in sample.items() if k != "mol"},
            },
        )
    except Exception as e:
        message = record_error(error_log, e, **sample)
        logger.error(message)


@ray.remote
class StopSignal:
    def __init__(self):
        self.read_stop = False
        self.process_stop = False

    def set_read_stop(self):
        self.read_stop = True

    def set_process_stop(self):
        self.process_stop = True

    def is_read_stop(self):
        return self.read_stop

    def is_process_stop(self):
        return self.process_stop


@ray.remote
def fill_queue(sample_queue, read_sample_func, stop_actor):
    count = 0
    for sample in read_sample_func():
        count += 1
        sample_queue.put(sample)
    print(f"Finished reading {count} samples")
    stop_actor.set_read_stop.remote()


@ray.remote
def process_sample_ray(
    sample_queue, result_queue, lmdb_path, error_log, stop_actor
):
    # waiting for creating lmdb files, or it will raise lmdb error
    while True:
        time.sleep(3)
        if Path(lmdb_path).exists():
            break

    dataset = LMDBDataset(lmdb_path=lmdb_path)
    stop = False
    while not stop:
        try:
            sample = sample_queue.get(timeout=1)
            if sample is None:
                continue
            result = process_sample(sample, dataset, error_log)
            if result is not None:
                result_queue.put(result)
        except Exception:
            stop = ray.get(stop_actor.is_read_stop.remote())


@ray.remote
def write_sample_ray(
    result_queue, lmdb_path, split_name, error_log, stop_actor
):
    Path(lmdb_path).parent.mkdir(parents=True, exist_ok=True)
    dataset = LMDBDataset(lmdb_path=lmdb_path, readonly=False)
    split_keys = []
    count = 0
    UPDATE_INTERVAL = 1000
    pbar = tqdm_ray()
    stop = False
    while not stop:
        try:
            result = result_queue.get(timeout=1)
            if result is None:
                continue

            count += 1
            pbar.update(1)

            inchi_key, value = result
            split_keys.append(inchi_key)
            if count % UPDATE_INTERVAL == 0:
                dataset.set_split(split_name, split_keys)

            try:
                if value is not None and not inchi_key_exists(
                    inchi_key, dataset, value["smi"]
                ):
                    dataset[inchi_key] = value
            except Exception as e:
                message = record_error(error_log, e, **value)
                logger.error(message)
        except Exception:
            stop = ray.get(stop_actor.is_process_stop.remote())

    dataset.set_split(split_name, split_keys)

    print(f"Finished writing {count} samples")
    print(f"Dataset path: {lmdb_path}")
    print(f"Error log: {error_log}")
    print(f"Summary: {dataset.summary}")


def write_smiles_to_lmdb_ray(
    lmdb_path: Union[str, Path],
    split_name: str,
    read_sample_func: callable,
    error_log: Union[str, Path] = None,
    threads: int = DEFAULT_THREADS,
):
    """
    sample: {
        mol: Mol,
        other_keys: Any,
    }
    """
    ray.init()
    stop_actor = StopSignal.remote()
    sample_queue = Queue(maxsize=threads * 32)
    result_queue = Queue()
    read_job = fill_queue.remote(sample_queue, read_sample_func, stop_actor)
    process_jobs = []
    for _ in range(threads):
        process_jobs.append(
            process_sample_ray.remote(
                sample_queue, result_queue, lmdb_path, error_log, stop_actor
            )
        )
    write_job = write_sample_ray.remote(
        result_queue, lmdb_path, split_name, error_log, stop_actor
    )
    ray.get([read_job, *process_jobs])
    stop_actor.set_process_stop.remote()
    ray.get(write_job)
    ray.shutdown()


def chunk_iter(iterable, chunk_size):
    iterator = iter(iterable)
    while True:
        chunk = list(itertools.islice(iterator, chunk_size))
        if not chunk:
            return
        yield chunk


def write_smiles_to_lmdb(
    lmdb_path: Union[str, Path],
    split_name: str,
    samples: List[Dict[str, Any]],
    error_log: Union[str, Path] = None,
    threads: int = DEFAULT_THREADS,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
):
    """
    sample: {
        mol: Mol,
        other_keys: Any,
    }
    """
    Path(lmdb_path).parent.mkdir(parents=True, exist_ok=True)
    dataset = LMDBDataset(lmdb_path=lmdb_path, readonly=False)

    if error_log is None:
        error_log = Path(lmdb_path).parent / f"{split_name}_error.log"

    split_keys = []
    count = 0
    try:
        pbar = tqdm()
        for chunk in chunk_iter(samples, chunk_size):
            with mp.Pool(threads) as pool:
                processed_data = pool.starmap(
                    process_sample,
                    zip(
                        chunk,
                        itertools.repeat(lmdb_path),
                        itertools.repeat(error_log),
                    ),
                )
                for result in processed_data:
                    if result is None:
                        continue
                    inchi_key, value = result
                    count += 1
                    pbar.update(1)
                    split_keys.append(inchi_key)
                    try:
                        if value is not None and not inchi_key_exists(
                            inchi_key, dataset, value["smi"]
                        ):
                            dataset[inchi_key] = value
                    except Exception as e:
                        message = record_error(error_log, e, **value)
                        logger.error(message)
            dataset.set_split(split_name, split_keys)
    except Exception as e:
        logger.exception(e)
        raise
    finally:
        dataset.set_split(split_name, split_keys)

        logger.info(f"Dataset path: {lmdb_path}")
        logger.info(f"Error log: {error_log}")
        logger.info(f"Processed {count} SMILES")
        logger.info(f"Added {len(split_keys)} SMILES to {split_name}")
        logger.info(f"Summary: {dataset.summary}")
