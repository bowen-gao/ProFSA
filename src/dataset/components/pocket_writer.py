import itertools
import json
import logging
import multiprocessing as mp
import sys

import numpy as np
import ray
from biopandas.mol2 import PandasMol2
from biopandas.pdb import PandasPdb
from ray.experimental.tqdm_ray import tqdm as tqdm_ray
from ray.util.queue import Queue
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem.rdchem import Mol
from scipy.spatial import distance_matrix
from tqdm import tqdm

from .mol_writer import get_inchikey

sys.path.append(".")  # noqa: E402
from src.dataset.components.lmdb import LMDBDataset  # noqa: E402

logger = logging.getLogger(__name__)


ERROR_LOG = "error.log"
DEFAULT_THREADS = mp.cpu_count()


def record_error(error_log, message, src=None, **kwargs):
    message = str(message)
    with open(error_log, "a") as f:
        record = {
            "error": f"{message}",
        }
        if src is not None:
            record["src"] = str(src)
        f.write(json.dumps(record) + "\n")
    if src is not None:
        message = f"{message} (source: {src})"
    return message


def get_protein_from_pdb(path: str):
    pdbdf = PandasPdb().read_pdb(path)
    result = {}
    protein = pdbdf.df["ATOM"]
    result["coords"] = protein[["x_coord", "y_coord", "z_coord"]].to_numpy()
    result["atoms"] = protein["atom_name"].tolist()
    result["residues"] = protein["residue_name"].tolist()
    result["residue_names"] = (
        protein["chain_id"] + protein["residue_number"].astype(str)
    ).tolist()
    return result


def get_protein_from_mol2(path: str):
    try:
        mol2_df = PandasMol2().read_mol2(
            str(path),
            columns={
                0: ("atom_id", int),
                1: ("atom_name", str),
                2: ("x", float),
                3: ("y", float),
                4: ("z", float),
                5: ("atom_type", str),
                6: ("subst_id", int),
                7: ("residue_name", str),
                8: ("useless1", float),
                9: ("useless2", str),
            },
        )
    except Exception:
        mol2_df = PandasMol2().read_mol2(
            str(path),
            columns={
                0: ("atom_id", int),
                1: ("atom_name", str),
                2: ("x", float),
                3: ("y", float),
                4: ("z", float),
                5: ("atom_type", str),
                6: ("subst_id", int),
                7: ("residue_name", str),
                8: ("useless1", float),
                9: ("useless2", str),
                10: ("useless3", str),
            },
        )
    coord = mol2_df.df[["x", "y", "z"]]
    atom_type = mol2_df.df["atom_type"]
    residue_data = mol2_df.df["residue_name"]
    residue_type = [res[:3] for res in list(residue_data)]
    protein = {
        "coords": np.array(coord),
        "atoms": list(atom_type),
        "residues": list(residue_type),
        "residue_names": list(residue_data),
    }
    return protein


def extract_pocket(protein, ligand, radius: int = 6):
    protein_coord = protein["coords"]
    ligand_coord = ligand["coords"]
    distance = distance_matrix(protein_coord, ligand_coord)
    pocket_idx = np.where(np.sum(distance < radius, axis=1) > 0)[0]
    res = {
        "coords": protein_coord[pocket_idx],
        "atoms": [protein["atoms"][i] for i in pocket_idx],
        "residues": [protein["residues"][i] for i in pocket_idx],
        "residue_names": [protein["residue_names"][i] for i in pocket_idx],
    }
    return res
