from mordred import Calculator, descriptors
from rdkit import Chem


def extract_descriptors(smi: str, ignore_3D: bool = False):
    mol = Chem.MolFromSmiles(smi)
    calc = Calculator(descriptors, ignore_3D=ignore_3D)
    desc = calc(mol)
    return desc
