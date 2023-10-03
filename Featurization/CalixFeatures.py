"""
Code for creating a variety of benchmarks for calixarene predictions
"""

import rdkit
from rdkit import Chem
from rdkit.Chem import AllChem

def create_ecfp6_fingerprint(smiles_string):
    """
    Create the ECFP6 fingerprint for a given smiles string
    """
    mol = Chem.MolFromSmiles(smiles_string)
    fpgen = AllChem.GetMorganGenerator(radius=3)
    ecfp = fpgen.GetFingerprint(mol)
    return ecfp



