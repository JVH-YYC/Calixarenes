"""
Code for creating a variety of benchmarks for calixarene predictions
"""

import rdkit
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem

def create_ecfp6_fingerprint(smiles_string):
    """
    Create the ECFP6 fingerprint for a given smiles string
    
    Converted into a numpy array for flexibility
    """
    mol = Chem.MolFromSmiles(smiles_string)
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=3)
    ecfp = np.array(fp)
    
    return ecfp



