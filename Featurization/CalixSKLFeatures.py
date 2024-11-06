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

def create_double_ecpf6_fingerprint(smiles_tuple,
                                    method):
    """
    Takes a tuple with two SMILES strings and returns a concatenated ECFP6 fingerprint if the mode is 'concat',
    or returns a difference of the two fingerprints if the mode is 'diff'
    """
    
    mol1 = Chem.MolFromSmiles(smiles_tuple[0])
    mol2 = Chem.MolFromSmiles(smiles_tuple[1])
    
    fp1 = AllChem.GetMorganFingerprintAsBitVect(mol1, radius=3)
    fp2 = AllChem.GetMorganFingerprintAsBitVect(mol2, radius=3)
    
    ecfp1 = np.array(fp1)
    ecfp2 = np.array(fp2)
    
    if method == 'concat':
        double_ecfp = np.concatenate((ecfp1, ecfp2))
    elif method == 'diff':
        double_ecfp = ecfp1 - ecfp2
    else:
        raise ValueError('Mode must be either "concat" or "diff"')

    return double_ecfp




