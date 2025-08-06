"""
ECFP (Extended Connectivity Fingerprints) utilities.

This module provides functions for generating ECFP fingerprints from molecular structures.
"""

import numpy as np
from typing import List
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem.AtomPairs import Pairs
from rdkit.Chem.AtomPairs import Torsions
from rdkit.Chem.rdMolDescriptors import GetMorganFingerprintAsBitVect
from rdkit import RDLogger
# Suppress RDKit warnings
lg = RDLogger.logger()
lg.setLevel(RDLogger.ERROR)

def generate_ecfp_fingerprints(
    smiles_list: List[str],
    radius: int = 2,
    n_bits: int = 2048,
    use_features: bool = False,
    use_chirality: bool = False
) -> np.array:
    """
    Generate ECFP fingerprints from SMILES strings.
    
    Args:
        smiles_list: List of SMILES strings
        radius: ECFP radius (default: 2)
        n_bits: Number of bits in fingerprint (default: 2048)
        use_features: Whether to use RDKit features (default: False)
        use_chirality: Whether to use chirality information (default: False)
    
    Returns:
        numpy.ndarray: Array of fingerprints with shape (n_molecules, n_bits)
    """
    fingerprints = []
    
    for smiles in smiles_list:
        mol = Chem.MolFromSmiles(smiles, sanitize=True)
        if mol is None:
            # If molecule is invalid, create zero fingerprint
            fp = np.zeros(n_bits, dtype=np.float32)
        else:
            # Use the new MorganGenerator to avoid deprecation warning
            fp = GetMorganFingerprintAsBitVect(mol, radius=radius, nBits=n_bits)
            fp = np.array(fp, dtype=np.float32)
        
        fingerprints.append(fp)
    
    return np.array(fingerprints)