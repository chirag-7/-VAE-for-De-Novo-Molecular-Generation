"""Cheminformatics helpers built on RDKit.

Invalid SMILES are reported by returning ``None`` (or ``False``) rather than
raising, so callers can filter generated strings cheaply. RDKit's own parser
logging is silenced at import time since those errors are expected here.
"""

from __future__ import annotations

from rdkit import Chem, RDLogger

RDLogger.DisableLog("rdApp.*")

__all__ = ["is_valid_smiles", "canonicalize_smiles", "randomize_smiles"]


def is_valid_smiles(smiles: str) -> bool:
    """Return True if RDKit can parse ``smiles`` into a molecule."""
    return Chem.MolFromSmiles(smiles) is not None


def canonicalize_smiles(smiles: str) -> str | None:
    """Return the RDKit canonical SMILES for ``smiles``, or None if invalid."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return Chem.MolToSmiles(mol)


def randomize_smiles(smiles: str) -> str | None:
    """Return a random (non-canonical) SMILES for the same molecule.

    This is useful as data augmentation ("SMILES enumeration"): the same
    molecule is written with a different atom ordering each call. Returns
    None if the input cannot be parsed.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return Chem.MolToSmiles(mol, doRandom=True, canonical=False)
