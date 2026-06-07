"""Molecular property calculators: QED, logP, molecular weight, SA score.

Each function returns ``None`` for unparseable SMILES. Synthetic accessibility
uses the Ertl SA scorer that ships with RDKit's contrib modules; if it is
unavailable on a given install, ``sa_score`` returns ``None``.
"""

from __future__ import annotations

import os
import sys
from collections.abc import Sequence
from statistics import mean

from rdkit import Chem
from rdkit.Chem import QED, Crippen, Descriptors, RDConfig

try:
    sys.path.append(os.path.join(RDConfig.RDContribDir, "SA_Score"))
    import sascorer  # noqa: E402

    _HAS_SA = True
except Exception:  # pragma: no cover - depends on the RDKit build
    _HAS_SA = False


def _mol(smiles: str):
    return Chem.MolFromSmiles(smiles)


def qed(smiles: str) -> float | None:
    """Quantitative Estimate of Drug-likeness, in [0, 1]."""
    mol = _mol(smiles)
    return None if mol is None else QED.qed(mol)


def logp(smiles: str) -> float | None:
    """Crippen octanol-water partition coefficient (logP)."""
    mol = _mol(smiles)
    return None if mol is None else Crippen.MolLogP(mol)


def molecular_weight(smiles: str) -> float | None:
    """Average molecular weight."""
    mol = _mol(smiles)
    return None if mol is None else Descriptors.MolWt(mol)


def sa_score(smiles: str) -> float | None:
    """Ertl synthetic-accessibility score, ~1 (easy) to ~10 (hard)."""
    mol = _mol(smiles)
    if mol is None or not _HAS_SA:
        return None
    return sascorer.calculateScore(mol)


def property_summary(smiles_list: Sequence[str]) -> dict[str, float]:
    """Mean of each property over the valid molecules in the list."""
    calculators = {
        "qed": qed,
        "logp": logp,
        "mol_weight": molecular_weight,
        "sa_score": sa_score,
    }
    summary: dict[str, float] = {}
    for name, fn in calculators.items():
        values = [v for s in smiles_list if (v := fn(s)) is not None]
        summary[name] = mean(values) if values else float("nan")
    return summary
