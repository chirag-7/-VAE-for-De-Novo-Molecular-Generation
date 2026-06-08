"""Distribution-learning metrics for generated molecule sets (MOSES-style).

All functions accept lists of SMILES strings. Validity-dependent metrics
operate on the canonicalized valid subset, matching the MOSES conventions.
"""

from __future__ import annotations

from collections.abc import Sequence

from rdkit import Chem
from rdkit.Chem import DataStructs, rdFingerprintGenerator
from rdkit.Chem.Scaffolds import MurckoScaffold

from molgen.chem import canonicalize_smiles
from molgen.properties import property_summary

_MORGAN = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)


def _canonical_valid(smiles_list: Sequence[str]) -> list[str]:
    """Return the canonical SMILES of the valid molecules in the list."""
    out = []
    for smiles in smiles_list:
        canonical = canonicalize_smiles(smiles)
        if canonical is not None:
            out.append(canonical)
    return out


def validity(smiles_list: Sequence[str]) -> float:
    """Fraction of generated strings that are valid molecules."""
    if not smiles_list:
        return 0.0
    return len(_canonical_valid(smiles_list)) / len(smiles_list)


def uniqueness(smiles_list: Sequence[str]) -> float:
    """Fraction of valid molecules that are unique (after canonicalization)."""
    valid = _canonical_valid(smiles_list)
    if not valid:
        return 0.0
    return len(set(valid)) / len(valid)


def novelty(smiles_list: Sequence[str], reference: Sequence[str]) -> float:
    """Fraction of unique valid molecules not present in the reference set."""
    generated = set(_canonical_valid(smiles_list))
    if not generated:
        return 0.0
    reference_set = set(_canonical_valid(reference))
    novel = [s for s in generated if s not in reference_set]
    return len(novel) / len(generated)


def _fingerprints(smiles_list: Sequence[str]):
    fps = []
    for smiles in smiles_list:
        mol = Chem.MolFromSmiles(smiles)
        if mol is not None:
            fps.append(_MORGAN.GetFingerprint(mol))
    return fps


def internal_diversity(smiles_list: Sequence[str]) -> float:
    """Mean pairwise Tanimoto *distance* (1 - similarity) over Morgan fingerprints.

    Higher means a more diverse set. Returns 0.0 for fewer than two molecules.
    """
    fps = _fingerprints(smiles_list)
    n = len(fps)
    if n < 2:
        return 0.0
    total, count = 0.0, 0
    for i in range(n - 1):
        sims = DataStructs.BulkTanimotoSimilarity(fps[i], fps[i + 1 :])
        total += float(sum(sims))
        count += len(sims)
    mean_similarity = total / count if count else 0.0
    return 1.0 - mean_similarity


def _scaffold(smiles: str) -> str | None:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return MurckoScaffold.MurckoScaffoldSmiles(mol=mol)


def unique_scaffolds(smiles_list: Sequence[str]) -> float:
    """Fraction of distinct Bemis-Murcko scaffolds among the valid molecules."""
    scaffolds = [s for s in (_scaffold(smi) for smi in smiles_list) if s is not None]
    if not scaffolds:
        return 0.0
    return len(set(scaffolds)) / len(scaffolds)


def snn(smiles_list: Sequence[str], reference: Sequence[str]) -> float:
    """Mean similarity-to-nearest-neighbour (Tanimoto) from generated to reference.

    For each generated molecule, take the maximum Tanimoto similarity to any
    reference molecule, then average. High SNN with low novelty can indicate
    memorisation of the training set.
    """
    gen_fps = _fingerprints(smiles_list)
    ref_fps = _fingerprints(reference)
    if not gen_fps or not ref_fps:
        return 0.0
    total = sum(max(DataStructs.BulkTanimotoSimilarity(fp, ref_fps)) for fp in gen_fps)
    return total / len(gen_fps)


def evaluate_generation(smiles_list: Sequence[str], reference: Sequence[str] | None = None) -> dict:
    """Compute a MOSES-style report for a set of generated SMILES.

    Includes validity, uniqueness, internal diversity, unique-scaffold fraction,
    and mean physico-chemical properties. When a ``reference`` (e.g. the training
    set) is given, novelty and SNN are added. Internal-diversity / SNN are O(n^2)
    in the set size, so pass a manageable sample for large runs.
    """
    report: dict = {
        "n_generated": len(smiles_list),
        "validity": validity(smiles_list),
        "uniqueness": uniqueness(smiles_list),
        "internal_diversity": internal_diversity(smiles_list),
        "unique_scaffolds": unique_scaffolds(smiles_list),
        "properties": property_summary(smiles_list),
    }
    if reference is not None:
        report["novelty"] = novelty(smiles_list, reference)
        report["snn"] = snn(smiles_list, reference)
    return report
