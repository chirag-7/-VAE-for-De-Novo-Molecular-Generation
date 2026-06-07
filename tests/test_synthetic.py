"""Tests for the synthetic dataset generator."""

import random

from rdkit import Chem

from molgen.synthetic import generate_fragment_molecule, generate_molecule


def test_generated_molecules_are_valid():
    random.seed(0)
    for _ in range(50):
        smiles = generate_molecule(min_carbon=5, max_carbon=12)
        assert Chem.MolFromSmiles(smiles) is not None, smiles


def test_backbone_respects_minimum_carbons():
    random.seed(1)
    smiles = generate_molecule(min_carbon=8, max_carbon=8)
    mol = Chem.MolFromSmiles(smiles)
    n_carbons = sum(1 for atom in mol.GetAtoms() if atom.GetSymbol() == "C")
    # 8-carbon backbone plus optional branches -> at least 8 carbons.
    assert n_carbons >= 8


def test_generation_is_reproducible_with_seed():
    random.seed(123)
    first = generate_molecule(min_carbon=6, max_carbon=10)
    random.seed(123)
    second = generate_molecule(min_carbon=6, max_carbon=10)
    assert first == second


def test_fragment_molecules_are_valid():
    random.seed(0)
    for _ in range(30):
        smiles = generate_fragment_molecule()
        assert Chem.MolFromSmiles(smiles) is not None, smiles


def test_fragment_generator_produces_rings_and_heteroatoms():
    from rdkit.Chem import rdMolDescriptors

    random.seed(1)
    mols = [generate_fragment_molecule() for _ in range(200)]
    has_ring = any(rdMolDescriptors.CalcNumRings(Chem.MolFromSmiles(m)) > 0 for m in mols)
    has_hetero = any(
        any(atom.GetSymbol() not in ("C", "H") for atom in Chem.MolFromSmiles(m).GetAtoms())
        for m in mols
    )
    assert has_ring
    assert has_hetero
    assert len(set(mols)) > 20  # the generator is reasonably diverse
