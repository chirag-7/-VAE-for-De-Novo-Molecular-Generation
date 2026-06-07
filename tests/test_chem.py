"""Tests for the RDKit chemistry helpers."""

from molgen.chem import canonicalize_smiles, is_valid_smiles, randomize_smiles


def test_is_valid_smiles():
    assert is_valid_smiles("CCO")
    assert is_valid_smiles("c1ccccc1")
    assert not is_valid_smiles("C(C")  # unbalanced parenthesis
    assert not is_valid_smiles("not a molecule")


def test_canonicalize_is_idempotent_and_order_invariant():
    canon = canonicalize_smiles("OCC")
    assert canon is not None
    assert canonicalize_smiles(canon) == canon
    # Different writings of ethanol map to the same canonical SMILES.
    assert canonicalize_smiles("OCC") == canonicalize_smiles("CCO")


def test_canonicalize_invalid_returns_none():
    assert canonicalize_smiles("xyz%%") is None


def test_randomize_smiles_preserves_molecule():
    rnd = randomize_smiles("c1ccccc1")
    assert rnd is not None
    assert canonicalize_smiles(rnd) == canonicalize_smiles("c1ccccc1")


def test_randomize_invalid_returns_none():
    assert randomize_smiles("???") is None
