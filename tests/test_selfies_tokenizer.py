"""Tests for the SELFIES tokenizer (skipped if the optional dep is absent)."""

import random

import pytest

pytest.importorskip("selfies")

from molgen.chem import canonicalize_smiles, is_valid_smiles  # noqa: E402
from molgen.selfies_tokenizer import SelfiesTokenizer  # noqa: E402


def test_from_smiles_includes_special_tokens():
    tok = SelfiesTokenizer.from_smiles(["CCO", "CCN", "c1ccccc1"])
    assert tok.pad_id == 0
    assert tok.vocab_size > 4  # specials plus at least a few SELFIES symbols


def test_encode_decode_recovers_molecule():
    tok = SelfiesTokenizer.from_smiles(["CCO", "c1ccccc1", "CC(=O)O"])
    ids = tok.encode("CC(=O)O")
    assert ids[0] == tok.bos_id
    assert ids[-1] == tok.eos_id
    decoded = tok.decode(ids)
    assert canonicalize_smiles(decoded) == canonicalize_smiles("CC(=O)O")


def test_random_ids_always_decode_to_valid_smiles():
    # The defining property of SELFIES: any symbol sequence is a valid molecule.
    tok = SelfiesTokenizer.from_smiles(["CCO", "CCN", "c1ccccc1", "CC(=O)O", "C1CCCCC1"])
    random.seed(0)
    for _ in range(50):
        ids = [random.randrange(tok.vocab_size) for _ in range(12)]
        smiles = tok.decode(ids)
        assert smiles == "" or is_valid_smiles(smiles)
