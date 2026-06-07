"""Tests for the regex SMILES tokenizer."""

import torch

from molgen.tokenizers import SPECIAL_TOKENS, SmilesTokenizer, smiles_tokens


def test_smiles_tokens_are_atom_level():
    assert smiles_tokens("Cc1ccc(Cl)cc1") == [
        "C",
        "c",
        "1",
        "c",
        "c",
        "c",
        "(",
        "Cl",
        ")",
        "c",
        "c",
        "1",
    ]
    # Bracket atoms are a single token.
    assert smiles_tokens("[nH]") == ["[nH]"]
    assert smiles_tokens("[C@@H]") == ["[C@@H]"]


def test_from_smiles_includes_specials_and_atoms():
    tok = SmilesTokenizer.from_smiles(["CCO", "CCN"])
    for special in SPECIAL_TOKENS:
        assert special in tok.stoi
    assert {"C", "O", "N"} <= set(tok.stoi)
    assert tok.vocab_size == len(tok.itos)


def test_encode_decode_roundtrip():
    tok = SmilesTokenizer.from_smiles(["CCO", "c1ccccc1", "CC(=O)O"])
    smiles = "CC(=O)O"
    ids = tok.encode(smiles)
    assert ids[0] == tok.bos_id
    assert ids[-1] == tok.eos_id
    assert tok.decode(ids) == smiles


def test_encode_pads_and_truncates_to_max_len():
    tok = SmilesTokenizer.from_smiles(["CCO"])
    ids = tok.encode("CCO", max_len=10)
    assert len(ids) == 10
    assert ids[-1] == tok.pad_id

    short = tok.encode("CCO", max_len=3)
    assert len(short) == 3


def test_unknown_tokens_map_to_unk():
    tok = SmilesTokenizer.from_smiles(["CCO"])  # vocab: C, O (+ specials)
    ids = tok.encode("CCN", add_bos_eos=False)  # N is out of vocabulary
    assert tok.unk_id in ids


def test_encode_batch_returns_padded_long_tensor():
    tok = SmilesTokenizer.from_smiles(["CCO", "CCN", "c1ccccc1"])
    batch = tok.encode_batch(["CCO", "c1ccccc1"])
    assert isinstance(batch, torch.Tensor)
    assert batch.dtype == torch.long
    assert batch.shape[0] == 2
    # All rows share a common (padded) width.
    assert batch.shape[1] == max(len(tok.encode("CCO")), len(tok.encode("c1ccccc1")))
