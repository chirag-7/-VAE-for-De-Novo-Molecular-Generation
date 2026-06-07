"""Tests for the SimpleTokenizer."""

import torch

from molgen.vae import SimpleTokenizer


def test_tokenize_shape_and_dtype():
    tok = SimpleTokenizer()
    out = tok.tokenize("CCO", max_len=10)
    assert isinstance(out, torch.Tensor)
    assert out.dtype == torch.long
    assert out.shape == (10,)


def test_tokenize_pads_with_pad_idx():
    tok = SimpleTokenizer()
    out = tok.tokenize("CC", max_len=5)
    assert out[0].item() == tok.char_to_idx["C"]
    assert out[1].item() == tok.char_to_idx["C"]
    assert bool((out[2:] == tok.pad_idx).all())


def test_tokenize_decode_roundtrip():
    tok = SimpleTokenizer()
    smiles = "C(O)CC"
    out = tok.tokenize(smiles, max_len=len(smiles))
    decoded = "".join(tok.idx_to_char[int(i)] for i in out)
    assert decoded == smiles
