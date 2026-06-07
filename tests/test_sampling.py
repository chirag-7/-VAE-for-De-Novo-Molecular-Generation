"""Tests for autoregressive sampling."""

import pytest
import torch

from molgen.char_rnn import CharRNN
from molgen.chem import is_valid_smiles
from molgen.data import load_sample_smiles
from molgen.molgpt import MolGPT
from molgen.sampling import sample
from molgen.tokenizers import SmilesTokenizer

CPU = torch.device("cpu")


def test_sample_returns_requested_count_of_strings():
    tok = SmilesTokenizer.from_smiles(load_sample_smiles()[:100])
    model = CharRNN(
        tok.vocab_size, embedding_dim=16, hidden_dim=32, num_layers=1, pad_idx=tok.pad_id
    )
    out = sample(model, tok, num_samples=10, max_len=20, device=CPU)
    assert len(out) == 10
    assert all(isinstance(s, str) for s in out)


def test_top_k_and_top_p_run():
    tok = SmilesTokenizer.from_smiles(load_sample_smiles()[:100])
    model = MolGPT(
        tok.vocab_size, embedding_dim=16, nhead=2, hidden_dim=32, num_layers=1, pad_idx=tok.pad_id
    )
    out = sample(
        model, tok, num_samples=8, max_len=15, temperature=0.9, top_k=5, top_p=0.95, device=CPU
    )
    assert len(out) == 8


def test_sampling_with_selfies_is_always_valid():
    # SELFIES decoding guarantees validity even from an untrained model.
    pytest.importorskip("selfies")
    from molgen.selfies_tokenizer import SelfiesTokenizer

    tok = SelfiesTokenizer.from_smiles(load_sample_smiles()[:100])
    model = MolGPT(
        tok.vocab_size, embedding_dim=16, nhead=2, hidden_dim=32, num_layers=1, pad_idx=tok.pad_id
    )
    out = sample(model, tok, num_samples=16, max_len=20, device=CPU)
    assert len(out) == 16
    assert all(s == "" or is_valid_smiles(s) for s in out)
