"""Tests for model + tokenizer checkpointing."""

import torch

from molgen.char_rnn import CharRNN
from molgen.checkpoint import load_checkpoint, save_checkpoint
from molgen.data import load_sample_smiles
from molgen.tokenizers import SmilesTokenizer


def test_checkpoint_roundtrip_reproduces_outputs(tmp_path):
    tok = SmilesTokenizer.from_smiles(load_sample_smiles()[:50])
    kwargs = dict(
        vocab_size=tok.vocab_size, embedding_dim=16, hidden_dim=32, num_layers=1, pad_idx=tok.pad_id
    )
    model = CharRNN(**kwargs)
    model.eval()
    x = torch.tensor([[tok.bos_id, 5, 6, 7]])
    with torch.no_grad():
        before = model(x)[0]

    path = tmp_path / "model.pt"
    save_checkpoint(path, model, "charrnn", kwargs, tok)
    model2, tok2 = load_checkpoint(path)
    model2.eval()
    with torch.no_grad():
        after = model2(x)[0]

    assert torch.allclose(before, after)
    assert tok2.itos == tok.itos
    assert tok2.vocab_size == tok.vocab_size


def test_save_checkpoint_rejects_unknown_model(tmp_path):
    import pytest

    tok = SmilesTokenizer.from_smiles(["CCO"])
    model = CharRNN(vocab_size=tok.vocab_size, pad_idx=tok.pad_id)
    with pytest.raises(ValueError):
        save_checkpoint(tmp_path / "m.pt", model, "diffusion", {}, tok)
