"""Tests for the CharRNN language model."""

import torch
import torch.nn.functional as F

from molgen.char_rnn import CharRNN


def test_forward_output_shapes_gru():
    model = CharRNN(vocab_size=20, embedding_dim=16, hidden_dim=32, num_layers=2, pad_idx=0)
    x = torch.randint(0, 20, (4, 10))
    logits, hidden = model(x)
    assert logits.shape == (4, 10, 20)
    assert hidden is not None


def test_forward_works_with_lstm_cell():
    model = CharRNN(vocab_size=12, embedding_dim=8, hidden_dim=16, num_layers=1, cell="lstm")
    logits, (h, c) = model(torch.randint(0, 12, (2, 6)))
    assert logits.shape == (2, 6, 12)


def test_invalid_cell_raises():
    import pytest

    with pytest.raises(ValueError):
        CharRNN(vocab_size=10, cell="transformer")


def test_can_overfit_a_single_sequence():
    torch.manual_seed(0)
    model = CharRNN(
        vocab_size=10, embedding_dim=16, hidden_dim=32, num_layers=1, pad_idx=0, dropout=0.0
    )
    seq = torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8]])
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    first_loss = None
    last_loss = None
    for step in range(60):
        logits, _ = model(seq[:, :-1])
        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), seq[:, 1:].reshape(-1))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if step == 0:
            first_loss = loss.item()
        last_loss = loss.item()
    # The model should clearly learn to reproduce the fixed sequence.
    assert last_loss < first_loss * 0.5
