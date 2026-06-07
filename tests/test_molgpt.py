"""Tests for the MolGPT decoder-only Transformer."""

import torch
import torch.nn.functional as F

from molgen.molgpt import MolGPT


def _tiny_model():
    return MolGPT(
        vocab_size=12,
        embedding_dim=16,
        nhead=2,
        hidden_dim=32,
        num_layers=2,
        pad_idx=0,
        dropout=0.0,
        max_len=64,
    )


def test_forward_output_shape():
    model = _tiny_model()
    logits = model(torch.randint(0, 12, (3, 9)))
    assert logits.shape == (3, 9, 12)


def test_attention_is_causal():
    model = _tiny_model()
    model.eval()
    a = torch.tensor([[1, 2, 3, 4, 5, 6]])
    b = a.clone()
    b[0, 4:] = torch.tensor([9, 10])  # change only the last two (future) positions
    with torch.no_grad():
        out_a = model(a)
        out_b = model(b)
    # Predictions for positions 0..3 must not depend on tokens at positions >= 4.
    assert torch.allclose(out_a[:, :4], out_b[:, :4], atol=1e-5)


def test_can_overfit_a_single_sequence():
    torch.manual_seed(0)
    model = _tiny_model()
    seq = torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8]])
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    first_loss = None
    last_loss = None
    for step in range(80):
        logits = model(seq[:, :-1])
        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), seq[:, 1:].reshape(-1))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if step == 0:
            first_loss = loss.item()
        last_loss = loss.item()
    assert last_loss < first_loss * 0.5
