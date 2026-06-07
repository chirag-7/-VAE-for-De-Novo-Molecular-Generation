"""Tests for the language-model trainer."""

import torch

from molgen.char_rnn import CharRNN
from molgen.data import build_dataloaders, load_sample_smiles
from molgen.tokenizers import SmilesTokenizer
from molgen.trainer import TrainConfig, evaluate_language_model, lm_loss, train_language_model


def test_lm_loss_ignores_padding():
    logits = torch.randn(1, 3, 5)
    targets = torch.tensor([[1, 2, 0]])  # last target is padding (idx 0)
    loss_a = lm_loss(logits, targets, pad_idx=0)
    logits2 = logits.clone()
    logits2[:, 2, :] = torch.randn(5)  # change logits at the padded position
    loss_b = lm_loss(logits2, targets, pad_idx=0)
    assert torch.allclose(loss_a, loss_b)


def test_training_reduces_loss_on_sample_data():
    torch.manual_seed(0)
    smiles = load_sample_smiles()[:120]
    tok = SmilesTokenizer.from_smiles(smiles)
    train_loader, val_loader = build_dataloaders(smiles, tok, batch_size=16, test_split=0.2, seed=0)
    model = CharRNN(
        tok.vocab_size,
        embedding_dim=32,
        hidden_dim=64,
        num_layers=1,
        pad_idx=tok.pad_id,
        dropout=0.0,
    )
    config = TrainConfig(epochs=3, lr=1e-2, amp=False)
    history = train_language_model(
        model, train_loader, val_loader, config, device=torch.device("cpu"), pad_idx=tok.pad_id
    )
    assert len(history) == 3
    assert history[-1]["train_loss"] < history[0]["train_loss"]
    assert history[-1]["val_loss"] is not None


def test_evaluate_returns_finite_loss():
    smiles = load_sample_smiles()[:40]
    tok = SmilesTokenizer.from_smiles(smiles)
    _, val_loader = build_dataloaders(smiles, tok, batch_size=8, test_split=0.5, seed=0)
    model = CharRNN(
        tok.vocab_size, embedding_dim=16, hidden_dim=32, num_layers=1, pad_idx=tok.pad_id
    )
    loss = evaluate_language_model(model, val_loader, torch.device("cpu"), tok.pad_id)
    assert loss == loss and loss >= 0  # finite, non-negative
