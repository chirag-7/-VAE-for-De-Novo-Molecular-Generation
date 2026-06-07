"""Tests for the BetaTCVAE model and loss."""

import torch

from molgen.vae import BetaTCVAE, loss_function, masked_token_accuracy


def _tiny_model():
    return BetaTCVAE(
        vocab_size=6,
        embedding_dim=16,
        hidden_dim=64,
        latent_dim=16,
        nhead=4,
        num_layers=2,
        pad_idx=4,
        device=torch.device("cpu"),
    )


def test_forward_output_shapes():
    model = _tiny_model()
    batch, seq = 4, 12
    x = torch.randint(0, 6, (batch, seq))
    out, mu, log_var = model(x)
    assert out.shape == (batch, seq, 6)
    assert mu.shape == (batch, seq, 16)
    assert log_var.shape == (batch, seq, 16)


def test_loss_is_finite_scalar():
    model = _tiny_model()
    x = torch.randint(0, 6, (4, 12))
    out, mu, log_var = model(x)
    loss = loss_function(out, x, mu, log_var, beta=1.0, gamma=0.1)
    assert loss.dim() == 0
    assert torch.isfinite(loss)


def test_backward_pass_populates_gradients():
    model = _tiny_model()
    x = torch.randint(0, 6, (2, 8))
    out, mu, log_var = model(x)
    loss = loss_function(out, x, mu, log_var, beta=1.0, gamma=0.1)
    loss.backward()
    grads = [p.grad for p in model.parameters() if p.requires_grad]
    assert any(g is not None and torch.isfinite(g).all() for g in grads)


def test_loss_ignores_pad_positions():
    # Targets at positions 2, 3, 4 are padding (pad_idx=4).
    x = torch.tensor([[0, 1, 4, 4, 4]])
    mu = torch.zeros(1, 5, 4)
    log_var = torch.zeros(1, 5, 4)
    recon_a = torch.randn(1, 5, 6)
    recon_b = recon_a.clone()
    # Change the logits only at the padding positions.
    recon_b[:, 2:, :] = torch.randn(1, 3, 6)

    # With masking, padding positions are ignored, so the loss is unchanged.
    loss_a = loss_function(recon_a, x, mu, log_var, beta=1.0, gamma=0.1, pad_idx=4)
    loss_b = loss_function(recon_b, x, mu, log_var, beta=1.0, gamma=0.1, pad_idx=4)
    assert torch.allclose(loss_a, loss_b)

    # Without masking (default ignore_index), those positions do count.
    loss_a_unmasked = loss_function(recon_a, x, mu, log_var, beta=1.0, gamma=0.1)
    loss_b_unmasked = loss_function(recon_b, x, mu, log_var, beta=1.0, gamma=0.1)
    assert not torch.allclose(loss_a_unmasked, loss_b_unmasked)


def test_masked_token_accuracy_excludes_padding():
    preds = torch.tensor([[0, 1, 2, 3]])
    targets = torch.tensor([[0, 1, 9, 9]])  # last two positions are padding (idx 9)
    correct, total = masked_token_accuracy(preds, targets, pad_idx=9)
    assert total == 2  # only the two non-padding positions are counted
    assert correct == 2

    # A wrong prediction at a non-pad position lowers the count; pad mismatches are ignored.
    preds_wrong = torch.tensor([[0, 7, 2, 3]])
    correct2, total2 = masked_token_accuracy(preds_wrong, targets, pad_idx=9)
    assert total2 == 2
    assert correct2 == 1


def test_encoder_processes_sequences_independently():
    # With batch_first attention each sequence is encoded independently of the
    # other rows in the batch. Under the old batch_first=False bug, attention
    # ran across the batch dimension and this invariant would fail.
    model = _tiny_model()
    model.eval()
    a = torch.tensor([[0, 1, 2, 3, 0, 1]])
    b = torch.tensor([[1, 1, 0, 0, 2, 3]])
    with torch.no_grad():
        enc_both = model.encoder(model.embedding(torch.cat([a, b], dim=0)))
        enc_a = model.encoder(model.embedding(a))
    assert torch.allclose(enc_both[0], enc_a[0], atol=1e-5)
