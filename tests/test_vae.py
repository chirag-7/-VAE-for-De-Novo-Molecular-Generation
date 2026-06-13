"""Tests for the BetaTCVAE model and loss."""

import math

import torch

from molgen.vae import (
    BetaTCVAE,
    PositionalEncoding,
    gaussian_log_density,
    kl_decomposition,
    loss_function,
    masked_token_accuracy,
)


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
    out, mu, log_var, z = model(x)
    assert out.shape == (batch, seq, 6)
    # One fixed-size latent vector per molecule (not per position).
    assert mu.shape == (batch, 16)
    assert log_var.shape == (batch, 16)
    assert z.shape == (batch, 16)


def test_loss_is_finite_scalar():
    model = _tiny_model()
    x = torch.randint(0, 6, (4, 12))
    out, mu, log_var, z = model(x)
    loss = loss_function(out, x, mu, log_var, z, beta=4.0, gamma=1.0)
    assert loss.dim() == 0
    assert torch.isfinite(loss)


def test_backward_pass_populates_gradients():
    model = _tiny_model()
    x = torch.randint(0, 6, (2, 8))
    out, mu, log_var, z = model(x)
    loss = loss_function(out, x, mu, log_var, z, beta=4.0, gamma=1.0)
    loss.backward()
    grads = [p.grad for p in model.parameters() if p.requires_grad]
    assert any(g is not None and torch.isfinite(g).all() for g in grads)


def test_loss_ignores_pad_positions():
    # Targets at positions 2, 3, 4 are padding (pad_idx=4).
    x = torch.tensor([[0, 1, 4, 4, 4]])
    mu = torch.zeros(1, 4)
    log_var = torch.zeros(1, 4)
    z = torch.zeros(1, 4)  # same latent for both, so only BCE can differ
    recon_a = torch.randn(1, 5, 6)
    recon_b = recon_a.clone()
    # Change the logits only at the padding positions.
    recon_b[:, 2:, :] = torch.randn(1, 3, 6)

    # With masking, padding positions are ignored, so the loss is unchanged.
    loss_a = loss_function(recon_a, x, mu, log_var, z, beta=1.0, gamma=1.0, pad_idx=4)
    loss_b = loss_function(recon_b, x, mu, log_var, z, beta=1.0, gamma=1.0, pad_idx=4)
    assert torch.allclose(loss_a, loss_b)

    # Without masking (default ignore_index), those positions do count.
    loss_a_unmasked = loss_function(recon_a, x, mu, log_var, z, beta=1.0, gamma=1.0)
    loss_b_unmasked = loss_function(recon_b, x, mu, log_var, z, beta=1.0, gamma=1.0)
    assert not torch.allclose(loss_a_unmasked, loss_b_unmasked)


def test_kl_decomposition_sums_to_total_kl():
    # The three estimated terms (MI + TC + dim-wise KL) must sum to the plain
    # Monte-Carlo KL[q(z|x) || p(z)] = E[log q(z|x) - log p(z)] on the same z.
    torch.manual_seed(0)
    mu = torch.randn(32, 8)
    log_var = torch.randn(32, 8) * 0.5
    z = mu + torch.randn_like(mu) * torch.exp(0.5 * log_var)

    mi, tc, dim_kl = kl_decomposition(z, mu, log_var, dataset_size=10000)

    log_qz_x = gaussian_log_density(z, mu, log_var).sum(dim=1)
    log_pz = gaussian_log_density(z, torch.zeros_like(z), torch.zeros_like(z)).sum(dim=1)
    mc_kl = (log_qz_x - log_pz).mean()

    assert torch.isclose(mi + tc + dim_kl, mc_kl, atol=1e-4)


def test_total_correlation_detects_dependence():
    # The stratified estimator should report ~0 total correlation when the
    # aggregate posterior factorizes, and clearly positive TC when two latent
    # dimensions are tied — the property that makes it a *real* TC term rather
    # than the old closed-form proxy.
    torch.manual_seed(0)

    # Independent posteriors -> aggregate posterior factorizes -> TC ~ 0.
    mu = torch.zeros(256, 6)
    log_var = torch.zeros(256, 6)
    z = torch.randn(256, 6)
    _, tc_indep, _ = kl_decomposition(z, mu, log_var, dataset_size=10000)
    assert abs(tc_indep.item()) < 0.5

    # Tie dims 0 and 1 across the batch (correlated means) -> TC clearly > 0.
    shared = torch.randn(256, 1) * 2
    mu_corr = torch.cat([shared, shared, torch.randn(256, 4) * 2], dim=1)
    log_var_corr = torch.full((256, 6), -2.0)
    z_corr = mu_corr + torch.randn_like(mu_corr) * torch.exp(0.5 * log_var_corr)
    _, tc_corr, _ = kl_decomposition(z_corr, mu_corr, log_var_corr, dataset_size=10000)
    assert tc_corr.item() > 1.0
    assert tc_corr.item() > tc_indep.item() + 1.0


def test_gaussian_log_density_matches_closed_form():
    z = torch.tensor([[0.5, -1.0]])
    mu = torch.tensor([[0.0, 0.0]])
    log_var = torch.zeros(1, 2)  # unit variance
    expected = -0.5 * (math.log(2 * math.pi) + z**2)  # standard normal log-pdf
    assert torch.allclose(gaussian_log_density(z, mu, log_var), expected, atol=1e-6)


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


def test_positional_encoding_is_position_dependent():
    pe = PositionalEncoding(embedding_dim=16)
    out = pe(torch.zeros(1, 10, 16))
    assert out.shape == (1, 10, 16)
    # With a zero input the output is the positional code, which differs per position.
    assert not torch.allclose(out[0, 0], out[0, 1])


def test_latent_dim_independent_of_embedding_dim():
    # The latent is projected back to model width, so it can be a true
    # bottleneck (smaller than the embedding) without raising.
    model = BetaTCVAE(10, 32, 64, 8, 4, 2, 0, torch.device("cpu"))
    out, mu, log_var, z = model(torch.randint(0, 10, (2, 6)))
    assert out.shape == (2, 6, 10)
    assert mu.shape == (2, 8)
    assert z.shape == (2, 8)
