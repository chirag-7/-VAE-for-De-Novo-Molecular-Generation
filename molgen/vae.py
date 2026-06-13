"""Transformer beta-TC-VAE for reconstruction and latent-space exploration.

This is a *sentence* VAE: each molecule is encoded to a single fixed-size latent
vector (a masked mean-pool over the encoder states), so a molecule corresponds
to one point in latent space. Unlike the autoregressive `CharRNN` / `MolGPT`
models, it supports *latent-space* operations — sampling molecules near a seed
and interpolating between two molecules — see :mod:`molgen.generate` and
:mod:`molgen.interpolate`.

The objective is a genuine **β-TC-VAE** (Chen et al., 2018, "Isolating Sources
of Disentanglement in VAEs"): the KL term is decomposed into the index-code
mutual information, the **total correlation**, and the dimension-wise KL, each
separately weighted (``alpha``, ``beta``, ``gamma``). The total correlation —
the term ``beta`` scales to encourage disentangled latents — is estimated from
the minibatch with the paper's minibatch-stratified-sampling estimator, so it is
the real ``KL[q(z) || prod_j q(z_j)]`` rather than a closed-form proxy.

Tokenization goes through the shared toolkit tokenizers
(:class:`molgen.selfies_tokenizer.SelfiesTokenizer` is recommended, since its
decoder always yields a syntactically valid molecule).
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = [
    "PositionalEncoding",
    "BetaTCVAE",
    "gaussian_log_density",
    "kl_decomposition",
    "loss_function",
    "masked_token_accuracy",
    "train",
    "test",
]


class PositionalEncoding(nn.Module):
    """Inject information about token positions using fixed sinusoids."""

    def __init__(self, embedding_dim, max_len=2048):
        super().__init__()
        pe = torch.zeros(max_len, embedding_dim)
        position = torch.arange(max_len).unsqueeze(1).float()
        div_term = torch.exp(
            torch.arange(0, embedding_dim, 2).float() * (-math.log(10000.0) / embedding_dim)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term[: embedding_dim // 2])
        self.register_buffer("pe", pe.unsqueeze(0))  # (1, max_len, embedding_dim)

    def forward(self, x):
        return x + self.pe[:, : x.size(1)]


class BetaTCVAE(nn.Module):
    """Transformer beta-TC-VAE with a single fixed-size latent per molecule.

    The encoder maps a token sequence to one latent vector (masked mean-pool of
    the encoder states); the decoder broadcasts that vector across positions,
    adds positional encodings, and predicts every token in parallel. ``latent_dim``
    is independent of ``embedding_dim`` — the latent is projected back up to the
    model width before decoding, so the latent acts as a genuine bottleneck.

    Args:
        vocab_size: Tokenizer vocabulary size.
        embedding_dim: Token embedding / model width.
        hidden_dim: Transformer feed-forward width.
        latent_dim: Latent vector width (the molecule embedding).
        nhead: Number of attention heads.
        num_layers: Number of encoder/decoder layers.
        pad_idx: Padding token id (masked out of attention, pooling, and loss).
        device: Device used for the reparameterization noise.
    """

    def __init__(
        self, vocab_size, embedding_dim, hidden_dim, latent_dim, nhead, num_layers, pad_idx, device
    ):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_idx)
        self.pos_encoder = PositionalEncoding(embedding_dim)
        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(embedding_dim, nhead, hidden_dim, batch_first=True),
            num_layers=num_layers,
        )
        self.mu = nn.Linear(embedding_dim, latent_dim)
        self.log_var = nn.Linear(embedding_dim, latent_dim)

        # Decoder: project the latent back to model width, broadcast across
        # positions, and refine with self-attention (no encoder cross-attention,
        # so there is no source memory / memory mask).
        self.latent_to_hidden = nn.Linear(latent_dim, embedding_dim)
        self.decoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(embedding_dim, nhead, hidden_dim, batch_first=True),
            num_layers=num_layers,
        )
        self.fc_out = nn.Linear(embedding_dim, vocab_size)

        self.device = device
        self.pad_idx = pad_idx

    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def encode(self, x):
        """Encode token ids to ``(mu, log_var)``, each of shape ``(batch, latent)``.

        Padding positions are masked out of both attention and the mean-pool, so
        the latent reflects only real tokens.
        """
        pad_mask = x == self.pad_idx
        h = self.encoder(self.pos_encoder(self.embedding(x)), src_key_padding_mask=pad_mask)
        weights = (~pad_mask).unsqueeze(-1).float()  # (batch, seq, 1)
        pooled = (h * weights).sum(dim=1) / weights.sum(dim=1).clamp(min=1.0)  # (batch, dim)
        return self.mu(pooled), self.log_var(pooled)

    def decode(self, z, seq_len):
        """Decode a latent ``z`` (batch, latent) to vocab logits (batch, seq_len, vocab)."""
        hidden = self.latent_to_hidden(z).unsqueeze(1).expand(-1, seq_len, -1)
        hidden = self.pos_encoder(hidden)
        return self.fc_out(self.decoder(hidden))

    def forward(self, x):
        """Return ``(logits, mu, log_var, z)`` — ``z`` is the sampled latent the
        loss needs for its minibatch total-correlation estimate."""
        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)
        return self.decode(z, x.size(1)), mu, log_var, z


def gaussian_log_density(z, mu, log_var):
    """Element-wise ``log N(z; mu, exp(log_var))`` (broadcasts over its inputs)."""
    return -0.5 * (math.log(2 * math.pi) + log_var + (z - mu) ** 2 * torch.exp(-log_var))


def _log_importance_weight_matrix(batch_size, dataset_size, device):
    """Minibatch-stratified-sampling weights (Chen et al., 2018, appendix C.1).

    Returns a ``(batch, batch)`` log-weight matrix that de-biases the aggregate
    posterior estimate so the individual total-correlation / dim-wise-KL terms
    are accurate (plain minibatch weighting biases them by ``(D-1) log M``).
    """
    n = dataset_size
    m = batch_size - 1
    strat_weight = (n - m) / (n * m)
    weight = torch.full((batch_size, batch_size), 1 / m, device=device)
    weight.view(-1)[:: m + 1] = 1 / n
    weight.view(-1)[1 :: m + 1] = strat_weight
    weight[m - 1, 0] = strat_weight
    return weight.log()


def kl_decomposition(z, mu, log_var, dataset_size=None):
    """Decompose the KL into (mutual information, total correlation, dim-wise KL).

    Estimates the aggregate posterior ``q(z)`` and its marginal product
    ``prod_j q(z_j)`` from the minibatch (Chen et al., 2018, "Isolating Sources
    of Disentanglement in VAEs"). With ``dataset_size`` (``N``) and a batch of at
    least two molecules this uses **minibatch stratified sampling**, whose
    importance weights de-bias the individual terms; otherwise it falls back to
    the plain minibatch weight ``log(1 / M)``.

    Returns three scalar tensors ``(mutual_info, total_correlation, dim_kl)``
    whose sum equals the Monte-Carlo ``KL[q(z|x) || N(0, I)]`` on the same ``z``.
    """
    batch_size = z.size(0)

    log_qz_x = gaussian_log_density(z, mu, log_var).sum(dim=1)  # (M,)
    log_pz = gaussian_log_density(z, torch.zeros_like(z), torch.zeros_like(z)).sum(dim=1)  # (M,)

    # Pairwise per-dim densities log q(z_i | x_j): (M, M, D).
    mat = gaussian_log_density(z.unsqueeze(1), mu.unsqueeze(0), log_var.unsqueeze(0))

    if dataset_size and dataset_size > batch_size and batch_size > 1:
        logiw = _log_importance_weight_matrix(batch_size, dataset_size, z.device)  # (M, M)
        log_qz = torch.logsumexp(logiw + mat.sum(dim=2), dim=1)  # (M,)
        log_qz_prod = torch.logsumexp(logiw.unsqueeze(-1) + mat, dim=1).sum(dim=1)  # (M,)
    else:
        log_norm = math.log(batch_size)
        log_qz = torch.logsumexp(mat.sum(dim=2), dim=1) - log_norm
        log_qz_prod = (torch.logsumexp(mat, dim=1) - log_norm).sum(dim=1)

    mutual_info = (log_qz_x - log_qz).mean()
    total_correlation = (log_qz - log_qz_prod).mean()
    dim_kl = (log_qz_prod - log_pz).mean()
    return mutual_info, total_correlation, dim_kl


def loss_function(
    recon_x, x, mu, log_var, z, beta=1.0, gamma=1.0, alpha=1.0, dataset_size=None, pad_idx=-100
):
    """β-TC-VAE loss: reconstruction + ``alpha*MI + beta*TC + gamma*dim_kl``.

    ``beta`` weights the **total correlation** (the disentanglement knob);
    ``alpha`` the index-code mutual information; ``gamma`` the dimension-wise KL.
    With ``alpha == beta == gamma`` the penalty reduces to the plain VAE KL.
    """
    BCE = F.cross_entropy(
        recon_x.view(-1, recon_x.size(-1)), x.view(-1), ignore_index=pad_idx, reduction="mean"
    )
    mutual_info, total_correlation, dim_kl = kl_decomposition(z, mu, log_var, dataset_size)
    return BCE + alpha * mutual_info + beta * total_correlation + gamma * dim_kl


def masked_token_accuracy(preds, targets, pad_idx):
    """Token-level accuracy over non-padding positions.

    Returns ``(num_correct, num_tokens)`` so callers can aggregate across batches.
    """
    mask = targets != pad_idx
    correct = int(((preds == targets) & mask).sum().item())
    total = int(mask.sum().item())
    return correct, total


def _dataset_size(loader):
    try:
        return len(loader.dataset)
    except (AttributeError, TypeError):
        return None


def train(model, train_loader, optimizer, device, beta, gamma, alpha=1.0):
    model.train()
    total_loss = 0.0
    dataset_size = _dataset_size(train_loader)

    for batch in train_loader:
        x = (batch[0] if isinstance(batch, (list, tuple)) else batch).to(device)
        optimizer.zero_grad()
        recon_x, mu, log_var, z = model(x)
        loss = loss_function(
            recon_x,
            x,
            mu,
            log_var,
            z,
            beta=beta,
            gamma=gamma,
            alpha=alpha,
            dataset_size=dataset_size,
            pad_idx=model.pad_idx,
        )
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    return total_loss / len(train_loader)


def test(model, test_loader, device, beta, gamma, alpha=1.0):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    dataset_size = _dataset_size(test_loader)

    with torch.no_grad():
        for batch in test_loader:
            x = (batch[0] if isinstance(batch, (list, tuple)) else batch).to(device)
            recon_x, mu, log_var, z = model(x)
            loss = loss_function(
                recon_x,
                x,
                mu,
                log_var,
                z,
                beta=beta,
                gamma=gamma,
                alpha=alpha,
                dataset_size=dataset_size,
                pad_idx=model.pad_idx,
            )
            total_loss += loss.item()

            preds = torch.argmax(recon_x, dim=-1)
            batch_correct, batch_total = masked_token_accuracy(preds, x, model.pad_idx)
            correct += batch_correct
            total += batch_total

    avg_loss = total_loss / len(test_loader)
    avg_accuracy = correct / total if total > 0 else 0.0
    return avg_loss, avg_accuracy


def main():
    """Train a small VAE on the bundled sample dataset and save a checkpoint."""
    from torch.utils.data import DataLoader, TensorDataset

    from molgen.checkpoint import save_checkpoint
    from molgen.data import load_sample_smiles
    from molgen.selfies_tokenizer import SelfiesTokenizer

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    smiles = load_sample_smiles()
    tokenizer = SelfiesTokenizer.from_smiles(smiles)
    data = tokenizer.encode_batch(smiles, max_len=None)

    model_kwargs = dict(
        vocab_size=tokenizer.vocab_size,
        embedding_dim=64,
        hidden_dim=256,
        latent_dim=32,
        nhead=4,
        num_layers=2,
        pad_idx=tokenizer.pad_id,
        device=device,
    )
    model = BetaTCVAE(**model_kwargs).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loader = DataLoader(TensorDataset(data), batch_size=64, shuffle=True)

    # beta>1 up-weights the total-correlation term (disentanglement); alpha and
    # gamma keep the mutual-information and dimension-wise-KL terms at 1.0.
    for epoch in range(10):
        loss = train(model, loader, optimizer, device, beta=4.0, gamma=1.0, alpha=1.0)
        print(f"Epoch {epoch + 1}: loss={loss:.4f}")

    save_checkpoint("vae_model.pt", model, "vae", model_kwargs, tokenizer)
    print("Saved checkpoint to vae_model.pt")


if __name__ == "__main__":
    main()
