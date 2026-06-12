"""Transformer beta-TC-VAE for reconstruction and latent-space exploration.

The VAE encodes a (padded) token sequence to a per-position latent and decodes
it back through a Transformer decoder. Unlike the autoregressive `CharRNN` /
`MolGPT` models, it supports *latent-space* operations — sampling molecules
near a seed and interpolating between two molecules — see
:mod:`molgen.generate` and :mod:`molgen.interpolate`.

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
    """Transformer beta-TC-VAE with a per-position latent.

    Args:
        vocab_size: Tokenizer vocabulary size.
        embedding_dim: Token embedding width. Must equal ``latent_dim`` because
            the decoder consumes the latent ``z`` directly as its target stream.
        hidden_dim: Transformer feed-forward width.
        latent_dim: Latent width (kept equal to ``embedding_dim``).
        nhead: Number of attention heads.
        num_layers: Number of encoder/decoder layers.
        pad_idx: Padding token id (masked out of attention and the loss).
        device: Device used for the reparameterization noise.
    """

    def __init__(
        self, vocab_size, embedding_dim, hidden_dim, latent_dim, nhead, num_layers, pad_idx, device
    ):
        super().__init__()
        if latent_dim != embedding_dim:
            raise ValueError(
                f"latent_dim ({latent_dim}) must equal embedding_dim ({embedding_dim}); "
                "the decoder consumes the latent directly as its target stream."
            )

        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_idx)
        self.pos_encoder = PositionalEncoding(embedding_dim)
        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(embedding_dim, nhead, hidden_dim, batch_first=True),
            num_layers=num_layers,
        )
        self.mu = nn.Linear(embedding_dim, latent_dim)
        self.log_var = nn.Linear(embedding_dim, latent_dim)

        self.decoder = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(embedding_dim, nhead, hidden_dim, batch_first=True),
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
        """Encode token ids to ``(memory, mu)``.

        ``memory`` is the encoder output consumed by the decoder; ``mu`` is the
        latent mean. Both have shape ``(batch, seq, dim)``. Padding positions
        are masked out of attention.
        """
        pad_mask = x == self.pad_idx
        embedded = self.pos_encoder(self.embedding(x))
        memory = self.encoder(embedded, src_key_padding_mask=pad_mask)
        return memory, self.mu(memory)

    def decode_logits(self, z, memory, memory_pad_mask=None):
        """Decode a latent ``z`` against encoder ``memory`` to vocab logits."""
        decoded = self.decoder(z, memory, memory_key_padding_mask=memory_pad_mask)
        return self.fc_out(decoded)

    def forward(self, x):
        pad_mask = x == self.pad_idx
        memory, mu = self.encode(x)
        log_var = self.log_var(memory)
        z = self.reparameterize(mu, log_var)
        decoded = self.decoder(
            z,
            memory,
            tgt_key_padding_mask=pad_mask,
            memory_key_padding_mask=pad_mask,
        )
        return self.fc_out(decoded), mu, log_var


def loss_function(recon_x, x, mu, log_var, beta, gamma, pad_idx=-100):
    BCE = F.cross_entropy(
        recon_x.view(-1, recon_x.size(-1)), x.view(-1), ignore_index=pad_idx, reduction="mean"
    )
    KLD = -0.5 * torch.mean(1 + log_var - mu.pow(2) - log_var.exp())
    TC = (log_var.exp() - 1 - log_var).mean()

    return BCE + beta * KLD + gamma * TC


def masked_token_accuracy(preds, targets, pad_idx):
    """Token-level accuracy over non-padding positions.

    Returns ``(num_correct, num_tokens)`` so callers can aggregate across batches.
    """
    mask = targets != pad_idx
    correct = int(((preds == targets) & mask).sum().item())
    total = int(mask.sum().item())
    return correct, total


def train(model, train_loader, optimizer, device, beta, gamma):
    model.train()
    total_loss = 0.0

    for batch in train_loader:
        x = (batch[0] if isinstance(batch, (list, tuple)) else batch).to(device)
        optimizer.zero_grad()
        recon_x, mu, log_var = model(x)
        loss = loss_function(recon_x, x, mu, log_var, beta, gamma, pad_idx=model.pad_idx)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    return total_loss / len(train_loader)


def test(model, test_loader, device, beta, gamma):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for batch in test_loader:
            x = (batch[0] if isinstance(batch, (list, tuple)) else batch).to(device)
            recon_x, mu, log_var = model(x)
            loss = loss_function(recon_x, x, mu, log_var, beta, gamma, pad_idx=model.pad_idx)
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

    embedding_dim = latent_dim = 64
    model_kwargs = dict(
        vocab_size=tokenizer.vocab_size,
        embedding_dim=embedding_dim,
        hidden_dim=256,
        latent_dim=latent_dim,
        nhead=4,
        num_layers=2,
        pad_idx=tokenizer.pad_id,
        device=device,
    )
    model = BetaTCVAE(**model_kwargs).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loader = DataLoader(TensorDataset(data), batch_size=64, shuffle=True)

    for epoch in range(10):
        loss = train(model, loader, optimizer, device, beta=0.5, gamma=0.1)
        print(f"Epoch {epoch + 1}: loss={loss:.4f}")

    save_checkpoint("vae_model.pt", model, "vae", model_kwargs, tokenizer)
    print("Saved checkpoint to vae_model.pt")


if __name__ == "__main__":
    main()
