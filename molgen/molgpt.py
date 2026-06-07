"""Decoder-only Transformer language model for SMILES (MolGPT-style).

A GPT-style stack: token embedding + positional encoding, a Transformer with
causal self-attention, then a linear head over the vocabulary. Padding is kept
at the end of each sequence, so the causal mask alone prevents attending to it.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from molgen.vae import PositionalEncoding


class MolGPT(nn.Module):
    """Autoregressive decoder-only Transformer over SMILES token ids."""

    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int = 256,
        nhead: int = 8,
        hidden_dim: int = 1024,
        num_layers: int = 4,
        pad_idx: int = 0,
        dropout: float = 0.1,
        max_len: int = 512,
    ):
        super().__init__()
        self.pad_idx = pad_idx
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_idx)
        self.pos_encoder = PositionalEncoding(embedding_dim, max_len=max_len)
        layer = nn.TransformerEncoderLayer(
            embedding_dim,
            nhead,
            hidden_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(embedding_dim)
        self.fc_out = nn.Linear(embedding_dim, vocab_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return next-token logits of shape ``(batch, seq, vocab)``."""
        seq_len = x.size(1)
        causal_mask = nn.Transformer.generate_square_subsequent_mask(seq_len, device=x.device)
        hidden = self.pos_encoder(self.embedding(x))
        hidden = self.transformer(hidden, mask=causal_mask, is_causal=True)
        hidden = self.norm(hidden)
        return self.fc_out(hidden)
