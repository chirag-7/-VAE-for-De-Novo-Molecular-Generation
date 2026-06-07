"""Autoregressive RNN language model over SMILES tokens.

A GRU/LSTM that predicts the next token given the prefix -- the classic,
lightweight, and surprisingly strong baseline for SMILES generation (it is one
of the reference models in the MOSES benchmark).
"""

from __future__ import annotations

import torch
import torch.nn as nn


class CharRNN(nn.Module):
    """Next-token RNN language model.

    Operates on ``(batch, seq)`` token-id tensors. Train with teacher forcing:
    feed ``tokens[:, :-1]`` and predict ``tokens[:, 1:]``.
    """

    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int = 128,
        hidden_dim: int = 512,
        num_layers: int = 3,
        pad_idx: int = 0,
        dropout: float = 0.2,
        cell: str = "gru",
    ):
        super().__init__()
        self.cell = cell.lower()
        if self.cell not in ("gru", "lstm"):
            raise ValueError(f"cell must be 'gru' or 'lstm', got {cell!r}")
        self.pad_idx = pad_idx
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_idx)
        rnn_cls = nn.LSTM if self.cell == "lstm" else nn.GRU
        self.rnn = rnn_cls(
            embedding_dim,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.fc_out = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x: torch.Tensor, hidden=None):
        """Return ``(logits, hidden)`` where logits is ``(batch, seq, vocab)``."""
        embedded = self.embedding(x)
        output, hidden = self.rnn(embedded, hidden)
        logits = self.fc_out(output)
        return logits, hidden
