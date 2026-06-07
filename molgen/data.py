"""Datasets and data loading for SMILES sequence models."""

from __future__ import annotations

from collections.abc import Sequence
from functools import partial
from importlib.resources import files

import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset


class SmilesDataset(Dataset):
    """A dataset of SMILES strings encoded to token-id tensors.

    Each item is a 1-D LongTensor of token ids (with BOS/EOS). Padding is done
    dynamically per batch by :func:`make_collate_fn`, not stored per item, so
    short molecules don't waste memory.
    """

    def __init__(self, smiles_list: Sequence[str], tokenizer, max_len: int | None = None):
        self.smiles_list = list(smiles_list)
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self) -> int:
        return len(self.smiles_list)

    def __getitem__(self, idx: int) -> torch.Tensor:
        ids = self.tokenizer.encode(self.smiles_list[idx], add_bos_eos=True, max_len=self.max_len)
        return torch.tensor(ids, dtype=torch.long)


def _pad_collate(batch: list[torch.Tensor], pad_id: int) -> torch.Tensor:
    """Pad a list of 1-D id tensors to the longest in the batch -> (batch, seq)."""
    width = max(t.size(0) for t in batch)
    out = torch.full((len(batch), width), pad_id, dtype=torch.long)
    for i, t in enumerate(batch):
        out[i, : t.size(0)] = t
    return out


def make_collate_fn(pad_id: int):
    """Return a picklable collate function that pads batches to a common length."""
    return partial(_pad_collate, pad_id=pad_id)


def read_smiles_csv(path: str, column: str = "SMILES") -> list[str]:
    """Read a column of SMILES strings from a CSV file."""
    return pd.read_csv(path)[column].astype(str).tolist()


def build_dataloaders(
    smiles_list: Sequence[str],
    tokenizer,
    batch_size: int = 64,
    test_split: float = 0.1,
    max_len: int | None = None,
    shuffle: bool = True,
    num_workers: int = 0,
    seed: int = 42,
) -> tuple[DataLoader, DataLoader]:
    """Split SMILES into train/test sets and build padded DataLoaders."""
    train_smiles, test_smiles = train_test_split(
        list(smiles_list), test_size=test_split, random_state=seed
    )
    collate = make_collate_fn(tokenizer.pad_id)
    train_ds = SmilesDataset(train_smiles, tokenizer, max_len)
    test_ds = SmilesDataset(test_smiles, tokenizer, max_len)
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate,
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, collate_fn=collate
    )
    return train_loader, test_loader


def load_sample_smiles() -> list[str]:
    """Return the bundled sample of ~500 diverse, valid SMILES strings.

    Useful for examples, tests, and metric smoke-checks without downloading a
    full dataset. The molecules are synthetic (see ``synthetic.py``) but cover
    rings, heteroatoms, halogens, and unsaturation.
    """
    text = files("molgen.datasets").joinpath("sample_smiles.smi").read_text(encoding="utf-8")
    return [line.strip() for line in text.splitlines() if line.strip()]
