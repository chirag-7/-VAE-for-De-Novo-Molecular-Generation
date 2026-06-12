"""Integration tests for the VAE latent-space entry points.

Train a tiny VAE on a handful of bundled molecules (CPU, a few seconds) and
check that ``generate_nearby_smiles`` and ``interpolate_smiles`` produce valid,
distinct molecules. With the SELFIES tokenizer every decode is syntactically
valid, so these assert the end-to-end plumbing rather than model quality.
"""

import os

import torch
from torch.utils.data import DataLoader, TensorDataset

from molgen.chem import is_valid_smiles
from molgen.generate import generate_nearby_smiles
from molgen.interpolate import interpolate_smiles
from molgen.selfies_tokenizer import SelfiesTokenizer
from molgen.vae import BetaTCVAE, train

_SAMPLE = os.path.join(os.path.dirname(__file__), "..", "molgen", "datasets", "sample_smiles.smi")


def _corpus(n=60):
    return [s.strip() for s in open(_SAMPLE).read().splitlines() if s.strip()][:n]


def _train_tiny_vae(smiles, tokenizer, epochs=6):
    torch.manual_seed(0)
    data = tokenizer.encode_batch(smiles, max_len=None)
    device = torch.device("cpu")
    model = BetaTCVAE(tokenizer.vocab_size, 32, 64, 32, 4, 2, tokenizer.pad_id, device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loader = DataLoader(TensorDataset(data), batch_size=16, shuffle=True)
    for _ in range(epochs):
        train(model, loader, optimizer, device, beta=0.5, gamma=0.1)
    return model, data.shape[1]


def test_generate_nearby_produces_valid_distinct_molecules():
    smiles = _corpus()
    tok = SelfiesTokenizer.from_smiles(smiles)
    model, max_len = _train_tiny_vae(smiles, tok)

    seed = smiles[0]
    out = generate_nearby_smiles(
        model, tok, seed, max_len, 50, torch.device("cpu"), temperature=1.0
    )
    assert len(out) > 0  # SELFIES decoding makes every sample a valid molecule
    assert all(is_valid_smiles(s) for s in out)
    assert len(out) == len(set(out))  # distinct
    assert seed not in out  # the seed itself is excluded


def test_interpolate_returns_valid_ordered_path():
    smiles = _corpus()
    tok = SelfiesTokenizer.from_smiles(smiles)
    model, max_len = _train_tiny_vae(smiles, tok)

    path = interpolate_smiles(
        model, tok, smiles[0], smiles[1], max_len, torch.device("cpu"), num_steps=8
    )
    assert len(path) > 0
    assert all(is_valid_smiles(s) for s in path)
    assert len(path) == len(set(path))  # distinct along the path
