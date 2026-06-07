"""Integration tests for the generate / interpolate entry points."""

import torch
from rdkit import Chem

from molgen.generate import generate_nearby_smiles
from molgen.interpolate import interpolate_smiles
from molgen.vae import BetaTCVAE, SimpleTokenizer


def _save_tiny_model(path):
    # Architecture must match the hyperparameters hardcoded in generate/interpolate.
    model = BetaTCVAE(
        vocab_size=6,
        embedding_dim=16,
        hidden_dim=64,
        latent_dim=16,
        nhead=4,
        num_layers=2,
        pad_idx=4,
        device=torch.device("cpu"),
    )
    torch.save(model.state_dict(), path)


def test_generate_nearby_smiles_runs_on_explicit_device(tmp_path):
    ckpt = tmp_path / "model.pth"
    _save_tiny_model(ckpt)
    out = generate_nearby_smiles(
        str(ckpt),
        "CCO",
        SimpleTokenizer(),
        max_len=12,
        num_samples=5,
        device=torch.device("cpu"),
        temperature=1.0,
    )
    assert isinstance(out, list)
    # The function only returns RDKit-parseable SMILES.
    for smi in out:
        assert Chem.MolFromSmiles(smi) is not None


def test_interpolate_smiles_runs_on_explicit_device(tmp_path):
    ckpt = tmp_path / "model.pth"
    _save_tiny_model(ckpt)
    out = interpolate_smiles(
        str(ckpt),
        "CCO",
        "CCCO",
        SimpleTokenizer(),
        max_len=12,
        device=torch.device("cpu"),
        num_steps=5,
        temperature=1.0,
    )
    assert isinstance(out, list)
    for smi in out:
        assert Chem.MolFromSmiles(smi) is not None
