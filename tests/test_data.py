"""Tests for the SMILES dataset and data loaders."""

import torch

from molgen.chem import is_valid_smiles
from molgen.data import SmilesDataset, build_dataloaders, load_sample_smiles, make_collate_fn
from molgen.tokenizers import SmilesTokenizer

SMILES = ["CCO", "CCN", "c1ccccc1", "CC(=O)O", "C1CCCCC1", "CCCl", "CCBr", "CCS"]


def _tokenizer():
    return SmilesTokenizer.from_smiles(SMILES)


def test_dataset_item_is_long_tensor_with_bos_eos():
    tok = _tokenizer()
    ds = SmilesDataset(["CCO"], tok)
    item = ds[0]
    assert item.dtype == torch.long
    assert item[0].item() == tok.bos_id
    assert item[-1].item() == tok.eos_id


def test_collate_pads_to_longest_in_batch():
    tok = _tokenizer()
    collate = make_collate_fn(tok.pad_id)
    a = torch.tensor([1, 2, 3])
    b = torch.tensor([4, 5])
    out = collate([a, b])
    assert out.shape == (2, 3)
    assert out[1, 2].item() == tok.pad_id  # shorter row is padded


def test_build_dataloaders_yields_batches():
    tok = _tokenizer()
    train_loader, test_loader = build_dataloaders(SMILES, tok, batch_size=4, test_split=0.25)
    batch = next(iter(train_loader))
    assert batch.dim() == 2
    assert batch.dtype == torch.long
    # Train/test split sizes add up to the corpus size.
    n_train = len(train_loader.dataset)
    n_test = len(test_loader.dataset)
    assert n_train + n_test == len(SMILES)


def test_load_sample_smiles_returns_valid_molecules():
    smiles = load_sample_smiles()
    assert len(smiles) >= 100
    # Spot-check that the bundled molecules parse.
    assert all(is_valid_smiles(s) for s in smiles[:50])


def test_augmentation_preserves_the_molecule():
    from molgen.chem import canonicalize_smiles

    smiles = "CCOc1ccccc1"
    tok = SmilesTokenizer.from_smiles([smiles, *load_sample_smiles()])
    ds = SmilesDataset([smiles], tok, augment=True)
    target = canonicalize_smiles(smiles)
    # Each augmented encoding is a different ordering of the *same* molecule.
    for _ in range(10):
        decoded = tok.decode(ds[0].tolist())
        assert canonicalize_smiles(decoded) == target
