"""Generate the figures shown in the README.

Trains a small SELFIES MolGPT on the bundled sample (so every sample is valid),
generates molecules, and writes four figures to ``assets/``:

* ``training_curve.png``        - train/validation loss
* ``property_distributions.png``- generated vs. training QED/logP/MW/SA
* ``chemical_space.png``        - Morgan-fingerprint PCA of both sets
* ``generated_molecules.png``   - a grid of generated structures

Usage::

    python scripts/make_figures.py --epochs 80 --num 1500
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from rdkit import Chem, DataStructs
from rdkit.Chem import Draw, rdFingerprintGenerator
from sklearn.decomposition import PCA

from molgen.chem import canonicalize_smiles
from molgen.data import build_dataloaders, load_sample_smiles
from molgen.molgpt import MolGPT
from molgen.properties import logp, molecular_weight, qed, sa_score
from molgen.sampling import sample
from molgen.selfies_tokenizer import SelfiesTokenizer
from molgen.trainer import TrainConfig, train_language_model
from molgen.utils import get_device, set_seed

ASSETS = Path(__file__).resolve().parent.parent / "assets"
TRAIN_COLOR = "#475569"  # slate
GEN_COLOR = "#14b8a6"  # teal


def _training_curve(history: list[dict]) -> None:
    epochs = [h["epoch"] for h in history]
    fig, ax = plt.subplots(figsize=(6.2, 4.0), dpi=140)
    ax.plot(epochs, [h["train_loss"] for h in history], color=GEN_COLOR, lw=2.4, label="train")
    ax.plot(
        epochs,
        [h["val_loss"] for h in history],
        color=TRAIN_COLOR,
        lw=2.4,
        ls="--",
        label="validation",
    )
    ax.set_xlabel("epoch")
    ax.set_ylabel("cross-entropy loss")
    ax.set_title("MolGPT training", fontweight="bold")
    ax.legend(frameon=False)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(ASSETS / "training_curve.png")
    plt.close(fig)


def _property_distributions(train: list[str], gen: list[str]) -> None:
    panels = [
        ("QED", qed),
        ("logP", logp),
        ("Molecular weight", molecular_weight),
        ("SA score", sa_score),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(9.0, 6.5), dpi=140)
    for ax, (name, fn) in zip(axes.ravel(), panels):
        t = [v for s in train if (v := fn(s)) is not None]
        g = [v for s in gen if (v := fn(s)) is not None]
        ax.hist(t, bins=30, density=True, color=TRAIN_COLOR, alpha=0.55, label="training")
        ax.hist(g, bins=30, density=True, color=GEN_COLOR, alpha=0.55, label="generated")
        ax.set_title(name, fontweight="bold")
        ax.grid(alpha=0.25)
    axes[0, 0].legend(frameon=False)
    fig.suptitle("Property distributions: generated vs. training", fontweight="bold", fontsize=13)
    fig.tight_layout()
    fig.savefig(ASSETS / "property_distributions.png")
    plt.close(fig)


def _fingerprint_matrix(smiles_list: list[str]) -> np.ndarray:
    generator = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=1024)
    rows = []
    for smiles in smiles_list:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            continue
        arr = np.zeros((1024,), dtype=np.int8)
        DataStructs.ConvertToNumpyArray(generator.GetFingerprint(mol), arr)
        rows.append(arr)
    return np.array(rows, dtype=np.float32)


def _chemical_space(train: list[str], gen: list[str]) -> None:
    xt, xg = _fingerprint_matrix(train), _fingerprint_matrix(gen)
    pca = PCA(n_components=2).fit(np.vstack([xt, xg]))
    pt, pg = pca.transform(xt), pca.transform(xg)
    fig, ax = plt.subplots(figsize=(6.5, 5.0), dpi=140)
    ax.scatter(
        pt[:, 0], pt[:, 1], s=18, color=TRAIN_COLOR, alpha=0.5, edgecolors="none", label="training"
    )
    ax.scatter(
        pg[:, 0], pg[:, 1], s=18, color=GEN_COLOR, alpha=0.5, edgecolors="none", label="generated"
    )
    ax.set_xlabel("PC 1")
    ax.set_ylabel("PC 2")
    ax.set_title("Chemical space (Morgan-fingerprint PCA)", fontweight="bold")
    ax.legend(frameon=False)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(ASSETS / "chemical_space.png")
    plt.close(fig)


def _molecule_grid(gen: list[str], n: int = 15) -> None:
    mols = []
    for smiles in gen:
        mol = Chem.MolFromSmiles(smiles)
        if mol is not None and 6 <= mol.GetNumHeavyAtoms() <= 24:
            mols.append(mol)
        if len(mols) >= n:
            break
    image = Draw.MolsToGridImage(mols, molsPerRow=5, subImgSize=(230, 180))
    image.save(ASSETS / "generated_molecules.png")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--num", type=int, default=1500)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    plt.switch_backend("Agg")
    plt.rcParams.update(
        {
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.facecolor": "white",
            "savefig.facecolor": "white",
            "font.size": 10,
        }
    )
    set_seed(args.seed)
    device = get_device()
    ASSETS.mkdir(exist_ok=True)

    data = load_sample_smiles()
    tokenizer = SelfiesTokenizer.from_smiles(data)
    train_loader, val_loader = build_dataloaders(
        data, tokenizer, batch_size=64, augment=True, seed=args.seed
    )
    model = MolGPT(
        tokenizer.vocab_size,
        embedding_dim=128,
        nhead=4,
        hidden_dim=256,
        num_layers=3,
        pad_idx=tokenizer.pad_id,
    )
    print(f"Training on {device} for {args.epochs} epochs...")
    history = train_language_model(
        model, train_loader, val_loader, TrainConfig(epochs=args.epochs), device, tokenizer.pad_id
    )

    print(f"Sampling {args.num} molecules...")
    raw = sample(model, tokenizer, num_samples=args.num, max_len=80, top_p=0.95, device=device)
    generated = sorted({c for s in raw if (c := canonicalize_smiles(s))})
    print(f"{len(generated)} unique valid molecules; writing figures to {ASSETS}")

    _training_curve(history)
    _property_distributions(data, generated)
    _chemical_space(data, generated)
    _molecule_grid(generated)
    print("done")


if __name__ == "__main__":
    main()
