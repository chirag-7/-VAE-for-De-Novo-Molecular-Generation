"""Generate the README figures (trains on the GPU; all output is real).

Produces two figures in ``assets/``:

* ``generated_molecules.png``   - a grid of sampled structures.
* ``controlled_generation.png`` - goal-directed generation: after fine-tuning
  the model toward drug-likeness, the generated QED distribution shifts higher.

Usage::

    python scripts/make_figures.py --epochs 60 --focus-epochs 40 --num 1000
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from rdkit import Chem
from rdkit.Chem import Draw
from scipy.stats import gaussian_kde
from torch.utils.data import DataLoader

from molgen.chem import canonicalize_smiles
from molgen.data import SmilesDataset, build_dataloaders, load_sample_smiles, make_collate_fn
from molgen.molgpt import MolGPT
from molgen.properties import qed
from molgen.sampling import sample
from molgen.selfies_tokenizer import SelfiesTokenizer
from molgen.trainer import TrainConfig, train_language_model
from molgen.utils import get_device, set_seed

ASSETS = Path(__file__).resolve().parent.parent / "assets"
BASE_COLOR = "#475569"  # slate
FOCUS_COLOR = "#14b8a6"  # teal


def _sample_canonical(model, tokenizer, n: int, device) -> list[str]:
    raw = sample(model, tokenizer, num_samples=n, max_len=80, top_p=0.95, device=device)
    return [c for s in raw if (c := canonicalize_smiles(s))]


def _molecule_grid(smiles: list[str], n: int = 15) -> None:
    mols = []
    for s in smiles:
        mol = Chem.MolFromSmiles(s)
        if mol is not None and 6 <= mol.GetNumHeavyAtoms() <= 24:
            mols.append(mol)
        if len(mols) >= n:
            break
    Draw.MolsToGridImage(mols, molsPerRow=5, subImgSize=(230, 180)).save(
        ASSETS / "generated_molecules.png"
    )


def _controlled_generation(train_qed, base_qed, focus_qed) -> None:
    xs = np.linspace(0.0, 1.0, 240)
    fig, ax = plt.subplots(figsize=(7.6, 4.7), dpi=150)

    ax.plot(
        xs,
        gaussian_kde(train_qed)(xs),
        color="#94a3b8",
        lw=1.6,
        ls=":",
        label=f"training set  (mean {np.mean(train_qed):.2f})",
    )
    kb = gaussian_kde(base_qed)(xs)
    ax.fill_between(xs, kb, color=BASE_COLOR, alpha=0.30)
    ax.plot(
        xs,
        kb,
        color=BASE_COLOR,
        lw=2.4,
        label=f"baseline generation  (mean {np.mean(base_qed):.2f})",
    )
    kf = gaussian_kde(focus_qed)(xs)
    ax.fill_between(xs, kf, color=FOCUS_COLOR, alpha=0.35)
    ax.plot(
        xs,
        kf,
        color=FOCUS_COLOR,
        lw=2.4,
        label=f"QED-focused generation  (mean {np.mean(focus_qed):.2f})",
    )

    mb, mf = float(np.mean(base_qed)), float(np.mean(focus_qed))
    y = ax.get_ylim()[1] * 0.9
    ax.annotate(
        "", xy=(mf, y), xytext=(mb, y), arrowprops=dict(arrowstyle="-|>", color="#0f172a", lw=1.8)
    )
    ax.text(
        (mb + mf) / 2,
        y * 1.03,
        f"+{mf - mb:.2f} QED",
        ha="center",
        fontweight="bold",
        color="#0f172a",
    )

    ax.set_xlim(0, 1)
    ax.set_xlabel("QED  (drug-likeness)")
    ax.set_ylabel("density")
    ax.set_title("Goal-directed generation: steering toward higher QED", fontweight="bold")
    ax.legend(frameon=False, loc="upper left")
    fig.tight_layout()
    fig.savefig(ASSETS / "controlled_generation.png")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--focus-epochs", type=int, default=40)
    parser.add_argument("--num", type=int, default=1000)
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

    print(f"Training base model on {device} for {args.epochs} epochs...")
    train_language_model(
        model, train_loader, val_loader, TrainConfig(epochs=args.epochs), device, tokenizer.pad_id
    )
    baseline = _sample_canonical(model, tokenizer, args.num, device)
    base_qed = [q for s in baseline if (q := qed(s)) is not None]
    _molecule_grid(baseline)

    # Goal-directed fine-tuning: continue training on the most drug-like molecules.
    scored = sorted(((qed(s), s) for s in data if qed(s) is not None), key=lambda t: -t[0])
    high_qed = [s for _, s in scored[: int(0.4 * len(scored))]]
    loader = DataLoader(
        SmilesDataset(high_qed, tokenizer, augment=True),
        batch_size=32,
        shuffle=True,
        collate_fn=make_collate_fn(tokenizer.pad_id),
    )
    print(f"Fine-tuning toward high QED for {args.focus_epochs} epochs...")
    train_language_model(
        model,
        loader,
        None,
        TrainConfig(epochs=args.focus_epochs, lr=3e-4),
        device,
        tokenizer.pad_id,
    )
    focus_qed = [
        q
        for s in _sample_canonical(model, tokenizer, args.num, device)
        if (q := qed(s)) is not None
    ]

    _controlled_generation([q for q, _ in scored], base_qed, focus_qed)
    print(
        f"QED mean: baseline {np.mean(base_qed):.3f} -> focused {np.mean(focus_qed):.3f}; "
        f"figures written to {ASSETS}"
    )


if __name__ == "__main__":
    main()
