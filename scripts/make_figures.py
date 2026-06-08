"""Generate the README figures (trains on the GPU; all output is real).

Produces two figures in ``assets/``:

* ``generated_molecules.png``   - a grid of sampled structures.
* ``controlled_generation.png`` - goal-directed generation. Starting from one
  base model, fine-tuning toward the most / least drug-like molecules steers the
  generated QED distribution in both directions (panel A), which also moves the
  samples through QED-vs-SA property space (panel B).

Usage::

    python scripts/make_figures.py --epochs 60 --focus-epochs 40 --num 1000
"""

from __future__ import annotations

import argparse
import copy
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
from molgen.properties import qed, sa_score
from molgen.sampling import sample
from molgen.selfies_tokenizer import SelfiesTokenizer
from molgen.trainer import TrainConfig, train_language_model
from molgen.utils import get_device, set_seed

ASSETS = Path(__file__).resolve().parent.parent / "assets"
# (label, colour) for the steered series, in plotting order.
SERIES = [("steer ↓ low", "#f59e0b"), ("baseline", "#475569"), ("steer ↑ high", "#14b8a6")]


def _sample_canonical(model, tokenizer, n: int, device) -> list[str]:
    raw = sample(model, tokenizer, num_samples=n, max_len=80, top_p=0.95, device=device)
    return [c for s in raw if (c := canonicalize_smiles(s))]


def _qed_sa(smiles: list[str]) -> tuple[np.ndarray, np.ndarray]:
    qs, sas = [], []
    for s in smiles:
        q, a = qed(s), sa_score(s)
        if q is not None and a is not None:
            qs.append(q)
            sas.append(a)
    return np.array(qs), np.array(sas)


def _finetune(model, subset, tokenizer, device, epochs: int) -> None:
    loader = DataLoader(
        SmilesDataset(subset, tokenizer, augment=True),
        batch_size=32,
        shuffle=True,
        collate_fn=make_collate_fn(tokenizer.pad_id),
    )
    train_language_model(
        model, loader, None, TrainConfig(epochs=epochs, lr=3e-4), device, tokenizer.pad_id
    )


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


def _controlled_figure(
    train_qed: np.ndarray, series: dict[str, tuple[np.ndarray, np.ndarray]]
) -> None:
    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(12.6, 5.0), dpi=150)
    xs = np.linspace(0.0, 1.0, 240)

    # Panel A: bidirectional QED steering as KDE curves.
    ax_a.plot(
        xs, gaussian_kde(train_qed)(xs), color="#cbd5e1", lw=1.6, ls=":", label="training set"
    )
    for label, color in SERIES:
        q = series[label][0]
        density = gaussian_kde(q)(xs)
        ax_a.fill_between(xs, density, color=color, alpha=0.30)
        ax_a.plot(xs, density, color=color, lw=2.4, label=f"{label}  (mean {q.mean():.2f})")
    lo_m, hi_m = series[SERIES[0][0]][0].mean(), series[SERIES[2][0]][0].mean()
    y = ax_a.get_ylim()[1] * 0.92
    ax_a.annotate(
        "",
        xy=(hi_m, y),
        xytext=(lo_m, y),
        arrowprops=dict(arrowstyle="<|-|>", color="#0f172a", lw=1.6),
    )
    ax_a.text(
        (lo_m + hi_m) / 2, y * 1.03, f"{hi_m - lo_m:.2f} QED range", ha="center", fontweight="bold"
    )
    ax_a.set_xlim(0, 1)
    ax_a.set_xlabel("QED  (drug-likeness)")
    ax_a.set_ylabel("density")
    ax_a.set_title("Steering the QED distribution", fontweight="bold")
    ax_a.legend(frameon=False, loc="upper left")

    # Panel B: the same samples in QED-vs-SA property space.
    for label, color in SERIES:
        q, sa = series[label]
        ax_b.scatter(
            q[:300], sa[:300], s=14, color=color, alpha=0.45, edgecolors="none", label=label
        )
    ax_b.set_xlim(0, 1)
    ax_b.set_xlabel("QED  (drug-likeness)")
    ax_b.set_ylabel("SA score  (lower = easier to make)")
    ax_b.set_title("Movement through property space", fontweight="bold")
    ax_b.legend(frameon=False, loc="upper right")

    fig.suptitle(
        "Goal-directed generation: one base model steered both ways", fontweight="bold", fontsize=14
    )
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
    base_state = copy.deepcopy(model.state_dict())

    baseline = _sample_canonical(model, tokenizer, args.num, device)
    _molecule_grid(baseline)

    scored = sorted(((qed(s), s) for s in data if qed(s) is not None), key=lambda t: t[0])
    cut = int(0.4 * len(scored))
    low_subset = [s for _, s in scored[:cut]]
    high_subset = [s for _, s in scored[-cut:]]

    print(f"Fine-tuning toward high QED for {args.focus_epochs} epochs...")
    model.load_state_dict(base_state)
    _finetune(model, high_subset, tokenizer, device, args.focus_epochs)
    high = _sample_canonical(model, tokenizer, args.num, device)

    print(f"Fine-tuning toward low QED for {args.focus_epochs} epochs...")
    model.load_state_dict(base_state)
    _finetune(model, low_subset, tokenizer, device, args.focus_epochs)
    low = _sample_canonical(model, tokenizer, args.num, device)

    series = {
        SERIES[0][0]: _qed_sa(low),
        SERIES[1][0]: _qed_sa(baseline),
        SERIES[2][0]: _qed_sa(high),
    }
    _controlled_figure(np.array([q for q, _ in scored]), series)
    means = {k: f"{v[0].mean():.3f}" for k, v in series.items()}
    print(f"QED means {means}; figures written to {ASSETS}")


if __name__ == "__main__":
    main()
