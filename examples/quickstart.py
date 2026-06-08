"""End-to-end quickstart for molgen.

Trains a small MolGPT on the bundled sample dataset, generates molecules, and
prints a MOSES-style evaluation report.

    python examples/quickstart.py --epochs 20 --num 500
"""

from __future__ import annotations

import argparse

from molgen.data import build_dataloaders, load_sample_smiles
from molgen.metrics import evaluate_generation
from molgen.molgpt import MolGPT
from molgen.sampling import sample
from molgen.tokenizers import SmilesTokenizer
from molgen.trainer import TrainConfig, train_language_model
from molgen.utils import get_device, set_seed


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--num", type=int, default=500)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    set_seed(args.seed)
    device = get_device()
    smiles = load_sample_smiles()
    tokenizer = SmilesTokenizer.from_smiles(smiles)
    train_loader, val_loader = build_dataloaders(
        smiles, tokenizer, batch_size=64, augment=True, seed=args.seed
    )

    model = MolGPT(
        tokenizer.vocab_size,
        embedding_dim=128,
        nhead=4,
        hidden_dim=256,
        num_layers=3,
        pad_idx=tokenizer.pad_id,
    )
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Training MolGPT ({n_params:,} params) on {len(smiles)} molecules ({device})...")
    train_language_model(
        model, train_loader, val_loader, TrainConfig(epochs=args.epochs), device, tokenizer.pad_id
    )

    print(f"Sampling {args.num} molecules...")
    generated = sample(model, tokenizer, num_samples=args.num, top_p=0.95, device=device)

    report = evaluate_generation(generated, reference=smiles)
    print("\n=== Evaluation ===")
    for key, value in report.items():
        if key == "properties":
            print("properties:", {k: round(v, 3) for k, v in value.items()})
        elif isinstance(value, float):
            print(f"{key}: {value:.3f}")
        else:
            print(f"{key}: {value}")


if __name__ == "__main__":
    main()
