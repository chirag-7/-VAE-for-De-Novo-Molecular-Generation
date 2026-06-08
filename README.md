# Molecule-Generator

[![CI](https://github.com/DaoyuanLi2816/Molecule-Generator/actions/workflows/ci.yml/badge.svg)](https://github.com/DaoyuanLi2816/Molecule-Generator/actions/workflows/ci.yml)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Code style: ruff](https://img.shields.io/badge/code%20style-ruff-000000.svg)](https://github.com/astral-sh/ruff)

A lightweight, modern toolkit for **de novo molecular generation** with deep
sequence models. It provides atom-level SMILES and SELFIES tokenizers, several
generator architectures, a mixed-precision training loop, configurable
sampling, and a MOSES-style evaluation suite — small enough to train on a single
GPU in minutes, but reflecting current practice.

![Image](./molecule.png)

## Features

- **Representations** — atom-aware regex SMILES tokenizer and a SELFIES
  tokenizer (every sequence decodes to a valid molecule).
- **Models** — a Transformer β-TC-VAE, a GRU/LSTM `CharRNN`, and a
  decoder-only `MolGPT`.
- **Training** — teacher-forced loop with AdamW, gradient clipping, and
  automatic mixed precision (AMP) on CUDA.
- **Sampling** — autoregressive generation with temperature, top-k, and
  top-p (nucleus) filtering.
- **Metrics** — validity, uniqueness, novelty, internal diversity, unique
  scaffolds, SNN, and QED / logP / MW / SA-score property summaries.
- **Tooling** — `molgen` CLI, a bundled sample dataset, tests, CI, and ruff.

## Installation

```bash
git clone https://github.com/DaoyuanLi2816/Molecule-Generator.git
cd Molecule-Generator
pip install -e .            # add ".[selfies]" for SELFIES, ".[dev]" for tests
```

## Quickstart (Python)

```python
from molgen.data import build_dataloaders, load_sample_smiles
from molgen.tokenizers import SmilesTokenizer
from molgen.molgpt import MolGPT
from molgen.trainer import TrainConfig, train_language_model
from molgen.sampling import sample
from molgen.metrics import evaluate_generation

smiles = load_sample_smiles()                      # bundled sample, or your own list
tokenizer = SmilesTokenizer.from_smiles(smiles)
train_loader, val_loader = build_dataloaders(smiles, tokenizer, augment=True)

model = MolGPT(tokenizer.vocab_size, pad_idx=tokenizer.pad_id)
train_language_model(model, train_loader, val_loader, TrainConfig(epochs=20), pad_idx=tokenizer.pad_id)

generated = sample(model, tokenizer, num_samples=1000, top_p=0.95)
print(evaluate_generation(generated, reference=smiles))
```

## Quickstart (CLI)

```bash
molgen train  --data molecules.smi --model molgpt --epochs 20 --out model.pt
molgen sample --checkpoint model.pt --num 1000 --top-p 0.95 --out generated.smi
molgen eval   --generated generated.smi --reference molecules.smi
```

## Models

| Model | Module | Description |
|-------|--------|-------------|
| `CharRNN` | `molgen.char_rnn` | GRU/LSTM next-token language model (classic strong baseline) |
| `MolGPT` | `molgen.molgpt` | Decoder-only Transformer with causal attention |
| `BetaTCVAE` | `molgen.vae` | Transformer VAE for reconstruction and latent interpolation |

Both `CharRNN` and `MolGPT` train and sample through the same trainer/sampler.

## Latent-space exploration (VAE)

The original VAE workflow is still available for generating molecules near a
seed or interpolating between two molecules in latent space:

```bash
python -m molgen.synthetic     # build a synthetic dataset (molecules.csv)
python -m molgen.vae           # train the VAE
python -m molgen.generate      # perturb the latent space
python -m molgen.interpolate   # interpolate between two molecules
```

## Notes

The bundled `load_sample_smiles()` set is **synthetic** (assembled from
fragments) and intended for examples and tests; for real results, train on a
dataset such as MOSES, QM9, or ZINC. SELFIES mode guarantees 100% validity;
SMILES mode tends to learn the data distribution more faithfully.

## Contributing

Contributions are welcome — see [CONTRIBUTING.md](CONTRIBUTING.md). Please run
`ruff check .`, `ruff format .`, and `pytest` before opening a pull request.

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE).
