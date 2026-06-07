# Contributing to Molecule-Generator

Thanks for your interest in improving this project! Contributions of all kinds
are welcome — bug reports, fixes, new models, metrics, and documentation.

## Development setup

```bash
git clone https://github.com/DaoyuanLi2816/Molecule-Generator.git
cd Molecule-Generator
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
pre-commit install
```

This installs `molgen` in editable mode with the development tools (ruff,
pytest, pre-commit). RDKit is pulled in automatically; see the
[RDKit install guide](https://www.rdkit.org/docs/Install.html) if you hit
platform-specific issues.

## Running checks locally

```bash
ruff check .            # lint
ruff format .           # auto-format
pytest                  # run the test suite
```

The pre-commit hooks run the lint/format checks automatically on `git commit`.
CI (`.github/workflows/ci.yml`) runs the same checks on Python 3.10–3.12.

## Submitting changes

1. Create a topic branch off `main` (e.g. `fix/loss-padding-mask`).
2. Make your change and add tests where it makes sense.
3. Ensure `ruff check .`, `ruff format --check .`, and `pytest` all pass.
4. Open a pull request describing **what** changed and **why**, and how you
   verified it.

## Code style

- Formatted and linted with [ruff](https://docs.astral.sh/ruff/) (line length 100).
- Prefer small, focused pull requests with a clear single purpose.
- Keep public functions documented with concise docstrings.

## License

By contributing, you agree that your contributions will be licensed under the
project's [MIT License](LICENSE).
