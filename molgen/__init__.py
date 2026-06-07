"""molgen: a lightweight VAE-based molecular SMILES string generator.

See https://github.com/DaoyuanLi2816/Molecule-Generator for documentation.
"""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("molgen")
except PackageNotFoundError:  # pragma: no cover - running from a source checkout
    __version__ = "0.1.0"

__all__ = ["__version__"]
