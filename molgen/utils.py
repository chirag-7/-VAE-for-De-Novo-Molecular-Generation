"""Shared utilities: reproducibility, device selection, and logging."""

from __future__ import annotations

import logging
import os
import random

import numpy as np
import torch

__all__ = ["set_seed", "get_device", "get_logger"]


def set_seed(seed: int, deterministic: bool = False) -> None:
    """Seed the Python, NumPy, and PyTorch RNGs for reproducible runs.

    Args:
        seed: The random seed to apply to all RNGs.
        deterministic: If True, also request deterministic cuDNN/cuBLAS
            behaviour. This is slower but makes GPU runs reproducible.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        # Required for deterministic cuBLAS matmuls on CUDA >= 10.2.
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")


def get_device(prefer_cuda: bool = True) -> torch.device:
    """Return the best available torch device (CUDA if present, else CPU)."""
    if prefer_cuda and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def get_logger(name: str = "molgen", level: int = logging.INFO) -> logging.Logger:
    """Return a configured logger that attaches a single stderr handler."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(
            logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")
        )
        logger.addHandler(handler)
        logger.setLevel(level)
        logger.propagate = False
    return logger
