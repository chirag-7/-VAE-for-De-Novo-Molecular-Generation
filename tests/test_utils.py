"""Tests for the shared utilities."""

import random

import numpy as np
import torch

from molgen.utils import get_device, get_logger, set_seed


def test_set_seed_makes_python_reproducible():
    set_seed(42)
    a = [random.random() for _ in range(5)]
    set_seed(42)
    b = [random.random() for _ in range(5)]
    assert a == b


def test_set_seed_makes_numpy_and_torch_reproducible():
    set_seed(7)
    na, ta = np.random.rand(4), torch.rand(4)
    set_seed(7)
    nb, tb = np.random.rand(4), torch.rand(4)
    assert np.allclose(na, nb)
    assert torch.allclose(ta, tb)


def test_get_device_returns_torch_device():
    dev = get_device()
    assert isinstance(dev, torch.device)
    assert dev.type in {"cuda", "cpu"}


def test_get_logger_attaches_single_handler():
    log1 = get_logger("molgen.test")
    log2 = get_logger("molgen.test")
    assert log1 is log2
    assert len(log1.handlers) == 1
