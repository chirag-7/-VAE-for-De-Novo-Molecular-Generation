"""Tests for the basic generation metrics."""

from molgen.metrics import internal_diversity, novelty, uniqueness, validity


def test_validity_fraction():
    assert validity(["CCO", "c1ccccc1", "not_valid", "C(C"]) == 0.5
    assert validity([]) == 0.0


def test_uniqueness_dedups_canonically():
    # "CCO" and "OCC" are the same molecule; 3 valid -> 2 unique.
    assert uniqueness(["CCO", "OCC", "c1ccccc1"]) == 2 / 3


def test_novelty_against_reference():
    # benzene is novel, ethanol is in the reference.
    assert novelty(["CCO", "c1ccccc1"], reference=["CCO"]) == 0.5


def test_internal_diversity_in_unit_range():
    div = internal_diversity(["CCO", "c1ccccc1", "CCCCCC", "CC(=O)O"])
    assert 0.0 <= div <= 1.0


def test_internal_diversity_zero_for_identical():
    assert internal_diversity(["CCO", "CCO", "CCO"]) < 1e-6
