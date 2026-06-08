"""Tests for the basic generation metrics."""

import pytest

from molgen.metrics import (
    evaluate_generation,
    internal_diversity,
    novelty,
    snn,
    unique_scaffolds,
    uniqueness,
    validity,
)


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


def test_unique_scaffolds_shares_benzene_ring():
    # Benzene and toluene have the same Bemis-Murcko scaffold (benzene).
    assert unique_scaffolds(["c1ccccc1", "Cc1ccccc1"]) == 0.5


def test_snn_is_one_against_itself():
    molecules = ["c1ccccc1", "CCO", "CC(=O)O"]
    assert snn(molecules, molecules) == pytest.approx(1.0)


def test_snn_lower_for_dissimilar_reference():
    similarity = snn(["c1ccccc1"], ["CCCCCCCCCC"])  # benzene vs. decane
    assert similarity < 0.5


def test_evaluate_generation_report_structure():
    generated = ["CCO", "c1ccccc1", "CC(=O)O", "invalid$$"]
    report = evaluate_generation(generated, reference=["CCO"])
    assert report["n_generated"] == 4
    assert report["validity"] == 0.75
    for key in ("uniqueness", "internal_diversity", "unique_scaffolds", "novelty", "snn"):
        assert key in report
    assert "qed" in report["properties"]


def test_evaluate_generation_without_reference_omits_novelty():
    report = evaluate_generation(["CCO", "c1ccccc1"])
    assert "novelty" not in report
    assert "snn" not in report
