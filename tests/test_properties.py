"""Tests for molecular property calculators."""

from molgen.properties import logp, molecular_weight, property_summary, qed, sa_score

ASPIRIN = "CC(=O)Oc1ccccc1C(=O)O"


def test_qed_logp_mw_for_aspirin():
    assert 0.0 <= qed(ASPIRIN) <= 1.0
    assert abs(molecular_weight(ASPIRIN) - 180.16) < 0.5
    assert isinstance(logp(ASPIRIN), float)


def test_properties_none_for_invalid():
    assert qed("nope$$") is None
    assert logp("nope$$") is None
    assert molecular_weight("nope$$") is None
    assert sa_score("nope$$") is None


def test_sa_score_in_expected_range():
    score = sa_score("CCO")
    if score is not None:  # SA scorer ships with RDKit contrib
        assert 1.0 <= score <= 10.0


def test_property_summary_keys_and_ranges():
    summary = property_summary(["CCO", "c1ccccc1", "nope$$"])
    assert {"qed", "logp", "mol_weight", "sa_score"} <= set(summary)
    assert 0.0 <= summary["qed"] <= 1.0
    assert summary["mol_weight"] > 0
