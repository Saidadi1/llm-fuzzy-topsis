import numpy as np
import pytest

from src.evaluation import (all_methods, compare_to_reference, counterfactual_report, kendall,
                            monte_carlo_weights, precision_at_k, sensitivity_one_at_a_time,
                            sensitivity_scenarios, spearman, tfidf_cosine, top_k_overlap)
from src.reference_examples import chen_inputs


def test_rank_metrics():
    a = [0.9, 0.8, 0.1, 0.5]
    assert spearman(a, a) == pytest.approx(1)
    assert kendall(a, [-v for v in a]) == pytest.approx(-1)
    assert top_k_overlap(a, [0.9, 0.1, 0.8, 0.5], 2) == 0.5
    assert precision_at_k([0.9, 0.1, 0.8], [True, True, False], 2) == 0.5


def test_tfidf_prefers_matching_cv():
    s = tfidf_cosine(["python sql machine learning", "cooking pastry chef"], "python machine learning engineer")
    assert s[0] > s[1]


def test_all_methods_and_reference():
    x, w = chen_inputs()
    x01 = x / 10.0
    m = all_methods(x01, w, ["benefit"] * 5)
    assert set(m) == {"fuzzy_topsis", "crisp_topsis", "weighted_sum"}
    df = compare_to_reference(m, m["fuzzy_topsis"], ks=(1, 2))
    assert df.loc["fuzzy_topsis", "spearman"] == pytest.approx(1)


def test_sensitivity_and_scenarios():
    x, w = chen_inputs()
    oat = sensitivity_one_at_a_time(x, w, ["benefit"] * 5, [f"C{i}" for i in range(1, 6)])
    assert len(oat) == 5 * 6 and oat["kendall_tau"].between(-1, 1).all()
    sc = sensitivity_scenarios(x, w, ["benefit"] * 5, [f"C{i}" for i in range(1, 6)])
    assert sc.shape == (3, 7)
    mc = monte_carlo_weights(x, w, ["benefit"] * 5, n_samples=50)
    assert mc["p_rank1"].sum() == pytest.approx(1)


def test_counterfactual_report():
    r = counterfactual_report([0.5, 0.6, 0.7], [0.5, 0.6, 0.7])
    assert r["share_identical_cc"] == 1 and r["mean_abs_rank_shift"] == 0
    r = counterfactual_report([0.5, 0.6, 0.7], [0.4, 0.6, 0.8])
    assert r["max_abs_delta_cc"] == pytest.approx(0.1)
