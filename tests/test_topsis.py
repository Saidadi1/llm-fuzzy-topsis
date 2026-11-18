import numpy as np
import pytest

from src.reference_examples import (ALTERNATIVES, EXPECTED_CC, EXPECTED_D_MINUS, EXPECTED_RANKING,
                                    PRINTED_TABLE5_W, PRINTED_TABLE5_X, RECOMPUTED_D_PLUS, chen_inputs)
from src.topsis import crisp_topsis, fuzzy_normalize, fuzzy_topsis, rank_from_scores, weighted_sum


def test_chen_2000_aggregation_matches_table5_where_paper_is_consistent():
    x, w = chen_inputs()
    np.testing.assert_allclose(x[0, 0], [5.67, 7.67, 9.33], atol=0.01)   # printed (5.7, 7.7, 9.3)
    np.testing.assert_allclose(x[0, 3], [8.33, 9.67, 10], atol=0.01)
    np.testing.assert_allclose(x[2, 4], [6.33, 8.33, 9.67], atol=0.01)   # printed (6.3, 8.3, 9.7)
    np.testing.assert_allclose(w[2], [0.77, 0.93, 1.0], atol=0.005)
    np.testing.assert_allclose(w[4], [0.43, 0.63, 0.83], atol=0.005)


def test_chen_2000_normalized_table6_from_printed_table5():
    r = fuzzy_normalize(PRINTED_TABLE5_X, ["benefit"] * 5)
    np.testing.assert_allclose(r[0, 0], [0.59, 0.79, 0.96], atol=0.01)
    np.testing.assert_allclose(r[2, 4], [0.63, 0.83, 0.97], atol=0.01)
    np.testing.assert_allclose(r[1, 1], [0.9, 1, 1], atol=0.01)


def test_chen_2000_from_printed_table5():
    """Same inputs as the paper -> same d-, CC2, CC3 and ranking; d1* equals the
    value recomputed from the paper's own Table 7 (the printed 2.10 is an erratum)."""
    res = fuzzy_topsis(PRINTED_TABLE5_X, PRINTED_TABLE5_W)
    np.testing.assert_allclose(res.d_minus, EXPECTED_D_MINUS, atol=0.015)
    np.testing.assert_allclose(res.d_plus, RECOMPUTED_D_PLUS, atol=0.01)
    np.testing.assert_allclose(res.cc[1:], EXPECTED_CC[1:], atol=0.005)
    assert res.cc[0] == pytest.approx(0.64, abs=0.005)
    assert [ALTERNATIVES[i] for i in res.order()] == EXPECTED_RANKING


def test_chen_2000_from_table4_ratings():
    """Full pipeline from the linguistic ratings of Table 4: same ranking as the paper."""
    x, w = chen_inputs()
    res = fuzzy_topsis(x, w)
    assert [ALTERNATIVES[i] for i in res.order()] == EXPECTED_RANKING
    np.testing.assert_allclose(res.cc, EXPECTED_CC, atol=0.03)


def test_componentwise_variant_matches_pydecision():
    pd_alg = pytest.importorskip("pyDecision.algorithm")
    for x, w in [chen_inputs(), (PRINTED_TABLE5_X, PRINTED_TABLE5_W)]:
        dataset = [[tuple(x[i, j]) for j in range(5)] for i in range(3)]
        cc_ref = np.asarray(pd_alg.fuzzy_topsis_method(dataset, [[tuple(v) for v in w]], ["max"] * 5,
                                                       graph=False, verbose=False)).ravel()
        np.testing.assert_allclose(fuzzy_topsis(x, w, ideal="componentwise").cc, cc_ref, atol=1e-12)


def test_dominant_alternative_ranks_first():
    x = np.array([[[0.75, 1, 1]] * 3, [[0.25, 0.5, 0.75]] * 3, [[0, 0, 0.25]] * 3])
    w = np.tile([0.5, 0.75, 1.0], (3, 1))
    res = fuzzy_topsis(x, w)
    assert res.rank.tolist() == [1, 2, 3]
    assert np.all((res.cc >= 0) & (res.cc <= 1))


def test_cost_criterion_reverses_preference():
    x = np.array([[[1, 2, 3]], [[4, 5, 6]]], float)
    w = np.array([[1, 1, 1]], float)
    assert fuzzy_topsis(x, w, ["benefit"]).rank.tolist() == [2, 1]
    assert fuzzy_topsis(x, w, ["cost"]).rank.tolist() == [1, 2]
    r = fuzzy_normalize(x, ["cost"])
    assert np.all(r[..., 0] <= r[..., 1]) and np.all(r[..., 1] <= r[..., 2])


def test_minmax_ideal_variant_runs():
    x, w = chen_inputs()
    res = fuzzy_topsis(x, w, ideal="minmax")
    assert [ALTERNATIVES[i] for i in res.order()][0] == "A2"


def test_weight_shape_check():
    x, w = chen_inputs()
    with pytest.raises(ValueError):
        fuzzy_topsis(x, w[:3])


def test_crisp_topsis_textbook():
    # Dominated alternative must get C = 0, dominating one C = 1
    x = np.array([[1, 1], [2, 2], [3, 3]], float)
    res = crisp_topsis(x, np.array([0.5, 0.5]))
    assert res.scores[0] == pytest.approx(0) and res.scores[2] == pytest.approx(1)
    assert res.rank.tolist() == [3, 2, 1]


def test_weighted_sum():
    x = np.array([[0, 10], [10, 0], [5, 5]], float)
    s = weighted_sum(x, np.array([3.0, 1.0])).scores
    np.testing.assert_allclose(s, [0.25, 0.75, 0.5])


def test_rank_ties_are_stable():
    assert rank_from_scores([0.5, 0.9, 0.5]).tolist() == [2, 1, 3]
