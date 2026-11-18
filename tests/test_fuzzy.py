import numpy as np
import pytest

from src.fuzzy import (CHEN_RATING_SCALE, RATING_SCALE_5, aggregate, centroid, membership, tfn,
                       validate, vertex_distance)


def test_tfn_validation():
    tfn(0, 0.5, 1)
    with pytest.raises(ValueError):
        tfn(0.6, 0.5, 1)
    with pytest.raises(ValueError):
        validate(np.zeros((2, 2)))


def test_vertex_distance_paper_examples():
    # Chen (2000), Property 3 / Fig. 4: A=(1,3,5), B=(2,4,7), C=(5,7,9)
    a, b, c = tfn(1, 3, 5), tfn(2, 4, 7), tfn(5, 7, 9)
    assert vertex_distance(a, b) == pytest.approx(np.sqrt(2))
    assert vertex_distance(a, c) == pytest.approx(4.0)


def test_vertex_distance_reduces_to_euclidean_for_crisp_numbers():
    # Property 1: for real numbers d(m, n) = |m - n|
    assert vertex_distance(tfn(3, 3, 3), tfn(7, 7, 7)) == pytest.approx(4.0)


def test_vertex_distance_broadcasts():
    x = np.array([[[0, 0, 0], [1, 1, 1]]], float)  # (1, 2, 3)
    d = vertex_distance(x, np.ones(3))
    assert d.shape == (1, 2)
    assert d[0, 1] == 0


def test_aggregation_methods():
    r = np.array([RATING_SCALE_5.to_tfn("Low"), RATING_SCALE_5.to_tfn("High")])
    np.testing.assert_allclose(aggregate(r, "mean"), [0.25, 0.5, 0.75])
    np.testing.assert_allclose(aggregate(r, "minmeanmax"), [0.0, 0.5, 1.0])
    g = aggregate(np.array([[1.0, 2.0, 4.0], [4.0, 8.0, 16.0]]), "geometric")
    np.testing.assert_allclose(g, [2.0, 4.0, 8.0])


def test_minmeanmax_widens_with_disagreement():
    same = np.array([RATING_SCALE_5.to_tfn("Medium")] * 3)
    mixed = np.array([RATING_SCALE_5.to_tfn(t) for t in ["Low", "Medium", "High"]])
    width = lambda a: a[2] - a[0]
    assert width(aggregate(mixed, "minmeanmax")) > width(aggregate(same, "minmeanmax"))


def test_scale_aliases_and_order():
    assert RATING_SCALE_5.to_tfn("vh").tolist() == [0.75, 1.0, 1.0]
    assert RATING_SCALE_5.to_tfn("très élevé").tolist() == [0.75, 1.0, 1.0]
    assert RATING_SCALE_5.index("Medium") == 2
    with pytest.raises(KeyError):
        RATING_SCALE_5.to_tfn("excellent")
    crisp = [RATING_SCALE_5.crisp(t) for t in RATING_SCALE_5.labels]
    assert crisp == sorted(crisp)


def test_to_tfns_shape_and_nearest_term():
    arr = RATING_SCALE_5.to_tfns([["Low", "High"], ["Medium", "Very Low"]])
    assert arr.shape == (2, 2, 3)
    assert RATING_SCALE_5.nearest_term(np.array([0.26, 0.5, 0.74])) == "Medium"
    assert CHEN_RATING_SCALE.to_tfn("VG").tolist() == [9, 10, 10]


def test_membership():
    a = tfn(0, 0.5, 1)
    np.testing.assert_allclose(membership([-1, 0, 0.25, 0.5, 0.75, 1, 2], a), [0, 0, 0.5, 1, 0.5, 0, 0])
    # degenerate left side (l == m)
    assert membership(0.0, tfn(0, 0, 0.25)) == pytest.approx(1.0)


def test_centroid():
    assert centroid(tfn(0, 0.5, 1)) == pytest.approx(0.5)
    assert centroid(tfn(0.75, 1, 1)) == pytest.approx(11 / 12)
