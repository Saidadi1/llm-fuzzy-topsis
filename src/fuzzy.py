"""Triangular fuzzy numbers (TFN) and linguistic scales.

A TFN a = (l, m, u) with l <= m <= u has the membership function

    mu_a(x) = 0                    x < l
            = (x - l) / (m - l)    l <= x <= m
            = (u - x) / (u - m)    m <= x <= u
            = 0                    x > u

TFNs are stored as numpy arrays whose last axis has length 3, so a whole
decision matrix (m alternatives x n criteria) is simply an array of shape
(m, n, 3). Every operation below is vectorised over the leading axes.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Mapping, Sequence

import numpy as np


# --------------------------------------------------------------------------
# Basic TFN construction and arithmetic
# --------------------------------------------------------------------------

def tfn(l: float, m: float, u: float) -> np.ndarray:
    """Build a validated TFN (l, m, u)."""
    a = np.array([l, m, u], dtype=float)
    validate(a)
    return a


def validate(a: np.ndarray) -> None:
    """Raise ValueError unless every TFN in `a` satisfies l <= m <= u."""
    a = np.asarray(a, dtype=float)
    if a.shape[-1] != 3:
        raise ValueError(f"TFN arrays need a last axis of size 3, got shape {a.shape}")
    if np.any(a[..., 0] > a[..., 1] + 1e-12) or np.any(a[..., 1] > a[..., 2] + 1e-12):
        raise ValueError("TFN must satisfy l <= m <= u")


def add(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """a (+) b = (l1 + l2, m1 + m2, u1 + u2)."""
    return np.asarray(a, float) + np.asarray(b, float)


def multiply(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Approximate product of two non-negative TFNs: (l1*l2, m1*m2, u1*u2)."""
    return np.asarray(a, float) * np.asarray(b, float)


def scale(a: np.ndarray, k: float) -> np.ndarray:
    """k (x) a = (k*l, k*m, k*u) for k >= 0."""
    if k < 0:
        raise ValueError("scalar must be non-negative")
    return np.asarray(a, float) * k


def vertex_distance(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Vertex distance (Chen, 2000):

        d(a, b) = sqrt( 1/3 * [ (l1-l2)^2 + (m1-m2)^2 + (u1-u2)^2 ] )

    Broadcasts over leading axes and returns an array without the last axis.
    """
    diff = np.asarray(a, float) - np.asarray(b, float)
    return np.sqrt(np.mean(diff ** 2, axis=-1))


def centroid(a: np.ndarray) -> np.ndarray:
    """Centroid defuzzification: (l + m + u) / 3."""
    return np.mean(np.asarray(a, float), axis=-1)


def membership(x: np.ndarray | float, a: np.ndarray) -> np.ndarray:
    """Evaluate the triangular membership function of TFN `a` at x."""
    l, m, u = np.asarray(a, float)
    x = np.asarray(x, float)
    left = np.where(m > l, (x - l) / (m - l + 1e-300), (x >= l).astype(float))
    right = np.where(u > m, (u - x) / (u - m + 1e-300), (x <= u).astype(float))
    mu = np.where(x <= m, left, right)
    mu = np.where((x < l) | (x > u), 0.0, mu)
    return np.clip(mu, 0.0, 1.0)


def aggregate(ratings: np.ndarray, method: str = "mean") -> np.ndarray:
    """Aggregate K fuzzy judgements along axis 0.

    `ratings` has shape (K, ..., 3), one slice per decision maker (or per
    repeated LLM run).

    - "mean"        (Chen, 2000):  x = 1/K * (x_1 (+) ... (+) x_K)
    - "minmeanmax"  (common extension): l = min_k l_k, m = mean_k m_k,
                    u = max_k u_k. Keeps the disagreement between judges
                    in the spread of the TFN, which is what we want when the
                    "judges" are repeated, non-deterministic LLM runs.
    - "geometric"   l = (prod l_k)^(1/K), etc.
    """
    r = np.asarray(ratings, float)
    if method == "mean":
        out = r.mean(axis=0)
    elif method == "minmeanmax":
        out = np.stack([r[..., 0].min(axis=0), r[..., 1].mean(axis=0), r[..., 2].max(axis=0)], axis=-1)
    elif method == "geometric":
        out = np.exp(np.log(np.clip(r, 1e-12, None)).mean(axis=0))
        out[r.min(axis=0) == 0] = 0.0  # a zero component stays zero
    else:
        raise ValueError(f"unknown aggregation method: {method}")
    validate(out)
    return out


# --------------------------------------------------------------------------
# Linguistic scales
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class LinguisticScale:
    """An ordered mapping linguistic term -> TFN."""

    name: str
    terms: Mapping[str, tuple[float, float, float]]
    aliases: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for t, v in self.terms.items():
            validate(np.array(v))

    @property
    def labels(self) -> list[str]:
        return list(self.terms.keys())

    def _canonical(self, term: str) -> str:
        key = term.strip()
        lookup = {k.lower(): k for k in self.terms}
        lookup.update({a.lower(): c for a, c in self.aliases.items()})
        try:
            return lookup[key.lower()]
        except KeyError as e:
            raise KeyError(f"'{term}' is not a term of scale {self.name}: {self.labels}") from e

    def to_tfn(self, term: str) -> np.ndarray:
        return np.array(self.terms[self._canonical(term)], dtype=float)

    def to_tfns(self, terms: Iterable) -> np.ndarray:
        """Vectorised: nested lists of terms -> array (..., 3)."""
        arr = np.asarray(terms, dtype=object)
        flat = [self.to_tfn(t) for t in arr.ravel()]
        return np.array(flat, dtype=float).reshape(arr.shape + (3,))

    def index(self, term: str) -> int:
        return self.labels.index(self._canonical(term))

    def crisp(self, term: str) -> float:
        """Centroid of the term's TFN, used by the crisp baselines."""
        return float(centroid(self.to_tfn(term)))

    def nearest_term(self, a: np.ndarray) -> str:
        """Linguistic approximation: term whose TFN is closest (vertex distance)."""
        d = [float(vertex_distance(a, np.array(v))) for v in self.terms.values()]
        return self.labels[int(np.argmin(d))]


_ALIASES_5 = {
    "VL": "Very Low", "L": "Low", "M": "Medium", "H": "High", "VH": "Very High",
    "very_low": "Very Low", "very_high": "Very High", "moderate": "Medium",
    "faible": "Low", "très faible": "Very Low", "moyen": "Medium",
    "élevé": "High", "très élevé": "Very High",
}

# 5-term rating scale on [0, 1] used for LLM scoring of candidates.
RATING_SCALE_5 = LinguisticScale(
    name="rating5",
    terms={
        "Very Low": (0.0, 0.0, 0.25),
        "Low": (0.0, 0.25, 0.5),
        "Medium": (0.25, 0.5, 0.75),
        "High": (0.5, 0.75, 1.0),
        "Very High": (0.75, 1.0, 1.0),
    },
    aliases=_ALIASES_5,
)

# Same shape for criteria importance weights.
WEIGHT_SCALE_5 = LinguisticScale(name="weight5", terms=dict(RATING_SCALE_5.terms), aliases=_ALIASES_5)

# The two 7-term scales of Chen (2000), Tables 1 and 2, used for the
# cross-check against the published numerical example.
CHEN_WEIGHT_SCALE = LinguisticScale(
    name="chen_weight7",
    terms={
        "VL": (0.0, 0.0, 0.1), "L": (0.0, 0.1, 0.3), "ML": (0.1, 0.3, 0.5),
        "M": (0.3, 0.5, 0.7), "MH": (0.5, 0.7, 0.9), "H": (0.7, 0.9, 1.0),
        "VH": (0.9, 1.0, 1.0),
    },
)
CHEN_RATING_SCALE = LinguisticScale(
    name="chen_rating7",
    terms={
        "VP": (0.0, 0.0, 1.0), "P": (0.0, 1.0, 3.0), "MP": (1.0, 3.0, 5.0),
        "F": (3.0, 5.0, 7.0), "MG": (5.0, 7.0, 9.0), "G": (7.0, 9.0, 10.0),
        "VG": (9.0, 10.0, 10.0),
    },
)


def as_matrix(rows: Sequence[Sequence[Sequence[float]]]) -> np.ndarray:
    """Convenience: nested (l, m, u) lists -> validated (m, n, 3) array."""
    a = np.asarray(rows, float)
    validate(a)
    return a
