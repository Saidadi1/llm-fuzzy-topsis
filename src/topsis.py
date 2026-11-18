"""Fuzzy TOPSIS (Chen, 2000) implemented from scratch with numpy,
plus the crisp baselines used in the evaluation (crisp TOPSIS, weighted sum).

Notation
--------
m alternatives (candidates) A_1..A_m, n criteria C_1..C_n.
x~_ij = (a_ij, b_ij, c_ij)  aggregated fuzzy rating of A_i on C_j
w~_j  = (w_j1, w_j2, w_j3)  fuzzy weight of C_j

Reference: C.-T. Chen, "Extensions of the TOPSIS for group decision-making
under fuzzy environment", Fuzzy Sets and Systems 114 (2000) 1-9.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from .fuzzy import validate, vertex_distance

EPS = 1e-12


@dataclass
class FuzzyTopsisResult:
    normalized: np.ndarray   # (m, n, 3)  r~_ij
    weighted: np.ndarray     # (m, n, 3)  v~_ij
    fpis: np.ndarray         # (n, 3)     A*
    fnis: np.ndarray         # (n, 3)     A-
    d_plus: np.ndarray       # (m,)       d_i*
    d_minus: np.ndarray      # (m,)       d_i-
    cc: np.ndarray           # (m,)       CC_i
    rank: np.ndarray         # (m,)       1 = best

    def order(self) -> np.ndarray:
        """Indices of alternatives from best to worst."""
        return np.argsort(self.rank)


def _check_types(types: Sequence[str] | None, n: int) -> list[str]:
    types = list(types) if types is not None else ["benefit"] * n
    if len(types) != n or any(t not in ("benefit", "cost") for t in types):
        raise ValueError("criteria types must be a list of 'benefit'/'cost' of length n")
    return types


def rank_from_scores(scores: np.ndarray) -> np.ndarray:
    """Rank 1 = highest score. Ties are broken by original order (stable)."""
    order = np.argsort(-np.asarray(scores, float), kind="stable")
    rank = np.empty(len(order), dtype=int)
    rank[order] = np.arange(1, len(order) + 1)
    return rank


def fuzzy_normalize(x: np.ndarray, types: Sequence[str]) -> np.ndarray:
    """Step 2 - linear scale normalisation (keeps triangular shape, range [0,1]).

    benefit:  r~_ij = (a_ij / c_j*, b_ij / c_j*, c_ij / c_j*),  c_j* = max_i c_ij
    cost:     r~_ij = (a_j- / c_ij, a_j- / b_ij, a_j- / a_ij), a_j- = min_i a_ij
    """
    r = np.empty_like(x)
    for j, t in enumerate(types):
        col = x[:, j, :]
        if t == "benefit":
            c_star = max(col[:, 2].max(), EPS)
            r[:, j, :] = col / c_star
        else:
            a_minus = col[:, 0].min()
            # reverse the order so that l <= m <= u still holds
            r[:, j, :] = a_minus / np.maximum(col[:, ::-1], EPS)
    return r


def fuzzy_topsis(
    x: np.ndarray,
    w: np.ndarray,
    types: Sequence[str] | None = None,
    ideal: str = "chen",
) -> FuzzyTopsisResult:
    """Run fuzzy TOPSIS on an aggregated decision matrix.

    Parameters
    ----------
    x : (m, n, 3) aggregated fuzzy decision matrix (step 1 is done upstream,
        see `fuzzy.aggregate`).
    w : (n, 3) fuzzy weights.
    types : 'benefit' / 'cost' per criterion (default: all benefit).
    ideal : "chen"   -> FPIS v_j* = (1,1,1), FNIS v_j- = (0,0,0)  (Chen, 2000)
            "minmax" -> FPIS v_j* = (max_i v_ij3,)*3, FNIS v_j- = (min_i v_ij1,)*3
                        (data-driven ideal points, e.g. Chen-Hwang variant)
            "componentwise" -> FPIS/FNIS = component-wise max/min of the v~_ij
    """
    x = np.asarray(x, float)
    w = np.asarray(w, float)
    validate(x)
    validate(w)
    m, n, _ = x.shape
    if w.shape != (n, 3):
        raise ValueError(f"weights must have shape ({n}, 3), got {w.shape}")
    types = _check_types(types, n)

    # Step 2: normalised fuzzy decision matrix R~ = [r~_ij]
    r = fuzzy_normalize(x, types)

    # Step 3: weighted normalised matrix  v~_ij = r~_ij (x) w~_j
    v = r * w[None, :, :]

    # Step 4: fuzzy positive / negative ideal solutions
    if ideal == "chen":
        fpis = np.ones((n, 3))
        fnis = np.zeros((n, 3))
    elif ideal == "minmax":
        fpis = np.repeat(v[:, :, 2].max(axis=0)[:, None], 3, axis=1)
        fnis = np.repeat(v[:, :, 0].min(axis=0)[:, None], 3, axis=1)
    elif ideal == "componentwise":
        # v_j* = (max_i l, max_i m, max_i u), v_j- = (min_i l, min_i m, min_i u)
        # (variant used e.g. by the pyDecision library)
        fpis = v.max(axis=0)
        fnis = v.min(axis=0)
    else:
        raise ValueError(f"unknown ideal type: {ideal}")

    # Step 5: distances with the vertex method
    #   d_i* = sum_j d(v~_ij, v~_j*),   d_i- = sum_j d(v~_ij, v~_j-)
    d_plus = vertex_distance(v, fpis[None, :, :]).sum(axis=1)
    d_minus = vertex_distance(v, fnis[None, :, :]).sum(axis=1)

    # Step 6: closeness coefficient  CC_i = d_i- / (d_i* + d_i-)
    cc = d_minus / np.maximum(d_plus + d_minus, EPS)

    # Step 7: rank by decreasing CC
    return FuzzyTopsisResult(r, v, fpis, fnis, d_plus, d_minus, cc, rank_from_scores(cc))


# --------------------------------------------------------------------------
# Crisp baselines
# --------------------------------------------------------------------------

@dataclass
class CrispResult:
    scores: np.ndarray
    rank: np.ndarray


def crisp_topsis(x: np.ndarray, w: np.ndarray, types: Sequence[str] | None = None) -> CrispResult:
    """Classical TOPSIS (Hwang & Yoon, 1981).

    r_ij = x_ij / sqrt(sum_i x_ij^2)            (vector normalisation)
    v_ij = w_j * r_ij                           (w normalised to sum 1)
    A+_j = max_i v_ij (benefit) / min_i (cost); A-_j the opposite
    S+_i = ||v_i - A+||_2,  S-_i = ||v_i - A-||_2,  C_i = S-_i / (S+_i + S-_i)
    """
    x = np.asarray(x, float)
    w = np.asarray(w, float)
    m, n = x.shape
    types = _check_types(types, n)
    w = w / max(w.sum(), EPS)
    r = x / np.maximum(np.sqrt((x ** 2).sum(axis=0)), EPS)
    v = r * w
    benefit = np.array([t == "benefit" for t in types])
    a_pos = np.where(benefit, v.max(axis=0), v.min(axis=0))
    a_neg = np.where(benefit, v.min(axis=0), v.max(axis=0))
    s_pos = np.sqrt(((v - a_pos) ** 2).sum(axis=1))
    s_neg = np.sqrt(((v - a_neg) ** 2).sum(axis=1))
    c = s_neg / np.maximum(s_pos + s_neg, EPS)
    return CrispResult(c, rank_from_scores(c))


def weighted_sum(x: np.ndarray, w: np.ndarray, types: Sequence[str] | None = None) -> CrispResult:
    """Simple Additive Weighting: S_i = sum_j w_j * r_ij with min-max normalisation
    (cost criteria reversed). Weights are normalised to sum to 1."""
    x = np.asarray(x, float)
    w = np.asarray(w, float)
    m, n = x.shape
    types = _check_types(types, n)
    w = w / max(w.sum(), EPS)
    lo, hi = x.min(axis=0), x.max(axis=0)
    span = np.where(hi - lo > EPS, hi - lo, 1.0)
    r = (x - lo) / span
    for j, t in enumerate(types):
        if t == "cost":
            r[:, j] = 1.0 - r[:, j]
        if hi[j] - lo[j] <= EPS:  # constant criterion carries no information
            r[:, j] = 0.0
    s = r @ w
    return CrispResult(s, rank_from_scores(s))
