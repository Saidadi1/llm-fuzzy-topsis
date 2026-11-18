"""Metrics, baselines, sensitivity analysis and fairness checks."""

from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd
from scipy.stats import kendalltau, spearmanr
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from .fuzzy import centroid
from .topsis import crisp_topsis, fuzzy_topsis, rank_from_scores, weighted_sum
from .weights import crisp_weights, scale_fuzzy_weight


# --------------------------------------------------------------------------
# Rank agreement metrics
# --------------------------------------------------------------------------

def spearman(a: Sequence[float], b: Sequence[float]) -> float:
    """Spearman rho between two score (or rank) vectors."""
    return float(spearmanr(a, b).statistic)


def kendall(a: Sequence[float], b: Sequence[float]) -> float:
    """Kendall tau-b (handles ties)."""
    return float(kendalltau(a, b).statistic)


def top_k_overlap(scores_pred: Sequence[float], scores_true: Sequence[float], k: int) -> float:
    """|top-k(pred) ∩ top-k(true)| / k  ('top-k accuracy' of the shortlist)."""
    p = set(np.argsort(-np.asarray(scores_pred), kind="stable")[:k])
    t = set(np.argsort(-np.asarray(scores_true), kind="stable")[:k])
    return len(p & t) / k


def top1_hit(scores_pred: Sequence[float], scores_true: Sequence[float]) -> float:
    return float(int(np.argmax(scores_pred)) == int(np.argmax(scores_true)))


def precision_at_k(scores: Sequence[float], relevant: Sequence[bool], k: int) -> float:
    """Share of relevant items among the k best-scored (weak-label evaluation)."""
    idx = np.argsort(-np.asarray(scores), kind="stable")[:k]
    return float(np.mean(np.asarray(relevant)[idx]))


def compare_to_reference(methods: dict[str, Sequence[float]], reference: Sequence[float],
                         ks: Sequence[int] = (3, 5)) -> pd.DataFrame:
    """One row per method: Spearman, Kendall, top-k overlap with the reference scores."""
    rows = []
    for name, s in methods.items():
        row = {"method": name, "spearman": spearman(s, reference), "kendall_tau": kendall(s, reference),
               "top1": top1_hit(s, reference)}
        for k in ks:
            row[f"top{k}_overlap"] = top_k_overlap(s, reference, k)
        rows.append(row)
    return pd.DataFrame(rows).set_index("method")


# --------------------------------------------------------------------------
# Baselines
# --------------------------------------------------------------------------

def tfidf_cosine(cv_texts: Sequence[str], job_text: str) -> np.ndarray:
    """Keyword baseline without any LLM: cosine similarity between TF-IDF
    vectors (unigrams + bigrams, English stop words) of each CV and the offer."""
    vec = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), sublinear_tf=True)
    m = vec.fit_transform(list(cv_texts) + [job_text])
    return cosine_similarity(m[:-1], m[-1]).ravel()


def all_methods(x_fuzzy: np.ndarray, w_fuzzy: np.ndarray, types: Sequence[str],
                cv_texts: Sequence[str] | None = None, job_text: str | None = None) -> dict[str, np.ndarray]:
    """Scores of every method on the same LLM ratings.

    - fuzzy_topsis : Chen (2000) on the fuzzy matrix and fuzzy weights
    - crisp_topsis : classical TOPSIS on centroid-defuzzified ratings / weights
    - weighted_sum : SAW on the same defuzzified data
    - tfidf_cosine : no LLM, keyword similarity CV <-> offer
    """
    x_crisp = centroid(x_fuzzy)
    w_crisp = crisp_weights(w_fuzzy)
    out = {
        "fuzzy_topsis": fuzzy_topsis(x_fuzzy, w_fuzzy, types).cc,
        "crisp_topsis": crisp_topsis(x_crisp, w_crisp, types).scores,
        "weighted_sum": weighted_sum(x_crisp, w_crisp, types).scores,
    }
    if cv_texts is not None and job_text is not None:
        out["tfidf_cosine"] = tfidf_cosine(cv_texts, job_text)
    return out


# --------------------------------------------------------------------------
# Sensitivity analysis on the weights
# --------------------------------------------------------------------------

def sensitivity_one_at_a_time(x: np.ndarray, w: np.ndarray, types: Sequence[str],
                              criteria_ids: Sequence[str],
                              factors: Sequence[float] = (0.5, 0.7, 0.9, 1.1, 1.3, 1.5)) -> pd.DataFrame:
    """Scale one criterion weight at a time and measure ranking stability
    (Kendall tau vs. the base ranking, whether the top-1 candidate changes)."""
    base = fuzzy_topsis(x, w, types)
    rows = []
    for j, cid in enumerate(criteria_ids):
        for f in factors:
            r = fuzzy_topsis(x, scale_fuzzy_weight(w, j, f), types)
            rows.append({"criterion": cid, "factor": f, "kendall_tau": kendall(r.cc, base.cc),
                         "top1_same": bool(np.argmax(r.cc) == np.argmax(base.cc)),
                         "max_rank_shift": int(np.abs(r.rank - base.rank).max())})
    return pd.DataFrame(rows)


def sensitivity_scenarios(x: np.ndarray, w: np.ndarray, types: Sequence[str],
                          criteria_ids: Sequence[str]) -> pd.DataFrame:
    """Classical scenario analysis: equal weights, then each criterion made
    dominant (Very High = (0.75,1,1)) while the others are set Low = (0,0.25,0.5).
    Returns the rank of every alternative in every scenario."""
    n = len(criteria_ids)
    scenarios = {"base": w, "equal": np.tile([0.25, 0.5, 0.75], (n, 1))}
    for j, cid in enumerate(criteria_ids):
        wj = np.tile([0.0, 0.25, 0.5], (n, 1))
        wj[j] = [0.75, 1.0, 1.0]
        scenarios[f"{cid}_dominant"] = wj
    ranks = {name: fuzzy_topsis(x, ws, types).rank for name, ws in scenarios.items()}
    return pd.DataFrame(ranks)


def monte_carlo_weights(x: np.ndarray, w: np.ndarray, types: Sequence[str], n_samples: int = 1000,
                        noise: float = 0.2, seed: int = 42) -> pd.DataFrame:
    """Perturb every weight by a random factor U(1-noise, 1+noise); report for each
    alternative how often it ranks first and its mean / std rank."""
    rng = np.random.default_rng(seed)
    m = x.shape[0]
    ranks = np.empty((n_samples, m), dtype=int)
    for s in range(n_samples):
        f = rng.uniform(1 - noise, 1 + noise, size=(w.shape[0], 1))
        ranks[s] = fuzzy_topsis(x, np.clip(w * f, 0, 1), types).rank
    return pd.DataFrame({"p_rank1": (ranks == 1).mean(0), "mean_rank": ranks.mean(0),
                         "std_rank": ranks.std(0)})


# --------------------------------------------------------------------------
# Fairness
# --------------------------------------------------------------------------

def counterfactual_report(cc_a: Sequence[float], cc_b: Sequence[float]) -> dict[str, float]:
    """Compare closeness coefficients of the same CVs that differ only in protected
    attributes (variant A vs. variant B). An unbiased system gives identical CCs."""
    a, b = np.asarray(cc_a, float), np.asarray(cc_b, float)
    ra, rb = rank_from_scores(a), rank_from_scores(b)
    diff = b - a
    return {
        "mean_delta_cc": float(diff.mean()),
        "mean_abs_delta_cc": float(np.abs(diff).mean()),
        "max_abs_delta_cc": float(np.abs(diff).max()),
        "share_identical_cc": float(np.mean(np.abs(diff) < 1e-9)),
        "kendall_tau_A_vs_B": kendall(a, b),
        "mean_abs_rank_shift": float(np.abs(ra - rb).mean()),
    }


def group_gap(scores: Sequence[float], groups: Sequence[str], k: int) -> pd.DataFrame:
    """Mean score and top-k selection rate per group (demographic-parity view)."""
    s = np.asarray(scores, float)
    top = np.zeros(len(s), dtype=bool)
    top[np.argsort(-s, kind="stable")[:k]] = True
    df = pd.DataFrame({"score": s, "group": list(groups), "selected": top})
    return df.groupby("group").agg(mean_score=("score", "mean"), selection_rate=("selected", "mean"),
                                   n=("score", "size"))
