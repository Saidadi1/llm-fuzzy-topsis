"""End-to-end pipeline: candidates + job offer + weights -> ranking table.

Used by the Streamlit app and by the experiment scripts so that both run
exactly the same code.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np
import pandas as pd

from .explain import Explanation, explain_candidate
from .extraction import Candidate, extract_profile
from .fuzzy import centroid
from .llm import LLMClient
from .scoring import Criterion, ScoredCandidate, decision_matrix, score_candidate
from .topsis import FuzzyTopsisResult, fuzzy_topsis


@dataclass
class PipelineResult:
    candidates: list[Candidate]
    criteria: list[Criterion]
    scored: list[ScoredCandidate]
    x: np.ndarray                     # (m, n, 3) fuzzy decision matrix
    w: np.ndarray                     # (n, 3) fuzzy weights
    topsis: FuzzyTopsisResult
    explanations: dict[str, Explanation] = field(default_factory=dict)

    def table(self) -> pd.DataFrame:
        rows = []
        for i, (c, s) in enumerate(zip(self.candidates, self.scored)):
            row = {"candidate_id": c.id, "file": c.filename, "rank": int(self.topsis.rank[i]),
                   "CC": float(self.topsis.cc[i]), "d_plus": float(self.topsis.d_plus[i]),
                   "d_minus": float(self.topsis.d_minus[i]), "run_agreement": s.agreement()}
            for j, cr in enumerate(self.criteria):
                row[f"{cr.id}_term"] = s.final_terms[cr.id]
                row[f"{cr.id}_crisp"] = float(centroid(s.fuzzy[j]))
                row[f"{cr.id}_verified"] = s.verified[cr.id]
            if c.id in self.explanations:
                e = self.explanations[c.id]
                row["explanation"] = e.summary
                row["strengths"] = "; ".join(e.strengths)
                row["weaknesses"] = "; ".join(e.weaknesses)
            rows.append(row)
        return pd.DataFrame(rows).sort_values("rank").reset_index(drop=True)


def run_pipeline(
    candidates: list[Candidate],
    job_text: str,
    criteria: list[Criterion],
    w_fuzzy: np.ndarray,
    client: LLMClient,
    weight_terms: dict[str, str] | None = None,
    n_runs: int = 1,
    unverified_policy: str = "downgrade",
    extract: bool = True,
    explain: bool = True,
    ideal: str = "chen",
    progress: Callable[[str, float], None] | None = None,
) -> PipelineResult:
    def tick(msg: str, frac: float) -> None:
        if progress:
            progress(msg, frac)

    m = len(candidates)
    steps = m * (1 + int(extract) + int(explain))
    done = 0
    if extract:
        for c in candidates:
            if c.profile is None:
                tick(f"Extracting profile {c.filename}", done / steps)
                c.profile = extract_profile(c.text, client)
            done += 1

    scored = []
    for c in candidates:
        tick(f"Scoring {c.filename}", done / steps)
        scored.append(score_candidate(c.id, c.text, job_text, criteria, client,
                                      n_runs=n_runs, unverified_policy=unverified_policy))
        done += 1

    x = decision_matrix(scored)
    res = fuzzy_topsis(x, w_fuzzy, [cr.type for cr in criteria], ideal=ideal)
    out = PipelineResult(candidates, criteria, scored, x, w_fuzzy, res)

    if explain:
        best = float(res.cc.max())
        for i, (c, s) in enumerate(zip(candidates, scored)):
            tick(f"Explaining {c.filename}", done / steps)
            out.explanations[c.id] = explain_candidate(
                s, int(res.rank[i]), m, float(res.cc[i]), best, criteria, weight_terms or {}, client)
            done += 1
    tick("Done", 1.0)
    return out
