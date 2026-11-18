"""LLM scoring of candidates against job criteria with linguistic terms.

For each candidate the LLM returns, for every criterion, one linguistic term
(Very Low .. Very High), a verbatim evidence quote from the CV and a one-line
rationale. Terms are mapped to triangular fuzzy numbers (fuzzy.RATING_SCALE_5).

Design choices
--------------
* One call per candidate scores all criteria at once: fewer calls, and the
  model sees the whole profile when calibrating its terms.
* Evidence grounding: every quote is checked against the CV text with fuzzy
  string matching. An ungrounded quote means the score is not supported by the
  CV (a likely hallucination) and is handled by `unverified_policy`.
* Prompt stability: with n_runs > 1 we query the model several times
  (run 0 at temperature 0, others at a higher temperature with other seeds)
  and treat the runs as K decision makers. Aggregation with "minmeanmax"
  turns the LLM's inconsistency into the spread (l, u) of the fuzzy number.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import numpy as np
import yaml
from pydantic import BaseModel, Field, field_validator
from rapidfuzz import fuzz

from .config import CONFIG_DIR
from .fuzzy import RATING_SCALE_5, LinguisticScale, aggregate
from .llm import LLMClient

Term = Literal["Very Low", "Low", "Medium", "High", "Very High"]
TERMS: list[str] = list(RATING_SCALE_5.labels)


@dataclass(frozen=True)
class Criterion:
    id: str
    name: str
    description: str
    type: str = "benefit"


def load_criteria(path: str | Path | None = None) -> list[Criterion]:
    path = Path(path) if path else CONFIG_DIR / "criteria.yaml"
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    return [Criterion(**c) for c in data["criteria"]]


# --------------------------------------------------------------------------
# Schema
# --------------------------------------------------------------------------

class CriterionScore(BaseModel):
    # Field order matters: with constrained decoding the model writes the evidence
    # and the rationale BEFORE choosing the term (evidence-first, a light chain of thought).
    criterion_id: str
    evidence: str = Field(default="", description="Short verbatim quote copied from the CV, '' if none")
    rationale: str = Field(default="", description="One short sentence")
    term: Term

    @field_validator("term", mode="before")
    @classmethod
    def _canonical_term(cls, v):
        # accept "very high", "VH", "élevé"... and map to the canonical label
        try:
            return RATING_SCALE_5.labels[RATING_SCALE_5.index(str(v))]
        except KeyError:
            return v

    @field_validator("evidence", "rationale", mode="before")
    @classmethod
    def _none_to_empty(cls, v):
        return "" if v is None else v


class CandidateScores(BaseModel):
    scores: list[CriterionScore]


@dataclass
class ScoredCandidate:
    candidate_id: str
    runs: list[dict[str, CriterionScore]]            # one dict criterion_id -> score per run
    final_terms: dict[str, str] = field(default_factory=dict)  # after grounding policy, run 0
    verified: dict[str, bool] = field(default_factory=dict)    # evidence grounded? (run 0)
    fuzzy: np.ndarray | None = None                  # (n_criteria, 3) aggregated TFNs

    def agreement(self) -> float:
        """Share of criteria on which all runs gave the same term (1.0 for a single run)."""
        if len(self.runs) < 2:
            return 1.0
        crits = self.runs[0].keys()
        return float(np.mean([len({r[c].term for r in self.runs}) == 1 for c in crits]))


# --------------------------------------------------------------------------
# Prompting
# --------------------------------------------------------------------------

SCORING_SYSTEM = (
    "You are a fair, evidence-based technical recruiter. You rate a candidate against a job "
    "offer using ONLY facts written in the CV. Ignore and never use name, gender, age, "
    "nationality, ethnicity, religion, marital status or photos. If the CV gives no evidence "
    "for a criterion, rate it 'Very Low' with an empty evidence string."
)

SCORING_PROMPT = """JOB OFFER:
\"\"\"
{job}
\"\"\"

CRITERIA:
{criteria}

Rating scale (use exactly these terms): Very Low, Low, Medium, High, Very High.
- Very High: fully meets or exceeds the job's requirement for this criterion
- High: meets most of the requirement
- Medium: partially meets it
- Low: weak or marginal match
- Very Low: no evidence / irrelevant

CANDIDATE CV:
\"\"\"
{cv}
\"\"\"

Return one entry per criterion id ({ids}). For each, in this order:
"criterion_id";
"evidence": a short phrase COPIED VERBATIM from the CANDIDATE CV (never from the job offer),
  max 25 words, e.g. "Data Scientist (2019 - 2023)" or "Master's degree in Statistics"; "" if none;
"rationale": one short sentence comparing that evidence with the requirement;
"term": the rating."""


def _criteria_block(criteria: list[Criterion]) -> str:
    return "\n".join(f"- {c.id} {c.name}: {c.description}" for c in criteria)


def _normalise(s: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^\w\s]", " ", s.lower())).strip()


def evidence_is_grounded(evidence: str, cv_text: str, threshold: float = 85.0) -> bool:
    """True if `evidence` appears (approximately) in the CV.

    partial_ratio finds the best matching substring of the CV, which tolerates
    small typos/whitespace changes but not paraphrases or invented facts.
    """
    ev = _normalise(evidence)
    if not ev:
        return False
    return fuzz.partial_ratio(ev, _normalise(cv_text)) >= threshold


def score_once(cv_text: str, job_text: str, criteria: list[Criterion], client: LLMClient,
               temperature: float = 0.0, seed: int | None = None) -> dict[str, CriterionScore]:
    ids = [c.id for c in criteria]

    def _all_criteria(obj: CandidateScores) -> None:
        got = [s.criterion_id.strip().upper() for s in obj.scores]
        missing = [i for i in ids if i not in got]
        if missing:
            raise ValueError(f"missing criteria {missing}; return exactly one entry per id {ids}")

    prompt = SCORING_PROMPT.format(job=job_text.strip(), criteria=_criteria_block(criteria),
                                   cv=cv_text.strip(), ids=", ".join(ids))
    obj = client.complete_json(prompt, CandidateScores, system=SCORING_SYSTEM,
                               temperature=temperature, seed=seed, validator=_all_criteria)
    out: dict[str, CriterionScore] = {}
    for s in obj.scores:
        cid = s.criterion_id.strip().upper()
        if cid in ids and cid not in out:
            out[cid] = s
    return out


def apply_grounding(term: str, grounded: bool, policy: str) -> str:
    """What to do with a score whose evidence is not found in the CV.

    - "flag":      keep the term, only report it
    - "downgrade": lower the term by one level
    - "very_low":  treat the criterion as unsupported -> Very Low
    A 'Very Low' term needs no evidence, so it is never modified.
    """
    if grounded or term == "Very Low" or policy == "flag":
        return term
    if policy == "downgrade":
        return TERMS[max(TERMS.index(term) - 1, 0)]
    if policy == "very_low":
        return "Very Low"
    raise ValueError(f"unknown policy {policy}")


def score_candidate(
    candidate_id: str,
    cv_text: str,
    job_text: str,
    criteria: list[Criterion],
    client: LLMClient,
    n_runs: int = 1,
    stability_temperature: float = 0.7,
    seed: int = 42,
    unverified_policy: str = "downgrade",
    aggregation: str = "minmeanmax",
    scale: LinguisticScale = RATING_SCALE_5,
) -> ScoredCandidate:
    runs = []
    for k in range(n_runs):
        temp = 0.0 if k == 0 else stability_temperature
        runs.append(score_once(cv_text, job_text, criteria, client, temperature=temp, seed=seed + k))

    return fuzzy_from_runs(candidate_id, runs, cv_text, criteria, unverified_policy, aggregation, scale)


def fuzzy_from_runs(
    candidate_id: str,
    runs: list[dict[str, CriterionScore]],
    cv_text: str,
    criteria: list[Criterion],
    unverified_policy: str = "downgrade",
    aggregation: str = "minmeanmax",
    scale: LinguisticScale = RATING_SCALE_5,
) -> ScoredCandidate:
    """Grounding check + term -> TFN mapping + aggregation over runs.

    Separated from the LLM calls so that ablations (grounding policy, number of
    runs, aggregation operator) are recomputed from stored answers for free.
    """
    sc = ScoredCandidate(candidate_id=candidate_id, runs=runs)
    tfns = []
    for k, run in enumerate(runs):
        row = []
        for c in criteria:
            s = run[c.id]
            grounded = evidence_is_grounded(s.evidence, cv_text)
            term = apply_grounding(s.term, grounded, unverified_policy)
            if k == 0:
                sc.final_terms[c.id] = term
                sc.verified[c.id] = grounded or s.term == "Very Low"
            row.append(scale.to_tfn(term))
        tfns.append(row)
    # (K, n, 3) -> (n, 3): the K runs play the role of K decision makers
    sc.fuzzy = aggregate(np.array(tfns), aggregation if len(runs) > 1 else "mean")
    return sc


def decision_matrix(scored: list[ScoredCandidate]) -> np.ndarray:
    """Stack aggregated TFNs into the (m, n, 3) fuzzy decision matrix."""
    return np.stack([s.fuzzy for s in scored])
