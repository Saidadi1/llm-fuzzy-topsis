"""Criteria weights: fuzzy linguistic weights from an expert preset, an LLM
suggestion, or manual edits (Streamlit sliders), plus crisp conversions.

A weight is a linguistic term of WEIGHT_SCALE_5 mapped to a TFN. Several
experts (or the expert + the LLM) can be combined with fuzzy.aggregate, as
in Chen's eq. (14):  w_j = 1/K (w_j^1 (+) ... (+) w_j^K).
"""

from __future__ import annotations

from typing import Literal

import numpy as np
from pydantic import BaseModel

from .fuzzy import WEIGHT_SCALE_5, aggregate, centroid
from .llm import LLMClient
from .scoring import Criterion

WeightTerm = Literal["Very Low", "Low", "Medium", "High", "Very High"]

# Default "expert" preset for a technical role: skills and experience dominate.
EXPERT_PRESET: dict[str, str] = {
    "C1": "Very High",
    "C2": "High",
    "C3": "Medium",
    "C4": "Low",
    "C5": "Medium",
    "C6": "Medium",
}


def terms_to_fuzzy(terms: dict[str, str], criteria: list[Criterion]) -> np.ndarray:
    """{criterion_id: term} -> (n, 3) array of fuzzy weights in criteria order."""
    return np.array([WEIGHT_SCALE_5.to_tfn(terms[c.id]) for c in criteria])


def combine(*weight_sets: np.ndarray) -> np.ndarray:
    """Average several fuzzy weight vectors (eq. 14)."""
    return aggregate(np.stack(weight_sets), "mean")


def crisp_weights(w_fuzzy: np.ndarray) -> np.ndarray:
    """Defuzzify by centroid and normalise to sum to 1 (for crisp baselines)."""
    w = centroid(w_fuzzy)
    return w / w.sum()


def scale_fuzzy_weight(w_fuzzy: np.ndarray, j: int, factor: float) -> np.ndarray:
    """Multiply the weight of criterion j by `factor` (sensitivity analysis),
    clipping to the [0, 1] universe of the scale."""
    w = w_fuzzy.copy()
    w[j] = np.clip(w[j] * factor, 0.0, 1.0)
    return w


class SuggestedWeight(BaseModel):
    criterion_id: str
    term: WeightTerm
    reason: str = ""


class SuggestedWeights(BaseModel):
    weights: list[SuggestedWeight]


WEIGHT_PROMPT = """You are an HR expert. Given the job offer, rate how IMPORTANT each criterion is
for selecting the right person, using exactly one of: Very Low, Low, Medium, High, Very High.

JOB OFFER:
\"\"\"
{job}
\"\"\"

CRITERIA:
{criteria}

Return one entry per criterion id ({ids}) with "criterion_id", "term", "reason" (one sentence)."""


def suggest_weights(job_text: str, criteria: list[Criterion], client: LLMClient) -> dict[str, SuggestedWeight]:
    ids = [c.id for c in criteria]

    def _complete(obj: SuggestedWeights) -> None:
        got = {w.criterion_id.strip().upper() for w in obj.weights}
        if not set(ids) <= got:
            raise ValueError(f"missing criteria {sorted(set(ids) - got)}")

    prompt = WEIGHT_PROMPT.format(
        job=job_text.strip(),
        criteria="\n".join(f"- {c.id} {c.name}: {c.description}" for c in criteria),
        ids=", ".join(ids),
    )
    obj = client.complete_json(prompt, SuggestedWeights, temperature=0.0, validator=_complete)
    return {w.criterion_id.strip().upper(): w for w in obj.weights if w.criterion_id.strip().upper() in ids}
