"""LLM-generated justification of each candidate's rank.

The LLM never sees the CV here: it only receives the rank, the closeness
coefficient, the linguistic term per criterion, the criterion weights and the
evidence quotes collected during scoring. It therefore cannot cite facts that
were not already grounded in the CV. A post-check rejects explanations that
mention protected attributes; the model is then asked again.
"""

from __future__ import annotations

import re

from pydantic import BaseModel, Field

from .llm import LLMClient, LLMError
from .scoring import Criterion, ScoredCandidate

PROTECTED_PATTERNS = [
    r"\b(gender|male|female|man|woman|he|she|his|her|him)\b",
    r"\b(age|aged|young|old|older|younger|years old)\b",
    r"\b(nationality|national|ethnic\w*|race|religio\w*|citizen\w*|married|marital)\b",
    r"\b(name)\b",
]
_PROTECTED = re.compile("|".join(PROTECTED_PATTERNS), re.I)


class Explanation(BaseModel):
    summary: str = Field(description="2-3 sentences explaining the rank")
    strengths: list[str] = Field(default_factory=list)
    weaknesses: list[str] = Field(default_factory=list)


EXPLAIN_SYSTEM = (
    "You explain the output of a multi-criteria hiring decision model. Use ONLY the evidence "
    "provided. Do not add facts. Never mention or infer gender, age, name, nationality, "
    "ethnicity, religion or marital status. Refer to the person as 'the candidate'."
)

EXPLAIN_PROMPT = """Candidate ranked {rank} of {n} (closeness coefficient CC = {cc:.3f}; best CC in pool = {best:.3f}).

Per-criterion assessment (criterion, importance weight, rating, evidence from the CV):
{rows}

Write: "summary" (2-3 sentences: why this rank, which criteria drove it, referring to the
weights), "strengths" (max 3 short bullet points), "weaknesses" (max 3 short bullet points).
Use only the evidence above."""


def find_protected_terms(text: str) -> list[str]:
    return sorted({m.group(0).lower() for m in _PROTECTED.finditer(text)})


def _check(obj: Explanation) -> None:
    text = " ".join([obj.summary, *obj.strengths, *obj.weaknesses])
    bad = find_protected_terms(text)
    if bad:
        raise ValueError(f"explanation mentions protected attributes {bad}; "
                         "rewrite it without them and call the person 'the candidate'")


def explain_candidate(
    scored: ScoredCandidate,
    rank: int,
    n: int,
    cc: float,
    best_cc: float,
    criteria: list[Criterion],
    weight_terms: dict[str, str],
    client: LLMClient,
) -> Explanation:
    run0 = scored.runs[0]
    rows = "\n".join(
        f"- {c.name}: weight={weight_terms.get(c.id, '?')}, rating={scored.final_terms[c.id]}, "
        f"evidence=\"{run0[c.id].evidence}\"" + ("" if scored.verified[c.id] else " (evidence NOT found in CV)")
        for c in criteria
    )
    prompt = EXPLAIN_PROMPT.format(rank=rank, n=n, cc=cc, best=best_cc, rows=rows)
    try:
        return client.complete_json(prompt, Explanation, system=EXPLAIN_SYSTEM,
                                    temperature=0.0, validator=_check)
    except LLMError:
        # Fallback: deterministic template, guaranteed free of protected terms.
        strong = [c.name for c in criteria if scored.final_terms[c.id] in ("High", "Very High")]
        weak = [c.name for c in criteria if scored.final_terms[c.id] in ("Very Low", "Low")]
        return Explanation(
            summary=f"Ranked {rank}/{n} with CC={cc:.3f}. Strong on: {', '.join(strong) or 'none'}. "
                    f"Weak on: {', '.join(weak) or 'none'}.",
            strengths=strong[:3], weaknesses=weak[:3],
        )
