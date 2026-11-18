"""CV parsing (PDF/TXT), anonymisation and LLM structured extraction.

Flow:  file -> raw text -> clean -> anonymise -> LLM -> CandidateProfile (pydantic)

Anonymisation happens *before* any LLM sees the text, so protected attributes
(name, gender, age, nationality, contact details) cannot influence scoring.
This is a structural fairness guarantee rather than a prompt instruction.
"""

from __future__ import annotations

import hashlib
import io
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field, field_validator

from .llm import LLMClient

MAX_CV_CHARS = 6000  # keeps prompts inside small local models' context windows


# --------------------------------------------------------------------------
# Schema
# --------------------------------------------------------------------------

DegreeLevel = Literal["none", "high_school", "associate", "bachelor", "master", "phd", "other"]


class Education(BaseModel):
    degree: str = Field(description="Degree title as written in the CV")
    field: str = Field(default="", description="Field of study")
    level: DegreeLevel = Field(default="other")


class CandidateProfile(BaseModel):
    skills: list[str] = Field(default_factory=list, description="Technical / hard skills and tools")
    years_experience: float = Field(default=0.0, ge=0, le=60,
                                    description="Total years of professional experience")
    job_titles: list[str] = Field(default_factory=list, description="Previous job titles")
    education: list[Education] = Field(default_factory=list)
    certifications: list[str] = Field(default_factory=list)
    languages: list[str] = Field(default_factory=list, description="Spoken languages")
    projects: list[str] = Field(default_factory=list, description="One short line per project")
    soft_skills: list[str] = Field(default_factory=list)

    @field_validator("years_experience", mode="before")
    @classmethod
    def _coerce_years(cls, v):
        if v is None or v == "":
            return 0.0
        if isinstance(v, str):
            m = re.search(r"\d+(\.\d+)?", v)
            return float(m.group()) if m else 0.0
        return v


@dataclass
class Candidate:
    id: str
    filename: str
    raw_text: str
    text: str                       # cleaned + anonymised text (what the LLM sees)
    removed: list[str] = field(default_factory=list)
    profile: CandidateProfile | None = None


# --------------------------------------------------------------------------
# Parsing
# --------------------------------------------------------------------------

def read_pdf(data: bytes) -> str:
    import pdfplumber

    with pdfplumber.open(io.BytesIO(data)) as pdf:
        return "\n".join((p.extract_text() or "") for p in pdf.pages)


def read_cv(source: str | Path | bytes, filename: str | None = None) -> str:
    """Return the text of a CV given a path, or raw bytes + filename."""
    if isinstance(source, (str, Path)):
        path = Path(source)
        filename = filename or path.name
        data = path.read_bytes()
    else:
        data = source
    if (filename or "").lower().endswith(".pdf") or data[:4] == b"%PDF":
        return read_pdf(data)
    for enc in ("utf-8", "latin-1"):
        try:
            return data.decode(enc)
        except UnicodeDecodeError:
            continue
    return data.decode("utf-8", errors="ignore")


def clean_text(text: str) -> str:
    text = text.replace("\r", "\n").replace(" ", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n\s*\n+", "\n\n", text)
    return text.strip()


# --------------------------------------------------------------------------
# Anonymisation
# --------------------------------------------------------------------------

_PATTERNS: list[tuple[str, re.Pattern, str]] = [
    ("email", re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+"), "[EMAIL]"),
    ("url", re.compile(r"https?://\S+|www\.\S+|linkedin\.com/\S+"), "[URL]"),
    # labelled personal fields: "Name: ...", "Age: 34", "Gender: F", "Nationality: ..."
    ("labelled_personal", re.compile(
        r"^\s*(name|full name|nom|age|âge|gender|sex|sexe|nationality|nationalité|citizenship|"
        r"date of birth|dob|birth ?date|born|marital status|religion|photo|address|adresse)"
        r"\s*[:\-].*$", re.I | re.M), ""),
    ("age", re.compile(r"\b\d{2}\s*(years old|yrs old|ans)\b", re.I), "[AGE]"),
]
_PHONE = re.compile(r"(?<!\w)\+?\d[\d ().-]{7,}\d(?!\w)")
_YEAR_RANGE = re.compile(r"(\d{1,2}[/.])?(19|20)\d{2}\s*[-–]\s*(\d{1,2}[/.])?(19|20)\d{2}")
# Gendered pronouns / titles are neutralised (approximate: "her" -> "their").
_GENDER_WORDS = {
    r"\bhe\b": "they", r"\bshe\b": "they", r"\bhim\b": "them", r"\bhis\b": "their",
    r"\bher\b": "their", r"\bhers\b": "theirs", r"\bhimself\b": "themselves",
    r"\bherself\b": "themselves", r"\bmr\.?\s": "", r"\bmrs\.?\s": "", r"\bms\.?\s": "",
    r"\bmale\b": "", r"\bfemale\b": "",
}


def anonymize(text: str) -> tuple[str, list[str]]:
    """Remove contact details and protected attributes. Returns (text, removed_categories)."""
    removed: list[str] = []

    def _phone(m: re.Match) -> str:
        s = m.group(0)
        # date ranges such as "2015 - 2019" or "01/2018 - 03/2020" are not phone numbers
        if sum(ch.isdigit() for ch in s) < 9 or _YEAR_RANGE.fullmatch(s.strip()):
            return s
        if "phone" not in removed:
            removed.append("phone")
        return "[PHONE]"

    text = _PHONE.sub(_phone, text)
    for name, pat, repl in _PATTERNS:
        if pat.search(text):
            removed.append(name)
            text = pat.sub(repl, text)
    for pat, repl in _GENDER_WORDS.items():
        if re.search(pat, text, flags=re.I):
            if "gender_words" not in removed:
                removed.append("gender_words")
            text = re.sub(pat, repl, text, flags=re.I)
    return clean_text(text), removed


# --------------------------------------------------------------------------
# LLM extraction
# --------------------------------------------------------------------------

EXTRACTION_SYSTEM = (
    "You are an HR information-extraction engine. Extract facts from the CV exactly as "
    "stated. Never invent information. If something is absent, use an empty list or 0. "
    "Do not output any personal or demographic information (name, gender, age, nationality)."
)

EXTRACTION_PROMPT = """Extract a structured profile from the CV below.

Rules:
- skills: technical skills, tools, programming languages, methods (short names).
- years_experience: total years of professional work (sum of positions; estimate from dates if needed).
- education.level is one of: none, high_school, associate, bachelor, master, phd, other.
- languages: spoken languages only (not programming languages).
- soft_skills: only those explicitly mentioned or clearly demonstrated.

CV:
\"\"\"
{cv}
\"\"\"
"""


def make_candidate(source: str | Path | bytes, filename: str | None = None,
                   cand_id: str | None = None, do_anonymize: bool = True) -> Candidate:
    raw = clean_text(read_cv(source, filename))
    name = filename or (Path(source).name if isinstance(source, (str, Path)) else "cv")
    text, removed = anonymize(raw) if do_anonymize else (raw, [])
    cid = cand_id or hashlib.sha1(raw.encode("utf-8")).hexdigest()[:8]
    return Candidate(id=cid, filename=name, raw_text=raw, text=text[:MAX_CV_CHARS], removed=removed)


def extract_profile(text: str, client: LLMClient) -> CandidateProfile:
    prompt = EXTRACTION_PROMPT.format(cv=text[:MAX_CV_CHARS])
    return client.complete_json(prompt, CandidateProfile, system=EXTRACTION_SYSTEM, temperature=0.0)


def extract_candidates(candidates: list[Candidate], client: LLMClient) -> list[Candidate]:
    for c in candidates:
        c.profile = extract_profile(c.text, client)
    return candidates
