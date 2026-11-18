"""Synthetic benchmark: 2 job offers, 20 CVs, known ground-truth ranking.

The ground truth is NOT produced by an LLM. Each candidate is first sampled as a
set of latent attributes (skills, positions, degree, certifications, soft
skills, languages) with a fixed seed. Per job, a deterministic sub-score in
[0, 1] is computed for each criterion C1..C6 and combined with the expert
weights -> ground-truth score and ranking. Only afterwards does an LLM write
the CV prose from those attributes (with a template fallback), and the
generated text is checked to contain every attribute. This avoids circularity:
the LLM being evaluated never defines the target.

Protected attributes (name, gender, age, nationality) are drawn independently
of the qualifications and only appear in a separable header, which lets us
run a counterfactual fairness test (same body, different header).
"""

from __future__ import annotations

import json
import random
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
from pydantic import BaseModel

from .llm import LLMClient, LLMError

REF_YEAR = 2026

# --------------------------------------------------------------------------
# Job offers
# --------------------------------------------------------------------------

@dataclass
class JobSpec:
    id: str
    title: str
    required_skills: list[str]
    preferred_skills: list[str]
    relevant_domains: list[str]
    min_years: int
    relevant_fields: list[str]
    relevant_certs: list[str]
    soft_skills: list[str]
    required_languages: list[str]
    text: str = ""


JOBS = [
    JobSpec(
        id="J1", title="Data Scientist",
        required_skills=["Python", "SQL", "Machine Learning", "Statistics", "Pandas", "Scikit-learn"],
        preferred_skills=["Deep Learning", "Spark", "Docker"],
        relevant_domains=["data_science", "data_analysis", "ml_engineering"],
        min_years=3,
        relevant_fields=["Computer Science", "Statistics", "Applied Mathematics", "Data Science"],
        relevant_certs=["AWS Certified Machine Learning - Specialty", "Google Professional Data Engineer",
                        "TensorFlow Developer Certificate", "Microsoft Certified: Azure Data Scientist Associate"],
        soft_skills=["communication", "teamwork", "problem solving", "autonomy"],
        required_languages=["English", "French"],
    ),
    JobSpec(
        id="J2", title="Backend Software Engineer (Java)",
        required_skills=["Java", "Spring Boot", "SQL", "REST APIs", "Docker", "Git"],
        preferred_skills=["Kubernetes", "Kafka", "AWS"],
        relevant_domains=["backend", "devops"],
        min_years=4,
        relevant_fields=["Computer Science", "Software Engineering", "Computer Engineering"],
        relevant_certs=["Oracle Certified Professional: Java SE Developer", "AWS Certified Solutions Architect",
                        "Certified Kubernetes Administrator (CKA)"],
        soft_skills=["teamwork", "communication", "leadership", "problem solving"],
        required_languages=["English"],
    ),
]


def job_text(job: JobSpec) -> str:
    """Human-readable job offer built from the spec."""
    langs = " and ".join(job.required_languages)
    return (
        f"Job title: {job.title}\n\n"
        f"We are hiring a {job.title} to join our team.\n\n"
        f"Required skills: {', '.join(job.required_skills)}.\n"
        f"Nice to have: {', '.join(job.preferred_skills)}.\n"
        f"Experience: at least {job.min_years} years of professional experience in a similar role.\n"
        f"Education: degree in {', '.join(job.relevant_fields[:-1])} or {job.relevant_fields[-1]}; "
        f"a Master's degree or PhD is a plus.\n"
        f"Certifications appreciated: {', '.join(job.relevant_certs)}.\n"
        f"Soft skills: {', '.join(job.soft_skills)}.\n"
        f"Languages: {langs} required (professional proficiency)."
    )


for _j in JOBS:
    _j.text = job_text(_j)


# --------------------------------------------------------------------------
# Candidate attributes
# --------------------------------------------------------------------------

ARCHETYPES = {
    "data_science": dict(
        titles=["Junior Data Scientist", "Data Scientist", "Senior Data Scientist"],
        skills=["Python", "SQL", "Machine Learning", "Statistics", "Pandas", "Scikit-learn",
                "Deep Learning", "Spark", "Docker", "R", "Tableau"],
        fields=["Statistics", "Computer Science", "Applied Mathematics", "Data Science"],
        certs=["AWS Certified Machine Learning - Specialty", "TensorFlow Developer Certificate",
               "Microsoft Certified: Azure Data Scientist Associate"]),
    "data_analysis": dict(
        titles=["Data Analyst", "Business Intelligence Analyst"],
        skills=["SQL", "Excel", "Tableau", "Power BI", "Python", "Statistics", "Pandas"],
        fields=["Economics", "Statistics", "Business Administration"],
        certs=["Google Data Analytics Certificate", "Microsoft Certified: Power BI Data Analyst Associate"]),
    "ml_engineering": dict(
        titles=["Machine Learning Engineer", "MLOps Engineer"],
        skills=["Python", "Machine Learning", "Deep Learning", "Docker", "Kubernetes", "Spark",
                "Scikit-learn", "Git", "AWS"],
        fields=["Computer Science", "Computer Engineering"],
        certs=["AWS Certified Machine Learning - Specialty", "Google Professional Data Engineer",
               "Certified Kubernetes Administrator (CKA)"]),
    "backend": dict(
        titles=["Backend Developer", "Java Developer", "Software Engineer"],
        skills=["Java", "Spring Boot", "SQL", "REST APIs", "Docker", "Git", "Kubernetes", "Kafka",
                "AWS", "Hibernate"],
        fields=["Computer Science", "Software Engineering", "Computer Engineering"],
        certs=["Oracle Certified Professional: Java SE Developer", "AWS Certified Solutions Architect"]),
    "frontend": dict(
        titles=["Frontend Developer", "Web Developer"],
        skills=["JavaScript", "TypeScript", "React", "HTML", "CSS", "Git", "REST APIs", "Figma"],
        fields=["Computer Science", "Web Design"],
        certs=["Meta Front-End Developer Certificate"]),
    "devops": dict(
        titles=["DevOps Engineer", "Site Reliability Engineer"],
        skills=["Docker", "Kubernetes", "AWS", "Terraform", "Linux", "Git", "Python", "Jenkins"],
        fields=["Computer Engineering", "Computer Science", "Network Engineering"],
        certs=["Certified Kubernetes Administrator (CKA)", "AWS Certified Solutions Architect"]),
    "accounting": dict(
        titles=["Accountant", "Financial Analyst"],
        skills=["Excel", "SAP", "Financial Reporting", "Budgeting", "IFRS"],
        fields=["Accounting", "Finance"],
        certs=["Certified Public Accountant (CPA)"]),
    "marketing": dict(
        titles=["Marketing Specialist", "Digital Marketing Manager"],
        skills=["SEO", "Google Analytics", "Content Strategy", "Excel", "Social Media Marketing"],
        fields=["Marketing", "Communication"],
        certs=["Google Ads Certification"]),
}

# 20 candidates: the archetype mix makes both offers have strong and weak fits.
ARCHETYPE_PLAN = (["data_science"] * 4 + ["data_analysis"] * 3 + ["ml_engineering"] * 2 +
                  ["backend"] * 4 + ["frontend"] * 2 + ["devops"] * 2 + ["accounting"] * 2 + ["marketing"])

SOFT_POOL = ["communication", "teamwork", "problem solving", "autonomy", "leadership",
             "creativity", "time management"]
LEVELS = ["high_school", "bachelor", "master", "phd"]
LEVEL_SCORE = {"none": 0.0, "high_school": 0.2, "bachelor": 0.6, "master": 0.85, "phd": 1.0}
LEVEL_LABEL = {"high_school": "High School Diploma", "bachelor": "Bachelor's degree",
               "master": "Master's degree", "phd": "PhD"}

# Protected attributes for the counterfactual test (independent of qualifications).
HEADER_A = [("Thomas Martin", "Male", 29, "French"), ("Lucas Bernard", "Male", 31, "French"),
            ("Hugo Petit", "Male", 27, "French"), ("Louis Durand", "Male", 33, "French")]
HEADER_B = [("Fatima Zahra El Amrani", "Female", 54, "Moroccan"), ("Ngozi Okafor", "Female", 52, "Nigerian"),
            ("Aïcha Diallo", "Female", 57, "Senegalese"), ("Mei Lin Chen", "Female", 55, "Chinese")]


@dataclass
class Position:
    title: str
    domain: str
    start: int
    end: int

    @property
    def years(self) -> int:
        return self.end - self.start


@dataclass
class CandidateSpec:
    id: str
    archetype: str
    skills: list[str]
    positions: list[Position]
    degree_level: str
    degree_field: str
    certifications: list[str]
    soft_skills: list[str]
    languages: list[str]
    header_a: tuple = ()
    header_b: tuple = ()
    body: str = ""
    body_source: str = ""   # "llm" or "template"

    def total_years(self) -> int:
        return sum(p.years for p in self.positions)


def sample_candidates(seed: int = 7) -> list[CandidateSpec]:
    rng = random.Random(seed)
    out = []
    for i, arch in enumerate(ARCHETYPE_PLAN):
        a = ARCHETYPES[arch]
        k = rng.randint(3, min(8, len(a["skills"])))
        skills = rng.sample(a["skills"], k)
        # career: 0-3 positions in the archetype, sometimes one earlier position elsewhere
        n_pos = rng.choice([1, 1, 2, 2, 3])
        # titles are ordered junior -> senior: the most recent position gets the most senior title
        durs = [rng.randint(1, 5) for _ in range(n_pos)]
        positions, end = [], REF_YEAR
        for k, dur in enumerate(durs):
            level = min(len(a["titles"]) - 1, n_pos - 1 - k)
            positions.append(Position(a["titles"][level], arch, end - dur, end))
            end -= dur
        if rng.random() < 0.3:
            other = rng.choice([d for d in ARCHETYPES if d != arch])
            dur = rng.randint(1, 3)
            positions.append(Position(rng.choice(ARCHETYPES[other]["titles"]), other, end - dur, end))
        level = rng.choices(LEVELS, weights=[1, 4, 4, 1])[0]
        field = rng.choice(a["fields"])
        certs = rng.sample(a["certs"], min(rng.choice([0, 0, 1, 1, 2]), len(a["certs"])))
        soft = rng.sample(SOFT_POOL, rng.randint(1, 5))
        langs = ["English"] if rng.random() < 0.85 else []
        langs += [l for l in ["French", "Spanish", "Arabic", "German"] if rng.random() < 0.35]
        if not langs:
            langs = [rng.choice(["French", "Spanish", "Arabic"])]
        out.append(CandidateSpec(
            id=f"S{i + 1:02d}", archetype=arch, skills=skills, positions=positions,
            degree_level=level, degree_field=field, certifications=certs, soft_skills=soft,
            languages=langs, header_a=HEADER_A[i % 4], header_b=HEADER_B[i % 4]))
    return out


# --------------------------------------------------------------------------
# Ground truth
# --------------------------------------------------------------------------

def ground_truth_subscores(c: CandidateSpec, job: JobSpec) -> dict[str, float]:
    """Deterministic sub-score in [0, 1] per criterion (the 'true' ratings)."""
    s = set(c.skills)
    c1 = (len(s & set(job.required_skills)) + 0.5 * len(s & set(job.preferred_skills))) / (
        len(job.required_skills) + 0.5 * len(job.preferred_skills))
    relevant_years = sum(p.years for p in c.positions if p.domain in job.relevant_domains)
    c2 = min(relevant_years / (2 * job.min_years), 1.0)
    c3 = LEVEL_SCORE[c.degree_level] * (1.0 if c.degree_field in job.relevant_fields else 0.5)
    c4 = min(len(set(c.certifications) & set(job.relevant_certs)) / 2, 1.0)
    c5 = len(set(c.soft_skills) & set(job.soft_skills)) / len(job.soft_skills)
    c6 = len(set(c.languages) & set(job.required_languages)) / len(job.required_languages)
    return {"C1": c1, "C2": c2, "C3": c3, "C4": c4, "C5": c5, "C6": c6}


def ground_truth_scores(cands: list[CandidateSpec], job: JobSpec, weights: np.ndarray) -> np.ndarray:
    """GT score = sum_j w_j * s_j with crisp, normalised weights (criteria order C1..C6)."""
    sub = np.array([[ground_truth_subscores(c, job)[k] for k in ["C1", "C2", "C3", "C4", "C5", "C6"]]
                    for c in cands])
    return sub @ (weights / weights.sum())


# --------------------------------------------------------------------------
# CV text
# --------------------------------------------------------------------------

def header(h: tuple) -> str:
    name, gender, age, nat = h
    return f"Name: {name}\nGender: {gender}\nAge: {age}\nNationality: {nat}\n"


def template_cv(c: CandidateSpec) -> str:
    """Deterministic CV body (fallback and reference)."""
    lines = ["PROFESSIONAL EXPERIENCE"]
    for p in c.positions:
        lines.append(f"{p.title} ({p.start} - {p.end})")
    lines += ["", "EDUCATION", f"{LEVEL_LABEL[c.degree_level]} in {c.degree_field}", "",
              "SKILLS", ", ".join(c.skills), ""]
    if c.certifications:
        lines += ["CERTIFICATIONS", *c.certifications, ""]
    lines += ["LANGUAGES", ", ".join(f"{l} (professional)" for l in c.languages), "",
              "SOFT SKILLS", ", ".join(c.soft_skills)]
    return "\n".join(lines)


class CVBody(BaseModel):
    cv: str


WRITER_PROMPT = """Write a realistic CV body (plain text, 180-300 words) for a candidate with EXACTLY these facts.
Do NOT add any other skill, tool, degree, certification, language or job. Do NOT include a name,
gender, age, nationality, pronouns, contact details or photo. Use sections:
PROFESSIONAL EXPERIENCE, EDUCATION, SKILLS, CERTIFICATIONS (only if any), LANGUAGES.
For each position write the title, the years (e.g. 2019 - 2023) and 1-2 achievement bullets that use
only the listed skills; demonstrate the listed soft skills through these achievements and
write each soft skill word explicitly.

FACTS:
{facts}

Return JSON: {{"cv": "<the CV text>"}}"""


def _facts(c: CandidateSpec) -> str:
    d = {
        "positions": [f"{p.title} ({p.start} - {p.end})" for p in c.positions],
        "education": f"{LEVEL_LABEL[c.degree_level]} in {c.degree_field}",
        "skills": c.skills, "certifications": c.certifications,
        "languages": c.languages, "soft_skills": c.soft_skills,
    }
    return json.dumps(d, indent=1)


def missing_facts(c: CandidateSpec, text: str) -> list[str]:
    """Attributes that the generated text fails to mention (case-insensitive)."""
    def norm(s: str) -> str:
        return " ".join(s.lower().replace("-", " ").split())

    t = norm(text)
    items = c.skills + c.certifications + c.languages + c.soft_skills + [c.degree_field]
    items += [str(p.start) for p in c.positions]
    return [x for x in items if norm(x) not in t]


def write_cv_body(c: CandidateSpec, client: LLMClient | None, max_tries: int = 3) -> tuple[str, str]:
    if client is None:
        return template_cv(c), "template"
    for attempt in range(max_tries):
        try:
            body = client.complete_json(WRITER_PROMPT.format(facts=_facts(c)), CVBody,
                                        temperature=0.7, seed=1000 + attempt).cv
        except LLMError:
            continue
        if not missing_facts(c, body):
            return body.strip(), "llm"
    return template_cv(c), "template"


def save_benchmark(cands: list[CandidateSpec], out_dir: Path, weights: np.ndarray) -> None:
    """Write CV texts (variant A/B), job offers and ground truth to disk."""
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "cvs").mkdir(exist_ok=True)
    (out_dir / "cvs_variant_b").mkdir(exist_ok=True)
    for c in cands:
        (out_dir / "cvs" / f"{c.id}.txt").write_text(header(c.header_a) + "\n" + c.body, encoding="utf-8")
        (out_dir / "cvs_variant_b" / f"{c.id}.txt").write_text(header(c.header_b) + "\n" + c.body,
                                                              encoding="utf-8")
    jobs = []
    for j in JOBS:
        (out_dir / f"job_{j.id}.txt").write_text(j.text, encoding="utf-8")
        jobs.append(asdict(j))
    (out_dir / "jobs.json").write_text(json.dumps(jobs, indent=1), encoding="utf-8")
    specs = []
    for c in cands:
        d = asdict(c)
        d["positions"] = [asdict(p) for p in c.positions]
        specs.append(d)
    (out_dir / "candidates_spec.json").write_text(json.dumps(specs, indent=1, ensure_ascii=False),
                                                  encoding="utf-8")
    rows = []
    for j in JOBS:
        gt = ground_truth_scores(cands, j, weights)
        order = np.argsort(-gt, kind="stable")
        rank = np.empty(len(gt), int)
        rank[order] = np.arange(1, len(gt) + 1)
        for c, s, r in zip(cands, gt, rank):
            sub = ground_truth_subscores(c, j)
            rows.append({"job_id": j.id, "candidate_id": c.id, "archetype": c.archetype,
                         "gt_score": round(float(s), 6), "gt_rank": int(r), **sub})
    import pandas as pd

    pd.DataFrame(rows).to_csv(out_dir / "ground_truth.csv", index=False)
