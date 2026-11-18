import json
import re

import numpy as np

from src.extraction import make_candidate
from src.pipeline import run_pipeline
from src.scoring import load_criteria
from src.weights import EXPERT_PRESET, terms_to_fuzzy

CVS = {
    "strong.txt": "Senior Data Scientist (2016 - 2026). Python, SQL, Machine Learning, Statistics. "
                  "Master's degree in Statistics. AWS Certified Machine Learning. English, French. teamwork",
    "weak.txt": "Accountant (2022 - 2024). Excel, SAP. Bachelor's degree in Accounting. Spanish.",
}


def fake_llm(prompt: str):
    """Rates the 'strong' CV high and the other one low; answers each prompt type."""
    if "Extract a structured profile" in prompt:
        return {"skills": ["Python"], "years_experience": 5}
    if "Per-criterion assessment" in prompt:
        return {"summary": "The candidate is ranked on skills.", "strengths": [], "weaknesses": []}
    strong = "Senior Data Scientist" in prompt
    quote = "Python, SQL, Machine Learning" if strong else "Excel, SAP"
    term = "Very High" if strong else "Low"
    return {"scores": [{"criterion_id": f"C{i}", "evidence": quote, "rationale": "", "term": term}
                       for i in range(1, 7)]}


def test_run_pipeline_end_to_end(make_client, tmp_path):
    for name, text in CVS.items():
        (tmp_path / name).write_text(text, encoding="utf-8")
    cands = [make_candidate(tmp_path / n) for n in CVS]
    crit = load_criteria()
    res = run_pipeline(cands, "Data Scientist job", crit, terms_to_fuzzy(EXPERT_PRESET, crit),
                       make_client([fake_llm]), weight_terms=EXPERT_PRESET, n_runs=1)
    t = res.table()
    assert t.iloc[0]["file"] == "strong.txt" and t.iloc[0]["rank"] == 1
    assert t.iloc[0]["CC"] > t.iloc[1]["CC"]
    assert {"explanation", "C1_term", "C1_verified"} <= set(t.columns)
    assert res.x.shape == (2, 6, 3)


def test_streamlit_app_runs(tmp_path, monkeypatch):
    """Execute app.py headless with a fake LLM backend and click 'Run analysis'."""
    from streamlit.testing.v1 import AppTest

    import src.llm as llm_mod

    def fake_generate(self, prompt, system="", temperature=0.0, seed=None, schema=None):
        return json.dumps(fake_llm(prompt))

    monkeypatch.setattr(llm_mod.LLMClient, "generate", fake_generate)
    from pathlib import Path

    app = Path(__file__).resolve().parents[1] / "app.py"
    at = AppTest.from_file(str(app), default_timeout=120).run()
    assert not at.exception
    run_btn = next(b for b in at.button if b.label == "Run analysis")
    run_btn.click().run()
    assert not at.exception, at.exception
    assert any("Done" in s.value for s in at.success)
