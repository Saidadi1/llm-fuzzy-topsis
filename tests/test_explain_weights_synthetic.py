import numpy as np

from src.explain import explain_candidate, find_protected_terms
from src.scoring import load_criteria, score_candidate
from src.synthetic import (JOBS, ground_truth_scores, ground_truth_subscores, missing_facts,
                           sample_candidates, template_cv)
from src.weights import EXPERT_PRESET, combine, crisp_weights, suggest_weights, terms_to_fuzzy

CV = "Python and SQL developer. Master's degree in Statistics."


def _scored(make_client):
    crit = load_criteria()
    ans = {"scores": [{"criterion_id": f"C{i}", "term": "High", "evidence": "Python and SQL"} for i in range(1, 7)]}
    return crit, score_candidate("x", CV, "job", crit, make_client([ans]))


def test_protected_term_detection():
    assert find_protected_terms("The candidate is strong in Python") == []
    assert set(find_protected_terms("She is young and French nationality")) >= {"she", "young", "nationality"}
    assert find_protected_terms("international management, renamed, other") == []


def test_explanation_rejects_protected_attributes(make_client):
    crit, sc = _scored(make_client)
    bad = {"summary": "He is a strong candidate", "strengths": [], "weaknesses": []}
    good = {"summary": "The candidate ranks first thanks to skills.", "strengths": ["Python"], "weaknesses": []}
    client = make_client([bad, good])
    e = explain_candidate(sc, 1, 5, 0.8, 0.8, crit, EXPERT_PRESET, client)
    assert e.summary.startswith("The candidate")
    assert "protected attributes" in client._backend.prompts[1]
    assert "Python and SQL" in client._backend.prompts[0]      # evidence is passed, not the CV


def test_explanation_fallback_is_safe(make_client):
    crit, sc = _scored(make_client)
    e = explain_candidate(sc, 2, 5, 0.6, 0.8, crit, EXPERT_PRESET, make_client(["oops"]))
    assert find_protected_terms(e.summary) == []


def test_weights(make_client):
    crit = load_criteria()
    w = terms_to_fuzzy(EXPERT_PRESET, crit)
    assert w.shape == (6, 3)
    cw = crisp_weights(w)
    assert np.isclose(cw.sum(), 1) and cw[0] == cw.max()
    np.testing.assert_allclose(combine(w, w), w)
    ans = {"weights": [{"criterion_id": c.id, "term": "High", "reason": ""} for c in crit]}
    sw = suggest_weights("job", crit, make_client([ans]))
    assert set(sw) == {c.id for c in crit}


def test_synthetic_benchmark_is_deterministic_and_discriminative():
    a, b = sample_candidates(7), sample_candidates(7)
    assert [c.skills for c in a] == [c.skills for c in b]
    assert len(a) == 20
    w = crisp_weights(terms_to_fuzzy(EXPERT_PRESET, load_criteria()))
    for job in JOBS:
        gt = ground_truth_scores(a, job, w)
        assert gt.std() > 0.05                     # the benchmark separates candidates
        assert all(0 <= v <= 1 for c in a for v in ground_truth_subscores(c, job).values())
    # a data-science candidate should beat an accountant on J1
    ds = next(c for c in a if c.archetype == "data_science")
    acc = next(c for c in a if c.archetype == "accounting")
    assert ground_truth_subscores(ds, JOBS[0])["C1"] > ground_truth_subscores(acc, JOBS[0])["C1"]
    # the template CV contains every fact
    assert all(missing_facts(c, template_cv(c)) == [] for c in a)
