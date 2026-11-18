import numpy as np
import pytest

from src.scoring import (CriterionScore, apply_grounding, evidence_is_grounded, load_criteria,
                         score_candidate)

CV = "Data Scientist (2019 - 2023): built churn models in Python and SQL. Master's degree in Statistics."


def answer(terms, evidence=None):
    ev = evidence or {}
    return {"scores": [{"criterion_id": f"C{i + 1}", "term": t, "evidence": ev.get(f"C{i + 1}", ""),
                        "rationale": "r"} for i, t in enumerate(terms)]}


def test_criteria_config():
    crit = load_criteria()
    assert [c.id for c in crit] == ["C1", "C2", "C3", "C4", "C5", "C6"]
    assert all(c.type == "benefit" for c in crit)


def test_term_normalisation():
    assert CriterionScore(criterion_id="C1", term="very high").term == "Very High"
    assert CriterionScore(criterion_id="C1", term="VL", evidence=None).evidence == ""


def test_grounding():
    assert evidence_is_grounded("built churn models in Python", CV)
    assert evidence_is_grounded("Built churn-models in python!", CV)
    assert not evidence_is_grounded("10 years of Kubernetes at Google", CV)
    assert not evidence_is_grounded("", CV)


def test_grounding_policies():
    assert apply_grounding("High", False, "flag") == "High"
    assert apply_grounding("High", False, "downgrade") == "Medium"
    assert apply_grounding("High", False, "very_low") == "Very Low"
    assert apply_grounding("High", True, "very_low") == "High"
    assert apply_grounding("Very Low", False, "downgrade") == "Very Low"


def test_score_candidate_single_run(make_client):
    crit = load_criteria()
    ev = {"C1": "churn models in Python and SQL", "C2": "Data Scientist (2019 - 2023)",
          "C3": "Master's degree in Statistics", "C5": "invented leadership award"}
    client = make_client([answer(["High", "Medium", "High", "Very Low", "High", "Low"], ev)])
    sc = score_candidate("x", CV, "job", crit, client, n_runs=1, unverified_policy="downgrade")
    assert sc.final_terms["C1"] == "High"
    assert sc.final_terms["C5"] == "Medium"         # ungrounded -> downgraded
    assert sc.verified["C4"] and not sc.verified["C5"]
    assert not sc.verified["C6"]                     # 'Low' without evidence is ungrounded
    np.testing.assert_allclose(sc.fuzzy[0], [0.5, 0.75, 1.0])
    assert sc.agreement() == 1.0


def test_missing_criterion_triggers_retry(make_client):
    crit = load_criteria()
    partial = {"scores": answer(["High"] * 6)["scores"][:4]}
    client = make_client([partial, answer(["High"] * 6, {f"C{i}": "Python" for i in range(1, 7)})])
    sc = score_candidate("x", CV, "job", crit, client)
    assert len(sc.final_terms) == 6
    assert "missing criteria" in client._backend.prompts[1]


def test_stability_runs_aggregate_min_mean_max(make_client):
    crit = load_criteria()
    ev = {f"C{i}": "Python" for i in range(1, 7)}
    client = make_client([answer(["Low"] * 6, ev), answer(["Medium"] * 6, ev), answer(["High"] * 6, ev)])
    sc = score_candidate("x", CV, "job", crit, client, n_runs=3)
    np.testing.assert_allclose(sc.fuzzy[0], [0.0, 0.5, 1.0])   # min l, mean m, max u
    assert sc.agreement() == 0.0
