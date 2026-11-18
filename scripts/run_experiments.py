"""Run every experiment and write tables/figures to results/<tag>/.

    python -m scripts.run_experiments --steps all
    python -m scripts.run_experiments --steps crosscheck
    python -m scripts.run_experiments --steps synthetic,fairness --model qwen3.5:4b

Steps
-----
crosscheck : Chen (2000) example: from-scratch vs. paper vs. pyDecision (+ scikit-fuzzy MF check)
synthetic  : extraction accuracy, LLM scoring (3 runs), fuzzy TOPSIS vs. baselines vs. ground
             truth, ablations (runs, grounding policy, weights source), sensitivity, explanations
fairness   : counterfactual test (same CV body, swapped name/gender/age/nationality header),
             with and without anonymisation
real       : weak-label evaluation on the public Kaggle / HF resume datasets (if downloaded)

All LLM calls go through the SQLite cache, so re-running is free and deterministic.
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import PROCESSED_DIR, RESULTS_DIR, SYNTHETIC_DIR, set_global_seed
from src.evaluation import (all_methods, compare_to_reference, counterfactual_report, kendall,
                            monte_carlo_weights, precision_at_k, sensitivity_one_at_a_time,
                            sensitivity_scenarios, spearman, tfidf_cosine)
from src.explain import explain_candidate, find_protected_terms
from src.extraction import Candidate, anonymize, clean_text, extract_profile
from src.fuzzy import RATING_SCALE_5, WEIGHT_SCALE_5, centroid
from src.llm import LLMClient
from src.scoring import CriterionScore, decision_matrix, fuzzy_from_runs, load_criteria, score_candidate
from src.topsis import fuzzy_topsis
from src.weights import EXPERT_PRESET, crisp_weights, suggest_weights, terms_to_fuzzy

log = logging.getLogger("experiments")
CRITERIA = load_criteria()
CIDS = [c.id for c in CRITERIA]
TYPES = [c.type for c in CRITERIA]


def save(df: pd.DataFrame, out: Path, name: str, index: bool = True) -> pd.DataFrame:
    df.to_csv(out / f"{name}.csv", index=index)
    return df


# ==========================================================================
# 1. Cross-check on Chen (2000)
# ==========================================================================

def step_crosscheck(out: Path) -> None:
    from src.reference_examples import (ALTERNATIVES, EXPECTED_CC, EXPECTED_D_MINUS, EXPECTED_D_PLUS,
                                        PRINTED_TABLE5_W, PRINTED_TABLE5_X, RECOMPUTED_D_PLUS, chen_inputs)

    try:
        from pyDecision.algorithm import fuzzy_topsis_method
    except Exception as e:  # noqa: BLE001
        fuzzy_topsis_method = None
        log.warning("pyDecision unavailable: %s", e)

    def pydecision_cc(x, w):
        dataset = [[tuple(x[i, j]) for j in range(x.shape[1])] for i in range(x.shape[0])]
        weights = [[tuple(w[j]) for j in range(w.shape[0])]]
        return np.asarray(fuzzy_topsis_method(dataset, weights, ["max"] * x.shape[1],
                                              graph=False, verbose=False)).ravel()

    rows = []
    for source, (x, w) in {"table4_ratings": chen_inputs(),
                           "printed_table5": (PRINTED_TABLE5_X, PRINTED_TABLE5_W)}.items():
        chen = fuzzy_topsis(x, w, ideal="chen")
        comp = fuzzy_topsis(x, w, ideal="componentwise")
        for i, a in enumerate(ALTERNATIVES):
            row = {"input": source, "alternative": a,
                   "paper_d_plus": EXPECTED_D_PLUS[i], "recomputed_from_table7_d_plus": RECOMPUTED_D_PLUS[i],
                   "ours_d_plus": round(chen.d_plus[i], 4),
                   "paper_d_minus": EXPECTED_D_MINUS[i], "ours_d_minus": round(chen.d_minus[i], 4),
                   "paper_CC": EXPECTED_CC[i], "ours_CC_chen": round(chen.cc[i], 4), "ours_rank": chen.rank[i],
                   "ours_CC_componentwise": round(comp.cc[i], 6)}
            rows.append(row)
        if fuzzy_topsis_method is not None:
            cc_pd = pydecision_cc(x, w)
            for k, i in enumerate(range(len(rows) - 3, len(rows))):
                rows[i]["pyDecision_CC"] = round(float(cc_pd[k]), 6)
                rows[i]["abs_diff_ours_componentwise_vs_pyDecision"] = float(abs(cc_pd[k] - comp.cc[k]))
    df = pd.DataFrame(rows)
    save(df, out, "crosscheck_chen2000", index=False)

    try:
        import skfuzzy as skf

        from src.fuzzy import membership

        from src.fuzzy import CHEN_RATING_SCALE

        grid = np.linspace(0, 10, 1001)
        errs = [np.abs(skf.trimf(grid, list(v)) - membership(grid, np.array(v))).max()
                for v in CHEN_RATING_SCALE.terms.values()]
        grid01 = np.linspace(0, 1, 1001)
        errs += [np.abs(skf.trimf(grid01, list(v)) - membership(grid01, np.array(v))).max()
                 for v in RATING_SCALE_5.terms.values()]
        # centroid defuzzification: ours (l+m+u)/3 vs skfuzzy.defuzz(..., 'centroid')
        cerr = max(abs(skf.defuzz(grid01, skf.trimf(grid01, list(v)), "centroid") - centroid(np.array(v)))
                   for v in RATING_SCALE_5.terms.values())
        (out / "crosscheck_skfuzzy.txt").write_text(
            f"max |skfuzzy.trimf - ours| over 12 TFNs (Chen 7-term + our 5-term scales), 1001 points: {max(errs):.2e}\n"
            f"max |skfuzzy centroid - (l+m+u)/3| over our 5-term scale (numeric integration): {cerr:.2e}\n"
            "Note: scikit-fuzzy has no TOPSIS; it is used only to validate membership functions and centroids.\n")
    except Exception as e:  # noqa: BLE001
        log.warning("scikit-fuzzy unavailable: %s", e)
    print(df.to_string(index=False))


# ==========================================================================
# Synthetic benchmark helpers
# ==========================================================================

def load_benchmark():
    jobs = json.loads((SYNTHETIC_DIR / "jobs.json").read_text(encoding="utf-8"))
    specs = json.loads((SYNTHETIC_DIR / "candidates_spec.json").read_text(encoding="utf-8"))
    gt = pd.read_csv(SYNTHETIC_DIR / "ground_truth.csv")
    return jobs, specs, gt


LIMIT: int | None = None  # --limit: only the first N synthetic CVs (smoke tests)


def load_candidates(folder: Path, do_anonymize: bool) -> list[Candidate]:
    cands = []
    for p in sorted(folder.glob("*.txt"))[:LIMIT]:
        raw = clean_text(p.read_text(encoding="utf-8"))
        text, removed = anonymize(raw) if do_anonymize else (raw, [])
        cands.append(Candidate(id=p.stem, filename=p.name, raw_text=raw, text=text, removed=removed))
    return cands


def runs_to_json(runs: list[dict[str, CriterionScore]]) -> list[dict]:
    return [{k: v.model_dump() for k, v in r.items()} for r in runs]


def extraction_accuracy(cands: list[Candidate], specs: list[dict]) -> pd.DataFrame:
    """Compare LLM-extracted profiles with the generating attributes."""
    by_id = {s["id"]: s for s in specs}
    rows = []
    norm = lambda s: " ".join(s.lower().replace("-", " ").split())
    for c in cands:
        s, p = by_id[c.id], c.profile
        true_sk, pred_sk = {norm(x) for x in s["skills"]}, {norm(x) for x in p.skills}
        hit = sum(any(t == q or t in q for q in pred_sk) for t in true_sk)
        true_lang = {x.lower() for x in s["languages"]}
        pred_lang = {norm(x).split(" ")[0] for x in p.languages}
        true_years = sum(q["end"] - q["start"] for q in s["positions"])
        rows.append({
            "candidate_id": c.id,
            "skill_recall": hit / max(len(true_sk), 1),
            "skill_precision": sum(any(t == q or t in q for t in true_sk) for q in pred_sk) / max(len(pred_sk), 1),
            "years_true": true_years, "years_pred": p.years_experience,
            "years_abs_err": abs(true_years - p.years_experience),
            "degree_level_ok": any(e.level == s["degree_level"] for e in p.education),
            "language_recall": len(true_lang & pred_lang) / max(len(true_lang), 1),
            "cert_recall": (sum(any(norm(t)[:20] in norm(q) for q in p.certifications) for t in s["certifications"])
                            / len(s["certifications"])) if s["certifications"] else np.nan,
        })
    return pd.DataFrame(rows)


def oracle_scores(gt_job: pd.DataFrame, w: np.ndarray) -> np.ndarray:
    """Fuzzy TOPSIS on the TRUE sub-scores mapped to their nearest linguistic term:
    the best a perfect LLM rater could do with a 5-term scale (upper bound)."""
    x = np.array([[RATING_SCALE_5.to_tfn(RATING_SCALE_5.nearest_term(np.repeat(v, 3)))
                   for v in row] for row in gt_job[CIDS].to_numpy()])
    return fuzzy_topsis(x, w, TYPES).cc


# ==========================================================================
# 2. Synthetic benchmark
# ==========================================================================

def step_synthetic(out: Path, client: LLMClient, n_runs: int, explain: bool) -> None:
    jobs, specs, gt = load_benchmark()
    cands = load_candidates(SYNTHETIC_DIR / "cvs", do_anonymize=True)
    texts = [c.text for c in cands]
    w_expert = terms_to_fuzzy(EXPERT_PRESET, CRITERIA)

    # --- extraction ---------------------------------------------------
    t0 = time.time()
    for c in cands:
        c.profile = extract_profile(c.text, client)
    log.info("extraction done in %.0fs", time.time() - t0)
    (out / "synthetic_profiles.json").write_text(
        json.dumps({c.id: c.profile.model_dump() for c in cands}, indent=1), encoding="utf-8")
    ext = save(extraction_accuracy(cands, specs), out, "extraction_accuracy", index=False)
    save(ext.drop(columns="candidate_id").mean(numeric_only=True).to_frame("mean").T, out, "extraction_accuracy_mean")

    metric_rows, ablation_rows, rating_rows, stab_rows = [], [], [], []
    for job in jobs:
        jid = job["id"]
        gt_job = gt[gt.job_id == jid].set_index("candidate_id").loc[[c.id for c in cands]]
        ref = gt_job["gt_score"].to_numpy()

        # --- LLM-suggested weights -------------------------------------
        sw = suggest_weights(job["text"], CRITERIA, client)
        llm_terms = {k: v.term for k, v in sw.items()}
        w_llm = terms_to_fuzzy(llm_terms, CRITERIA)
        pd.DataFrame({"criterion": CIDS, "expert": [EXPERT_PRESET[c] for c in CIDS],
                      "llm": [llm_terms[c] for c in CIDS],
                      "llm_reason": [sw[c].reason for c in CIDS]}).to_csv(out / f"weights_{jid}.csv", index=False)

        # --- scoring (n_runs per candidate) ------------------------------
        t0 = time.time()
        scored = [score_candidate(c.id, c.text, job["text"], CRITERIA, client, n_runs=n_runs,
                                  unverified_policy="downgrade") for c in cands]
        log.info("%s scoring done in %.0fs", jid, time.time() - t0)
        (out / f"scores_raw_{jid}.json").write_text(
            json.dumps({s.candidate_id: runs_to_json(s.runs) for s in scored}, indent=1), encoding="utf-8")
        x = decision_matrix(scored)

        # --- main comparison --------------------------------------------
        methods = all_methods(x, w_expert, TYPES, texts, job["text"])
        methods["fuzzy_topsis_llm_weights"] = fuzzy_topsis(x, w_llm, TYPES).cc
        methods["fuzzy_topsis_minmax_ideal"] = fuzzy_topsis(x, w_expert, TYPES, ideal="minmax").cc
        methods["oracle_terms_fuzzy_topsis"] = oracle_scores(gt_job, w_expert)
        df = compare_to_reference(methods, ref, ks=(3, 5))
        df.insert(0, "job", jid)
        metric_rows.append(df)

        # --- ranking table ----------------------------------------------
        res = fuzzy_topsis(x, w_expert, TYPES)
        tab = pd.DataFrame({"candidate_id": [c.id for c in cands], "archetype": gt_job["archetype"].values,
                            "gt_score": ref, "gt_rank": gt_job["gt_rank"].values, "CC": res.cc,
                            "rank": res.rank, "d_plus": res.d_plus, "d_minus": res.d_minus,
                            "tfidf": methods["tfidf_cosine"], "agreement": [s.agreement() for s in scored]})
        for j, cid in enumerate(CIDS):
            tab[f"{cid}_term"] = [s.final_terms[cid] for s in scored]
            tab[f"{cid}_fuzzy"] = [tuple(np.round(s.fuzzy[j], 3)) for s in scored]
        tab = tab.sort_values("rank")

        # --- rating quality per criterion (LLM vs true sub-score) --------
        for j, cid in enumerate(CIDS):
            pred = centroid(x[:, j, :])
            true = gt_job[cid].to_numpy()
            rating_rows.append({"job": jid, "criterion": cid, "spearman": spearman(pred, true),
                                "mae": float(np.abs(pred - true).mean()),
                                "evidence_verified_rate": float(np.mean([s.verified[cid] for s in scored]))})

        # --- ablations from stored answers (no extra LLM call) ----------
        for policy in ["flag", "downgrade", "very_low"]:
            for runs_used in sorted({1, n_runs}):
                for agg in (["minmeanmax", "mean"] if runs_used > 1 else ["mean"]):
                    sc2 = [fuzzy_from_runs(s.candidate_id, s.runs[:runs_used], c.text, CRITERIA, policy, agg)
                           for s, c in zip(scored, cands)]
                    cc = fuzzy_topsis(decision_matrix(sc2), w_expert, TYPES).cc
                    ablation_rows.append({"job": jid, "grounding_policy": policy, "runs": runs_used,
                                          "aggregation": agg, "spearman": spearman(cc, ref),
                                          "kendall_tau": kendall(cc, ref)})

        # --- stability across runs ---------------------------------------
        if n_runs > 1:
            per_run_cc = []
            for k in range(n_runs):
                sck = [fuzzy_from_runs(s.candidate_id, [s.runs[k]], c.text, CRITERIA, "downgrade")
                       for s, c in zip(scored, cands)]
                per_run_cc.append(fuzzy_topsis(decision_matrix(sck), w_expert, TYPES).cc)
            for a in range(n_runs):
                for b in range(a + 1, n_runs):
                    stab_rows.append({"job": jid, "run_a": a, "run_b": b,
                                      "kendall_tau": kendall(per_run_cc[a], per_run_cc[b])})
            term_agree = np.mean([s.agreement() for s in scored])
            spread = float(np.mean(x[..., 2] - x[..., 0]))
            stab_rows.append({"job": jid, "run_a": "all", "run_b": "term_agreement", "kendall_tau": term_agree})
            stab_rows.append({"job": jid, "run_a": "all", "run_b": "mean_tfn_width", "kendall_tau": spread})

        # --- sensitivity ---------------------------------------------------
        save(sensitivity_one_at_a_time(x, w_expert, TYPES, CIDS), out, f"sensitivity_oat_{jid}", index=False)
        sc = sensitivity_scenarios(x, w_expert, TYPES, CIDS)
        sc.index = [c.id for c in cands]
        save(sc, out, f"sensitivity_scenarios_{jid}")
        mc = monte_carlo_weights(x, w_expert, TYPES, n_samples=2000, noise=0.3)
        mc.index = [c.id for c in cands]
        save(mc.sort_values("mean_rank"), out, f"sensitivity_montecarlo_{jid}")

        # --- explanations ---------------------------------------------------
        if explain:
            best = float(res.cc.max())
            exps = []
            for i, (c, s) in enumerate(zip(cands, scored)):
                e = explain_candidate(s, int(res.rank[i]), len(cands), float(res.cc[i]), best,
                                      CRITERIA, EXPERT_PRESET, client)
                txt = " ".join([e.summary, *e.strengths, *e.weaknesses])
                exps.append({"candidate_id": c.id, "rank": int(res.rank[i]), "summary": e.summary,
                             "strengths": "; ".join(e.strengths), "weaknesses": "; ".join(e.weaknesses),
                             "protected_terms": ",".join(find_protected_terms(txt))})
            save(pd.DataFrame(exps).sort_values("rank"), out, f"explanations_{jid}", index=False)
            tab = tab.merge(pd.DataFrame(exps)[["candidate_id", "summary"]], on="candidate_id")
        save(tab, out, f"ranking_{jid}", index=False)

    save(pd.concat(metric_rows), out, "synthetic_metrics")
    save(pd.DataFrame(ablation_rows), out, "ablation", index=False)
    save(pd.DataFrame(rating_rows), out, "rating_quality", index=False)
    if stab_rows:
        save(pd.DataFrame(stab_rows), out, "stability", index=False)
    print(pd.concat(metric_rows).round(3).to_string())


# ==========================================================================
# 3. Counterfactual fairness
# ==========================================================================

def step_fairness(out: Path, client: LLMClient, job_ids: list[str]) -> None:
    jobs, _, _ = load_benchmark()
    w = terms_to_fuzzy(EXPERT_PRESET, CRITERIA)
    rows, detail = [], []
    for anonymised in [False, True]:
        ca = load_candidates(SYNTHETIC_DIR / "cvs", anonymised)
        cb = load_candidates(SYNTHETIC_DIR / "cvs_variant_b", anonymised)
        same_text = float(np.mean([a.text == b.text for a, b in zip(ca, cb)]))
        for job in [j for j in jobs if j["id"] in job_ids]:
            cc = {}
            for tag, cands in [("A", ca), ("B", cb)]:
                sc = [score_candidate(c.id, c.text, job["text"], CRITERIA, client, n_runs=1) for c in cands]
                cc[tag] = fuzzy_topsis(decision_matrix(sc), w, TYPES).cc
                for c, s, v in zip(cands, sc, cc[tag]):
                    detail.append({"anonymised": anonymised, "job": job["id"], "variant": tag,
                                   "candidate_id": c.id, "CC": v, **{k: s.final_terms[k] for k in CIDS}})
            r = counterfactual_report(cc["A"], cc["B"])
            rows.append({"anonymised": anonymised, "job": job["id"], "share_identical_text": same_text, **r})
    save(pd.DataFrame(rows), out, "fairness_counterfactual", index=False)
    save(pd.DataFrame(detail), out, "fairness_detail", index=False)
    print(pd.DataFrame(rows).round(4).to_string(index=False))


# ==========================================================================
# 4. Real data (weak labels)
# ==========================================================================

REAL_JOBS = {
    "INFORMATION-TECHNOLOGY": (
        "Job title: IT Support / Systems Specialist\n\nRequired skills: Windows and Linux administration, "
        "networking (TCP/IP, LAN/WAN), Active Directory, help desk ticketing, hardware troubleshooting, "
        "SQL basics.\nExperience: 3+ years in IT support or system administration.\n"
        "Education: Bachelor's degree in Information Technology or Computer Science.\n"
        "Certifications appreciated: CompTIA A+, Network+, CCNA, ITIL.\n"
        "Soft skills: communication, customer service, problem solving, teamwork.\nLanguages: English."),
    "ACCOUNTANT": (
        "Job title: Staff Accountant\n\nRequired skills: general ledger, accounts payable/receivable, "
        "reconciliations, month-end close, financial statements, Excel, QuickBooks or SAP.\n"
        "Experience: 3+ years of accounting experience.\nEducation: Bachelor's degree in Accounting or Finance.\n"
        "Certifications appreciated: CPA.\nSoft skills: attention to detail, communication, teamwork.\n"
        "Languages: English."),
}


def step_real(out: Path, client: LLMClient, per_job: int, seed: int) -> None:
    path = PROCESSED_DIR / "kaggle_resumes.csv"
    if not path.exists():
        path = PROCESSED_DIR / "resume_atlas.csv"
    if not path.exists():
        log.warning("no real dataset in data/processed; run scripts.download_data first")
        return
    df = pd.read_csv(path).dropna(subset=["text"])
    df["category_norm"] = df["category"].str.upper().str.replace(" ", "-")
    w = terms_to_fuzzy(EXPERT_PRESET, CRITERIA)
    rows = []
    for cat, jtext in REAL_JOBS.items():
        pos = df[df.category_norm == cat]
        if pos.empty:
            log.warning("category %s not in %s", cat, path.name)
            continue
        neg = df[df.category_norm != cat]
        n_pos = per_job // 3
        sample = pd.concat([pos.sample(n_pos, random_state=seed), neg.sample(per_job - n_pos, random_state=seed)])
        texts, rel = [], sample.category_norm.eq(cat).to_numpy()
        scored = []
        for _, r in sample.iterrows():
            text, _ = anonymize(clean_text(str(r.text)))
            text = text[:6000]
            texts.append(text)
            scored.append(score_candidate(str(r.id), text, jtext, CRITERIA, client, n_runs=1))
        x = decision_matrix(scored)
        methods = all_methods(x, w, TYPES, texts, jtext)
        for name, s in methods.items():
            rows.append({"dataset": path.stem, "job": cat, "n": len(sample), "n_relevant": int(rel.sum()),
                         "method": name, "precision@5": precision_at_k(s, rel, 5),
                         f"precision@{n_pos}": precision_at_k(s, rel, n_pos)})
    if rows:
        save(pd.DataFrame(rows), out, "real_weak_labels", index=False)
        print(pd.DataFrame(rows).round(3).to_string(index=False))


# ==========================================================================

def keep_awake() -> None:
    """Prevent Windows from sleeping while this process runs (released at exit)."""
    try:
        import ctypes

        ES_CONTINUOUS, ES_SYSTEM_REQUIRED = 0x80000000, 0x00000001
        ctypes.windll.kernel32.SetThreadExecutionState(ES_CONTINUOUS | ES_SYSTEM_REQUIRED)
    except (AttributeError, OSError):  # not on Windows
        pass


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", default="all")
    ap.add_argument("--provider", default=None)
    ap.add_argument("--model", default=None)
    ap.add_argument("--runs", type=int, default=3)
    ap.add_argument("--tag", default=None, help="results sub-folder (default: model name)")
    ap.add_argument("--no-explain", action="store_true")
    ap.add_argument("--fairness-jobs", default="J1,J2")
    ap.add_argument("--real-n", type=int, default=30)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--limit", type=int, default=None, help="use only the first N synthetic CVs")
    args = ap.parse_args()
    global LIMIT
    LIMIT = args.limit
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    set_global_seed(args.seed)
    keep_awake()

    steps = ["crosscheck", "synthetic", "fairness", "real"] if args.steps == "all" else args.steps.split(",")
    client = LLMClient(provider=args.provider, model=args.model)
    tag = args.tag or client.model.replace(":", "_").replace("/", "_")
    out = RESULTS_DIR / tag
    out.mkdir(parents=True, exist_ok=True)
    (RESULTS_DIR / "crosscheck").mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    if "crosscheck" in steps:
        step_crosscheck(RESULTS_DIR / "crosscheck")
    if "synthetic" in steps:
        step_synthetic(out, client, args.runs, not args.no_explain)
    if "fairness" in steps:
        step_fairness(out, client, args.fairness_jobs.split(","))
    if "real" in steps:
        step_real(out, client, args.real_n, args.seed)
    meta = {"provider": client.provider, "model": client.model, "runs": args.runs, "steps": steps,
            "llm_calls": client.n_calls, "cache_hits": client.n_cache_hits, "seconds": round(time.time() - t0)}
    (out / f"run_meta_{'_'.join(steps)}.json").write_text(json.dumps(meta, indent=1), encoding="utf-8")
    print(meta)


if __name__ == "__main__":
    main()
