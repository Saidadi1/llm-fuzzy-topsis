"""Streamlit dashboard: LLM + Fuzzy TOPSIS personnel selection.

    streamlit run app.py
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.config import SYNTHETIC_DIR, get_settings
from src.evaluation import monte_carlo_weights
from src.extraction import make_candidate
from src.fuzzy import WEIGHT_SCALE_5, centroid
from src.llm import LLMClient, LLMError
from src.pipeline import run_pipeline
from src.scoring import load_criteria
from src.topsis import fuzzy_topsis
from src.weights import EXPERT_PRESET, suggest_weights, terms_to_fuzzy

st.set_page_config(page_title="LLM + Fuzzy TOPSIS", layout="wide")
CRITERIA = load_criteria()
TERMS = WEIGHT_SCALE_5.labels
ss = st.session_state
ss.setdefault("weights", dict(EXPERT_PRESET))
ss.setdefault("result", None)

# ---------------------------------------------------------------- sidebar
settings = get_settings()
with st.sidebar:
    st.header("LLM")
    provider = st.selectbox("Provider", ["ollama", "openai", "gemini"],
                            index=["ollama", "openai", "gemini"].index(settings.provider)
                            if settings.provider in ("ollama", "openai", "gemini") else 0)
    model = st.text_input("Model (empty = default)", value=settings.model)
    st.header("Method")
    n_runs = st.radio("Scoring runs (prompt stability)", [1, 3], index=0, horizontal=True,
                      help="3 runs = 3 'decision makers'; the disagreement becomes fuzziness (min, mean, max).")
    policy = st.selectbox("Ungrounded evidence policy", ["downgrade", "very_low", "flag"],
                          help="What to do when the evidence quote is not found in the CV.")
    ideal = st.selectbox("Ideal solutions", ["chen", "minmax"],
                         help="chen: FPIS=(1,1,1), FNIS=(0,0,0); minmax: data-driven")
    do_explain = st.checkbox("Generate explanations", value=True)


@st.cache_resource
def get_client(provider: str, model: str) -> LLMClient:
    return LLMClient(provider=provider, model=model or None)


client = get_client(provider, model)
st.title("Personnel selection: LLM profile analysis + Fuzzy TOPSIS")
st.caption(f"LLM: {client.provider} / {client.model} - all calls cached in {settings.cache_path}")

tab_in, tab_rank, tab_prof, tab_exp, tab_sens = st.tabs(
    ["1 · Job, weights & CVs", "2 · Ranking", "3 · Profiles & evidence", "4 · Explanations", "5 · Sensitivity"])

# ---------------------------------------------------------------- inputs
with tab_in:
    left, right = st.columns([3, 2])
    with left:
        default_job = ""
        if (SYNTHETIC_DIR / "job_J1.txt").exists():
            default_job = (SYNTHETIC_DIR / "job_J1.txt").read_text(encoding="utf-8")
        job_text = st.text_area("Job offer", value=default_job, height=260)
        files = st.file_uploader("Upload CVs (PDF or TXT)", type=["pdf", "txt"], accept_multiple_files=True)
        use_demo = st.checkbox("Use the 20 synthetic demo CVs", value=not files,
                               disabled=not (SYNTHETIC_DIR / "cvs").exists())
        anonymise = st.checkbox("Anonymise CVs before the LLM (recommended)", value=True)
    with right:
        st.subheader("Criteria weights (fuzzy)")
        if st.button("Suggest weights with the LLM", disabled=not job_text.strip()):
            try:
                with st.spinner("Asking the LLM..."):
                    sw = suggest_weights(job_text, CRITERIA, client)
                ss.weights = {k: v.term for k, v in sw.items()}
                ss.weight_reasons = {k: v.reason for k, v in sw.items()}
            except LLMError as e:
                st.error(str(e))
        for c in CRITERIA:
            ss.weights[c.id] = st.select_slider(
                f"{c.id} {c.name}", options=TERMS, value=ss.weights.get(c.id, "Medium"),
                help=ss.get("weight_reasons", {}).get(c.id, c.description), key=f"w_{c.id}")
        w_fuzzy = terms_to_fuzzy(ss.weights, CRITERIA)
        st.dataframe(pd.DataFrame(w_fuzzy, columns=["l", "m", "u"], index=[c.id for c in CRITERIA]),
                     use_container_width=True)

    if st.button("Run analysis", type="primary", disabled=not job_text.strip()):
        cands = []
        if files:
            cands += [make_candidate(f.getvalue(), f.name, do_anonymize=anonymise) for f in files]
        if use_demo:
            cands += [make_candidate(p, cand_id=p.stem, do_anonymize=anonymise)
                      for p in sorted((SYNTHETIC_DIR / "cvs").glob("*.txt"))]
        if not cands:
            st.warning("Add at least one CV.")
        else:
            bar = st.progress(0.0, text="Starting")
            try:
                ss.result = run_pipeline(cands, job_text, CRITERIA, w_fuzzy, client, weight_terms=dict(ss.weights),
                                         n_runs=n_runs, unverified_policy=policy, explain=do_explain,
                                         ideal=ideal, progress=lambda m, f: bar.progress(min(f, 1.0), text=m))
                st.success(f"Done: {len(cands)} candidates. Real LLM calls: {client.n_calls}, "
                           f"cache hits: {client.n_cache_hits}.")
            except LLMError as e:
                st.error(f"LLM error: {e}. Is the provider running / the API key set?")

res = ss.result

# ---------------------------------------------------------------- ranking
with tab_rank:
    if res is None:
        st.info("Run the analysis first.")
    else:
        # weights may have been edited after the run: TOPSIS is recomputed instantly (no LLM)
        w_now = terms_to_fuzzy(ss.weights, CRITERIA)
        live = fuzzy_topsis(res.x, w_now, [c.type for c in CRITERIA], ideal=ideal)
        res.topsis = live
        table = res.table()
        st.subheader("Ranking (closeness coefficient CC = d⁻ / (d* + d⁻))")
        fig = go.Figure(go.Bar(x=table["CC"], y=table["candidate_id"], orientation="h",
                               text=table["CC"].round(3), textposition="outside"))
        fig.update_layout(yaxis=dict(autorange="reversed"), height=30 * len(table) + 120,
                          xaxis_title="CC", margin=dict(l=10, r=10, t=10, b=10))
        st.plotly_chart(fig, use_container_width=True)
        show = ["rank", "candidate_id", "file", "CC", "d_plus", "d_minus", "run_agreement"] + \
               [f"{c.id}_term" for c in CRITERIA]
        st.dataframe(table[show], use_container_width=True, hide_index=True)
        st.download_button("Download CSV", table.to_csv(index=False).encode("utf-8"),
                           "ranking.csv", "text/csv")

# ---------------------------------------------------------------- profiles
with tab_prof:
    if res is None:
        st.info("Run the analysis first.")
    else:
        ids = [c.id for c in res.candidates]
        order = list(np.array(ids)[res.topsis.order()])
        sel = st.multiselect("Candidates to compare (radar)", ids, default=order[:3])
        radar = go.Figure()
        names = [c.name for c in CRITERIA]
        for cid in sel:
            i = ids.index(cid)
            vals = centroid(res.x[i]).tolist()
            radar.add_trace(go.Scatterpolar(r=vals + vals[:1], theta=names + names[:1], fill="toself", name=cid))
        radar.update_layout(polar=dict(radialaxis=dict(range=[0, 1])), height=450)
        st.plotly_chart(radar, use_container_width=True)

        one = st.selectbox("Candidate details", order)
        i = ids.index(one)
        cand, sc = res.candidates[i], res.scored[i]
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**Per-criterion fuzzy rating**")
            tri = go.Figure()
            for j, c in enumerate(CRITERIA):
                l, m, u = sc.fuzzy[j]
                tri.add_trace(go.Scatter(x=[l, m, u], y=[0, 1, 0], mode="lines", name=c.id))
            tri.update_layout(height=300, xaxis=dict(range=[0, 1]), yaxis_title="membership")
            st.plotly_chart(tri, use_container_width=True)
            rows = [{"criterion": c.name, "term": sc.final_terms[c.id], "verified": sc.verified[c.id],
                     "evidence": sc.runs[0][c.id].evidence, "rationale": sc.runs[0][c.id].rationale}
                    for c in CRITERIA]
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
        with c2:
            st.markdown("**Extracted profile (LLM)**")
            st.json(json.loads(cand.profile.model_dump_json()) if cand.profile else {})
            with st.expander("Text seen by the LLM (anonymised)"):
                st.text(cand.text)
            if cand.removed:
                st.caption("Removed before the LLM: " + ", ".join(cand.removed))

# ---------------------------------------------------------------- explanations
with tab_exp:
    if res is None or not res.explanations:
        st.info("Run the analysis with explanations enabled.")
    else:
        st.caption("Generated from the ratings and evidence only (the LLM does not see the CV here); "
                   "rank at generation time.")
        for i in res.topsis.order():
            c = res.candidates[i]
            e = res.explanations.get(c.id)
            if e is None:
                continue
            with st.expander(f"#{res.topsis.rank[i]} {c.id} - CC {res.topsis.cc[i]:.3f}", expanded=i == res.topsis.order()[0]):
                st.write(e.summary)
                a, b = st.columns(2)
                a.markdown("**Strengths**\n" + "\n".join(f"- {s}" for s in e.strengths))
                b.markdown("**Weaknesses**\n" + "\n".join(f"- {s}" for s in e.weaknesses))

# ---------------------------------------------------------------- sensitivity
with tab_sens:
    if res is None:
        st.info("Run the analysis first.")
    else:
        st.markdown("Change one weight and watch the ranking (no LLM call: only TOPSIS is recomputed).")
        crit = st.selectbox("Criterion", [c.id for c in CRITERIA])
        factor = st.slider("Weight multiplier", 0.0, 2.0, 1.0, 0.1)
        j = [c.id for c in CRITERIA].index(crit)
        w0 = terms_to_fuzzy(ss.weights, CRITERIA)
        w1 = w0.copy()
        w1[j] = np.clip(w1[j] * factor, 0, 1)
        types = [c.type for c in CRITERIA]
        r0, r1 = fuzzy_topsis(res.x, w0, types, ideal), fuzzy_topsis(res.x, w1, types, ideal)
        ids = [c.id for c in res.candidates]
        comp = pd.DataFrame({"candidate": ids, "rank_base": r0.rank, "rank_new": r1.rank,
                             "CC_base": r0.cc.round(4), "CC_new": r1.cc.round(4)}).sort_values("rank_new")
        st.dataframe(comp, use_container_width=True, hide_index=True)
        st.markdown("**Monte-Carlo robustness** (all weights perturbed by U(0.7, 1.3), 1000 draws)")
        mc = monte_carlo_weights(res.x, w0, types, n_samples=1000, noise=0.3)
        mc.insert(0, "candidate", ids)
        st.dataframe(mc.sort_values("mean_rank").round(3), use_container_width=True, hide_index=True)
