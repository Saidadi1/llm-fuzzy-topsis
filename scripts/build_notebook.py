"""Build notebooks/results.ipynb (tables + figures from results/).

    python -m scripts.build_notebook
    jupyter nbconvert --to notebook --execute --inplace notebooks/results.ipynb
"""

from pathlib import Path

import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]

cells = []
md = lambda s: cells.append(nbf.v4.new_markdown_cell(s))
code = lambda s: cells.append(nbf.v4.new_code_cell(s))

md("""# LLM + Fuzzy TOPSIS — reproduction of the results

This notebook only reads `results/` (produced by `python -m scripts.run_experiments`) and
re-runs the cheap, LLM-free computations (TOPSIS, sensitivity). Re-running the experiment
script itself is also free once the LLM cache `.cache/llm_cache.sqlite` exists.""")

code("""import sys, json
from pathlib import Path
ROOT = Path.cwd().parent if Path.cwd().name == 'notebooks' else Path.cwd()
sys.path.insert(0, str(ROOT))
import numpy as np, pandas as pd, matplotlib.pyplot as plt
pd.set_option('display.precision', 3); pd.set_option('display.width', 200)
RES = ROOT / 'results'; FIG = RES / 'figures'; FIG.mkdir(parents=True, exist_ok=True)
MODELS = sorted(p.name for p in RES.iterdir() if p.is_dir() and (p / 'synthetic_metrics.csv').exists())
MAIN = 'qwen3.5_4b' if 'qwen3.5_4b' in MODELS else (MODELS[-1] if MODELS else None)
print('result folders:', MODELS, '| main:', MAIN)""")

md("## 1. Validation of the from-scratch Fuzzy TOPSIS on Chen (2000)")
code("""cc = pd.read_csv(RES / 'crosscheck' / 'crosscheck_chen2000.csv'); display(cc)
p = RES / 'crosscheck' / 'crosscheck_skfuzzy.txt'
print(p.read_text() if p.exists() else 'scikit-fuzzy check not run')""")

md("## 2. Synthetic benchmark — agreement with the ground-truth ranking")
code("""m = pd.read_csv(RES / MAIN / 'synthetic_metrics.csv', index_col=0); display(m)
fig, axes = plt.subplots(1, 2, figsize=(13, 4), sharey=True)
for ax, (job, g) in zip(axes, m.groupby('job')):
    g[['spearman', 'kendall_tau', 'top5_overlap']].plot.barh(ax=ax)
    ax.set_title(f'{job} vs ground truth ({MAIN})'); ax.set_xlim(-0.2, 1); ax.axvline(0, c='k', lw=.5)
plt.tight_layout(); plt.savefig(FIG / 'metrics_by_method.png', dpi=150); plt.show()""")

code("""for job in ['J1', 'J2']:
    t = pd.read_csv(RES / MAIN / f'ranking_{job}.csv')
    cols = ['rank', 'candidate_id', 'archetype', 'CC', 'gt_rank', 'gt_score'] + [f'C{i}_term' for i in range(1, 7)]
    print(job); display(t[cols].head(10))
    plt.figure(figsize=(5, 4)); plt.scatter(t.gt_score, t.CC)
    for _, r in t.iterrows(): plt.annotate(r.candidate_id, (r.gt_score, r.CC), fontsize=7)
    plt.xlabel('ground-truth score'); plt.ylabel('fuzzy TOPSIS CC'); plt.title(f'{job}: CC vs ground truth')
    plt.tight_layout(); plt.savefig(FIG / f'cc_vs_gt_{job}.png', dpi=150); plt.show()""")

md("## 3. Extraction quality and per-criterion rating quality")
code("""display(pd.read_csv(RES / MAIN / 'extraction_accuracy_mean.csv', index_col=0))
display(pd.read_csv(RES / MAIN / 'rating_quality.csv').pivot(index='criterion', columns='job'))""")

md("## 4. Ablations (computed from stored LLM answers: no extra call)")
code("""a = pd.read_csv(RES / MAIN / 'ablation.csv'); display(a)
s = RES / MAIN / 'stability.csv'
if s.exists(): display(pd.read_csv(s))""")

md("## 5. Sensitivity analysis on the weights")
code("""for job in ['J1', 'J2']:
    o = pd.read_csv(RES / MAIN / f'sensitivity_oat_{job}.csv')
    fig, ax = plt.subplots(figsize=(6, 3.5))
    for c, g in o.groupby('criterion'): ax.plot(g.factor, g.kendall_tau, marker='o', label=c)
    ax.set_xlabel('weight multiplier'); ax.set_ylabel('Kendall tau vs base'); ax.set_title(f'{job}: one-at-a-time')
    ax.legend(ncol=3, fontsize=8); plt.tight_layout(); plt.savefig(FIG / f'sensitivity_oat_{job}.png', dpi=150); plt.show()
    sc = pd.read_csv(RES / MAIN / f'sensitivity_scenarios_{job}.csv', index_col=0).sort_values('base')
    fig, ax = plt.subplots(figsize=(8, 6)); im = ax.imshow(sc.values, cmap='viridis_r', aspect='auto')
    ax.set_xticks(range(sc.shape[1]), sc.columns, rotation=45, ha='right'); ax.set_yticks(range(len(sc)), sc.index)
    plt.colorbar(im, label='rank'); ax.set_title(f'{job}: rank per weighting scenario'); plt.tight_layout()
    plt.savefig(FIG / f'sensitivity_scenarios_{job}.png', dpi=150); plt.show()
    display(pd.read_csv(RES / MAIN / f'sensitivity_montecarlo_{job}.csv', index_col=0).head(8))""")

md("## 6. Counterfactual fairness (same CV body, swapped name / gender / age / nationality)")
code("""f = RES / MAIN / 'fairness_counterfactual.csv'
display(pd.read_csv(f) if f.exists() else 'not run')""")

md("## 7. Explanations — protected-attribute check")
code("""for job in ['J1', 'J2']:
    p = RES / MAIN / f'explanations_{job}.csv'
    if p.exists():
        e = pd.read_csv(p).fillna('')
        print(job, 'explanations mentioning protected terms:', int((e.protected_terms != '').sum()), '/', len(e))
        display(e.head(3))""")

md("## 8. Real data (weak labels: category of the CV)")
code("""rows = [pd.read_csv(RES / m / 'real_weak_labels.csv').assign(model=m) for m in MODELS if (RES / m / 'real_weak_labels.csv').exists()]
display(pd.concat(rows) if rows else 'not run')""")

md("## 9. Effect of the LLM size")
code("""rows = []
for mdl in MODELS:
    t = pd.read_csv(RES / mdl / 'synthetic_metrics.csv', index_col=0).reset_index()
    rows.append(t[t.method.isin(['fuzzy_topsis', 'tfidf_cosine'])].assign(model=mdl))
if rows: display(pd.concat(rows).pivot_table(index=['model', 'method'], columns='job', values=['spearman', 'kendall_tau']))""")

nb = nbf.v4.new_notebook(cells=cells, metadata={"kernelspec": {"name": "python3", "display_name": "Python 3"}})
out = ROOT / "notebooks" / "results.ipynb"
out.parent.mkdir(exist_ok=True)
nbf.write(nb, out)
print("wrote", out)
