"""Generate the synthetic benchmark (2 job offers, 20 CVs, ground truth).

    python -m scripts.generate_synthetic            # LLM writes CV prose (cached)
    python -m scripts.generate_synthetic --template # no LLM, deterministic templates
"""

from __future__ import annotations

import argparse
import logging

from src.config import SYNTHETIC_DIR, set_global_seed
from src.llm import LLMClient
from src.scoring import load_criteria
from src.synthetic import sample_candidates, save_benchmark, write_cv_body
from src.weights import EXPERT_PRESET, crisp_weights, terms_to_fuzzy


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--template", action="store_true", help="do not use an LLM for CV prose")
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    set_global_seed(args.seed)

    client = None if args.template else LLMClient()
    cands = sample_candidates(args.seed)
    for c in cands:
        c.body, c.body_source = write_cv_body(c, client)
        logging.info("%s (%s): %s", c.id, c.archetype, c.body_source)

    w = crisp_weights(terms_to_fuzzy(EXPERT_PRESET, load_criteria()))
    save_benchmark(cands, SYNTHETIC_DIR, w)
    n_llm = sum(c.body_source == "llm" for c in cands)
    print(f"Saved {len(cands)} CVs to {SYNTHETIC_DIR} ({n_llm} LLM-written, {len(cands) - n_llm} template)")


if __name__ == "__main__":
    main()
