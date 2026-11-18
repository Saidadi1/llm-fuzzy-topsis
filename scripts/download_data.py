"""Download and prepare the public datasets.

| dataset                                   | licence (checked 2026-10)  | size   |
|-------------------------------------------|----------------------------|--------|
| Kaggle snehaanbhawal/resume-dataset       | CC0: Public Domain         | 118 MB |
| Hugging Face ahmedheakl/resume-atlas      | MIT                        |  54 MB |
| Kaggle arshkon/linkedin-job-postings      | CC BY-SA 4.0               | 556 MB |

Public Kaggle datasets can be downloaded anonymously through the Kaggle API URL;
kagglehub (with credentials) is used as a fallback. LinkedIn postings are optional
(--linkedin) because of their size.

    python -m scripts.download_data [--linkedin]
"""

from __future__ import annotations

import argparse
import shutil
import traceback
import zipfile
from pathlib import Path

import pandas as pd
import requests

from src.config import PROCESSED_DIR, RAW_DIR

KAGGLE_URL = "https://www.kaggle.com/api/v1/datasets/download/{ref}"
HF_ATLAS_URL = "https://huggingface.co/datasets/ahmedheakl/resume-atlas/resolve/main/train.csv"


def _download(url: str, dst: Path) -> Path:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() and dst.stat().st_size > 0:
        return dst
    tmp = dst.with_suffix(dst.suffix + ".part")
    with requests.get(url, stream=True, timeout=120, allow_redirects=True) as r:
        r.raise_for_status()
        with open(tmp, "wb") as f:
            for chunk in r.iter_content(1 << 20):
                f.write(chunk)
    tmp.replace(dst)
    return dst


def kaggle_dataset(ref: str) -> Path:
    """Download + unzip a public Kaggle dataset into data/raw/<name>/ (its own folder)."""
    name = ref.split("/")[1]
    out = RAW_DIR / name
    if out.exists() and any(out.rglob("*.csv")):
        return out
    try:
        z = _download(KAGGLE_URL.format(ref=ref), RAW_DIR / f"{name}.zip")
        with zipfile.ZipFile(z) as zf:
            zf.extractall(out)
    except Exception:  # noqa: BLE001 - fall back on authenticated kagglehub
        import kagglehub

        shutil.copytree(Path(kagglehub.dataset_download(ref)), out, dirs_exist_ok=True)
    return out


def kaggle_resumes() -> None:
    src = kaggle_dataset("snehaanbhawal/resume-dataset")
    df = pd.read_csv(next(src.rglob("Resume.csv")))
    out = df.rename(columns={"Resume_str": "text", "Category": "category", "ID": "id"})[["id", "category", "text"]]
    out["source"] = "kaggle_livecareer"
    out.to_csv(PROCESSED_DIR / "kaggle_resumes.csv", index=False)
    print(f"Kaggle Resume Dataset: {len(out)} CVs, {out.category.nunique()} categories")


def hf_resume_atlas() -> None:
    csv = _download(HF_ATLAS_URL, RAW_DIR / "resume_atlas" / "train.csv")
    df = pd.read_csv(csv)
    text_col = next(c for c in df.columns if c.lower() in ("text", "resume", "resume_str"))
    cat_col = next(c for c in df.columns if c.lower() in ("category", "label", "class"))
    out = pd.DataFrame({"id": [f"atlas_{i}" for i in range(len(df))], "category": df[cat_col],
                        "text": df[text_col], "source": "hf_resume_atlas"})
    out.to_csv(PROCESSED_DIR / "resume_atlas.csv", index=False)
    print(f"resume-atlas: {len(out)} CVs, {out.category.nunique()} categories")


def linkedin_jobs() -> None:
    src = kaggle_dataset("arshkon/linkedin-job-postings")
    df = pd.read_csv(next(src.rglob("postings.csv")),
                     usecols=lambda c: c in {"job_id", "title", "description", "skills_desc",
                                             "formatted_experience_level"})
    df = df.dropna(subset=["description"])
    df.to_csv(PROCESSED_DIR / "linkedin_postings.csv", index=False)
    print(f"LinkedIn postings: {len(df)} offers")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--linkedin", action="store_true", help="also download the 556 MB LinkedIn postings")
    ap.add_argument("--no-atlas", action="store_true")
    args = ap.parse_args()
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    jobs = [("kaggle_resumes", kaggle_resumes)]
    if not args.no_atlas:
        jobs.append(("hf_resume_atlas", hf_resume_atlas))
    if args.linkedin:
        jobs.append(("linkedin_jobs", linkedin_jobs))
    status = {}
    for name, fn in jobs:
        try:
            fn()
            status[name] = "ok"
        except Exception as e:  # noqa: BLE001 - report and continue
            status[name] = f"FAILED: {type(e).__name__}: {str(e)[:200]}"
            traceback.print_exc(limit=1)
    print("\nSummary:")
    for k, v in status.items():
        print(f"  {k:16s} {v}")


if __name__ == "__main__":
    main()
