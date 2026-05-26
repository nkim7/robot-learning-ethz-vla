#!/usr/bin/env python3
"""Download and merge eval3_90_permutation + eval3_lmr_210 into
data/eval3_90_plus_lmr_210, matching the dataset used on the shadeform server.

Run once before training:
    python scripts/eval3_cotrain/prep_run2_dataset.py
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT_DIR / "lerobot" / "src"))

# ── Config ────────────────────────────────────────────────────────────────────
REPO_90   = "robot-learning-group47/eval3_90_permutation"
REPO_LMR  = "robot-learning-group47/eval3_lmr_210"
LOCAL_90  = ROOT_DIR / "data" / "eval3_90_permutation"
LOCAL_LMR = ROOT_DIR / "data" / "eval3_lmr_210"
MERGED    = ROOT_DIR / "data" / "eval3_90_plus_lmr_210"
MERGED_REPO_ID = "robot-learning-group47/eval3_90_plus_lmr_210"

os.environ.setdefault("HF_HOME", "/tmp/hf-home")
os.environ.setdefault("HF_DATASETS_CACHE", "/tmp/hf-datasets-cache")


def main() -> None:
    if MERGED.exists():
        print(f"Merged dataset already exists at {MERGED} — skipping.")
        return

    from lerobot.datasets import LeRobotDataset
    from lerobot.datasets.dataset_tools import merge_datasets

    print(f"Loading {REPO_90} …")
    ds_90 = LeRobotDataset(REPO_90, root=LOCAL_90)

    print(f"Loading {REPO_LMR} …")
    ds_lmr = LeRobotDataset(REPO_LMR, root=LOCAL_LMR)

    print(f"Merging → {MERGED} …")
    merge_datasets([ds_90, ds_lmr], output_repo_id=MERGED_REPO_ID, output_dir=MERGED)
    print(f"Done. Merged dataset at: {MERGED}")


if __name__ == "__main__":
    main()
