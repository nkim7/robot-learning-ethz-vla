#!/usr/bin/env python3
"""
Relabel eval2_180_chengming_promptaug_clean_256_v3 with position-based tasks
(left / middle / right from robot perspective) derived from layout metadata.

Videos and states are copied unchanged. Only task labels change.

Usage:
  python build_eval2_position_dataset.py [--push]
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
from huggingface_hub import HfApi, snapshot_download

SRC_REPO  = "robot-learning-group47/eval2_180_chengming_promptaug_clean_256_v3"
OUT_DIR   = Path("/tmp/eval2_position")
OUT_REPO  = "robot-learning-group47/eval2_180_position_v1"

POSITION_TASKS = {
    "left":   "Put the banana into the left bowl from the robot perspective.",
    "middle": "Put the banana into the middle bowl from the robot perspective.",
    "right":  "Put the banana into the right bowl from the robot perspective.",
}
POS_IDX = {"left": 0, "middle": 1, "right": 2}


def derive_position(layout_left_to_right: str, target_color: str) -> str:
    parts = [p.strip() for p in layout_left_to_right.split("-")]
    return ["left", "middle", "right"][parts.index(target_color)]


def main(push: bool = False) -> None:
    if OUT_DIR.exists():
        shutil.rmtree(OUT_DIR)

    print(f"Downloading {SRC_REPO} ...")
    src = Path(snapshot_download(SRC_REPO, repo_type="dataset"))
    print(f"Source: {src}")

    # --- Copy everything verbatim first ---
    shutil.copytree(src, OUT_DIR)

    # --- Build task_index → position map from source tasks ---
    src_tasks = pd.read_parquet(src / "meta" / "tasks.parquet")
    task_to_pos: dict[int, str] = {}
    for _, row in src_tasks.iterrows():
        pos = derive_position(row["layout_left_to_right"], row["target_color"])
        task_to_pos[int(row["task_index"])] = pos

    # --- Overwrite tasks.parquet with 3 position tasks ---
    new_tasks = pd.DataFrame([
        {"task_index": POS_IDX[pos], "task": task}
        for pos, task in POSITION_TASKS.items()
    ])
    new_tasks.to_parquet(OUT_DIR / "meta" / "tasks.parquet", index=False)
    with open(OUT_DIR / "meta" / "tasks.jsonl", "w") as f:
        for _, row in new_tasks.iterrows():
            f.write(json.dumps({"task_index": int(row["task_index"]), "task": row["task"]}) + "\n")

    # --- Remap task_index in every data parquet ---
    # Build old_task_idx → new_task_idx
    old_to_new = {old: POS_IDX[pos] for old, pos in task_to_pos.items()}

    data_dir = OUT_DIR / "data"
    parquets = sorted(data_dir.rglob("*.parquet"))
    print(f"Remapping task_index in {len(parquets)} parquet files ...")
    for p in parquets:
        df = pd.read_parquet(p)
        if "task_index" in df.columns:
            df["task_index"] = df["task_index"].map(old_to_new)
            df.to_parquet(p, index=False)

    # --- Remap task_index in episodes parquet ---
    eps_dir = OUT_DIR / "meta" / "episodes"
    for p in sorted(eps_dir.rglob("*.parquet")):
        df = pd.read_parquet(p)
        if "task_index" in df.columns:
            df["task_index"] = df["task_index"].map(old_to_new)
            df.to_parquet(p, index=False)

    # --- Update info.json ---
    info = json.loads((OUT_DIR / "meta" / "info.json").read_text())
    info["total_tasks"] = len(POSITION_TASKS)
    (OUT_DIR / "meta" / "info.json").write_text(json.dumps(info, indent=2))

    print(f"\nDataset written to {OUT_DIR}")
    print(f"  tasks: {list(POSITION_TASKS.values())}")

    if push:
        print(f"\nPushing to {OUT_REPO} ...")
        api = HfApi()
        api.create_repo(OUT_REPO, repo_type="dataset", exist_ok=True)
        api.upload_folder(folder_path=str(OUT_DIR), repo_id=OUT_REPO, repo_type="dataset")
        print("Done.")


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--push", action="store_true")
    main(push=p.parse_args().push)
