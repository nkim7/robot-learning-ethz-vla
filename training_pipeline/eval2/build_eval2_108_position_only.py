#!/usr/bin/env python3
"""
Build robot-learning-group47/eval2_108_position_only

Source: robot-learning-group47/eval2_fixed_bowl_fixed_banana_108
Map:    combo_episode_map_eval2_fixed_bowl_fixed_banana_108.csv  (in dataset root)

Steps:
  1. Copy dataset verbatim into /tmp/eval2_108_position_only/
  2. Derive episode → target_position from combo map
  3. Replace tasks.parquet / tasks.jsonl with 3 position-only tasks
  4. Remap task_index in every data parquet and episodes parquet
  5. Update meta/info.json  (total_tasks = 3)
  6. Push to HF Hub as robot-learning-group47/eval2_108_position_only

Usage:
    python build_eval2_108_position_only.py [--dry-run] [--push]
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import pandas as pd
from huggingface_hub import HfApi, snapshot_download

# ── constants ─────────────────────────────────────────────────────────────────
SRC_REPO  = "robot-learning-group47/eval2_fixed_bowl_fixed_banana_108"
OUT_REPO  = "robot-learning-group47/eval2_108_position_only"
OUT_DIR   = Path("/tmp/eval2_108_position_only")
CACHE_DIR = Path("/tmp/stage0_primary")   # reuse Stage 0 download if present

COLOR_MAP = {"b": "blue", "g": "green", "r": "red"}
SLOTS     = ["left", "middle", "right"]

POSITION_TASKS = {
    "left":   "Put the banana into the left bowl from the robot perspective.",
    "middle": "Put the banana into the middle bowl from the robot perspective.",
    "right":  "Put the banana into the right bowl from the robot perspective.",
}
POS_IDX = {"left": 0, "middle": 1, "right": 2}


# ── helpers ───────────────────────────────────────────────────────────────────
def get_source() -> Path:
    if CACHE_DIR.exists():
        print(f"  Reusing cached source at {CACHE_DIR}")
        return CACHE_DIR
    print(f"  Downloading {SRC_REPO} …")
    p = Path(snapshot_download(SRC_REPO, repo_type="dataset"))
    shutil.copytree(p, CACHE_DIR)
    return CACHE_DIR


def build_ep_to_pos(src: Path) -> dict[int, int]:
    """
    Returns {episode_index: new_task_index}  where new_task_index ∈ {0,1,2}
    corresponding to left/middle/right.
    """
    csv_path = src / "combo_episode_map_eval2_fixed_bowl_fixed_banana_108.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"Combo map not found: {csv_path}")

    df = pd.read_csv(csv_path)
    # expected columns: combo, episode_start, episode_end, target_bowl_color, bowl_order_left_to_right
    ep_to_pos: dict[int, int] = {}
    for _, row in df.iterrows():
        layout_code = str(row["bowl_order_left_to_right"]).strip()
        layout      = [COLOR_MAP[c] for c in layout_code]   # e.g. "bgr" → ["blue","green","red"]
        target_color = str(row["target_bowl_color"]).strip().lower()
        target_pos   = SLOTS[layout.index(target_color)]
        new_idx      = POS_IDX[target_pos]
        for ep in range(int(row["episode_start"]), int(row["episode_end"]) + 1):
            ep_to_pos[ep] = new_idx

    print(f"  Episode→position map built: {len(ep_to_pos)} episodes")
    # sanity: show distribution
    from collections import Counter
    dist = Counter(ep_to_pos.values())
    for idx, pos in enumerate(SLOTS):
        print(f"    task_index={idx} ({pos}): {dist[idx]} episodes")
    return ep_to_pos


def remap_parquet_col(path: Path, ep_to_new_idx: dict[int, int],
                      task_col: str = "task_index",
                      ep_col: str | None = "episode_index") -> int:
    """
    In-place remap of `task_col` in a parquet file.
    If ep_col is present, use it to look up per-episode mapping.
    Otherwise remap task_index directly via old→new map.
    Returns number of rows modified.
    """
    df = pd.read_parquet(path)
    if task_col not in df.columns:
        return 0

    before = df[task_col].copy()

    if ep_col and ep_col in df.columns:
        df[task_col] = df[ep_col].map(ep_to_new_idx)
    else:
        # episodes parquet: episode_index IS the key
        # For episodes parquet with no ep_col, use the index column if available
        if "episode_index" in df.columns:
            df[task_col] = df["episode_index"].map(ep_to_new_idx)
        else:
            # fallback: shouldn't happen for well-formed LeRobot datasets
            raise ValueError(f"Cannot remap {path}: no episode_index column")

    changed = int((before != df[task_col]).sum())
    df.to_parquet(path, index=False)
    return changed


# ── main ──────────────────────────────────────────────────────────────────────
def main(dry_run: bool = False, push: bool = False) -> None:
    print("=" * 62)
    print("  Build eval2_108_position_only")
    print("=" * 62)

    # 1. source
    src = get_source()

    # 2. copy to output dir
    if OUT_DIR.exists():
        print(f"\n  Removing existing {OUT_DIR} …")
        shutil.rmtree(OUT_DIR)
    print(f"\n  Copying dataset to {OUT_DIR} …")
    shutil.copytree(src, OUT_DIR)

    # 3. build episode → new task_index map
    print("\n  Building episode → position map …")
    ep_to_new_idx = build_ep_to_pos(src)

    if dry_run:
        print("\n  [DRY RUN] Skipping all writes.")
        return

    # 4. overwrite tasks.parquet
    print("\n  Writing new tasks.parquet …")
    new_tasks = pd.DataFrame([
        {"task_index": POS_IDX[pos], "task": task}
        for pos, task in POSITION_TASKS.items()
    ])
    new_tasks.to_parquet(OUT_DIR / "meta" / "tasks.parquet", index=False)

    # 5. overwrite tasks.jsonl (if present)
    tasks_jsonl = OUT_DIR / "meta" / "tasks.jsonl"
    print(f"  Writing new tasks.jsonl …")
    with open(tasks_jsonl, "w") as f:
        for pos, task in POSITION_TASKS.items():
            f.write(json.dumps({"task_index": POS_IDX[pos], "task": task}) + "\n")

    # 6. remap task_index in data parquets
    data_dir = OUT_DIR / "data"
    data_files = sorted(data_dir.rglob("*.parquet"))
    print(f"\n  Remapping task_index in {len(data_files)} data parquet(s) …")
    total_changed = 0
    for pf in data_files:
        changed = remap_parquet_col(pf, ep_to_new_idx,
                                    task_col="task_index", ep_col="episode_index")
        total_changed += changed
        print(f"    {pf.relative_to(OUT_DIR)}: {changed} rows updated")
    print(f"  Total data rows updated: {total_changed}")

    # 7. remap task_index in episodes parquet
    eps_dir = OUT_DIR / "meta" / "episodes"
    eps_files = sorted(eps_dir.rglob("*.parquet")) if eps_dir.exists() else []
    if not eps_files:
        flat = OUT_DIR / "meta" / "episodes.parquet"
        if flat.exists():
            eps_files = [flat]
    print(f"\n  Remapping task_index in {len(eps_files)} episodes parquet(s) …")
    for pf in eps_files:
        # episodes parquet stores task info in a 'tasks' list column (task strings),
        # and also stats/task_index columns.  We need to update the stats cols.
        df = pd.read_parquet(pf)
        print(f"    {pf.relative_to(OUT_DIR)}: {len(df)} episodes")

        # Update 'tasks' column (list of task strings)
        if "tasks" in df.columns:
            def remap_task_strings(task_list):
                # Replace each color task string with the corresponding position string
                result = []
                for t in task_list:
                    t = str(t)
                    # find which episode this is — use episode_index column
                    # We'll do this after the merge below
                    result.append(t)
                return result
            # Better: rebuild tasks column from episode_index
            ep_to_pos_str = {ep: POSITION_TASKS[SLOTS[idx]] for ep, idx in ep_to_new_idx.items()}
            df["tasks"] = df["episode_index"].map(lambda ep: [ep_to_pos_str[ep]])
            print(f"      'tasks' column rewritten ✓")

        # Update stats/task_index/* columns (min, max, mean, std → all same value since 1 task/episode)
        for stat_col in ["stats/task_index/min", "stats/task_index/max",
                         "stats/task_index/mean", "stats/task_index/q01",
                         "stats/task_index/q10", "stats/task_index/q50",
                         "stats/task_index/q90", "stats/task_index/q99"]:
            if stat_col in df.columns:
                df[stat_col] = df["episode_index"].map(
                    lambda ep: [float(ep_to_new_idx[ep])]
                )
        # std → 0
        if "stats/task_index/std" in df.columns:
            df["stats/task_index/std"] = df["episode_index"].map(lambda ep: [0.0])
        # count unchanged

        df.to_parquet(pf, index=False)
        print(f"      Saved ✓")

    # 8. update info.json
    print(f"\n  Updating meta/info.json …")
    info_path = OUT_DIR / "meta" / "info.json"
    info = json.loads(info_path.read_text())
    info["total_tasks"] = len(POSITION_TASKS)
    info_path.write_text(json.dumps(info, indent=2))
    print(f"    total_tasks → {len(POSITION_TASKS)}")

    # 9. verification pass
    print(f"\n  Verification …")
    tasks_check = pd.read_parquet(OUT_DIR / "meta" / "tasks.parquet")
    print(f"    tasks.parquet:")
    for _, r in tasks_check.iterrows():
        print(f"      [{r['task_index']}] {r['task']}")

    data_check = pd.read_parquet(data_files[0])
    from collections import Counter
    dist = Counter(data_check["task_index"].tolist())
    total_frames = len(data_check)
    print(f"    data parquet task_index distribution ({total_frames} frames in first shard):")
    for idx in sorted(dist):
        print(f"      task_index={idx} ({SLOTS[idx]}): {dist[idx]} frames")

    # verify episodes tasks column sample
    df_eps_check = pd.read_parquet(eps_files[0]) if eps_files else None
    if df_eps_check is not None:
        print(f"    episodes tasks sample (first 6):")
        for _, r in df_eps_check.head(6).iterrows():
            ep = int(r["episode_index"])
            ts = r["tasks"]
            print(f"      ep={ep} tasks={ts}")

    print(f"\n  Output: {OUT_DIR}")

    # 10. push
    if push:
        print(f"\n  Pushing to {OUT_REPO} …")
        api = HfApi()
        api.create_repo(OUT_REPO, repo_type="dataset", exist_ok=True)
        api.upload_folder(
            folder_path=str(OUT_DIR),
            repo_id=OUT_REPO,
            repo_type="dataset",
            commit_message="Add eval2_108_position_only: remap color tasks to left/middle/right",
        )
        print(f"  Pushed ✓  →  https://huggingface.co/datasets/{OUT_REPO}")
    else:
        print(f"\n  (Pass --push to upload to {OUT_REPO})")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dry-run", action="store_true",
                   help="Build the map and print stats; do not write files.")
    p.add_argument("--push", action="store_true",
                   help="Upload to Hugging Face Hub after building.")
    args = p.parse_args()
    main(dry_run=args.dry_run, push=args.push)
