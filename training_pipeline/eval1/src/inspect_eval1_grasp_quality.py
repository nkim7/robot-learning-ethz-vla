#!/usr/bin/env python3
"""
Rank eval1_final_90_chengming episodes by grasp quality:
  - centering:   shoulder_pan.pos during ground dwell (closer to 0 = more centered)
  - dwell_time:  frames arm spent stable at ground before closing gripper (more = better)

All episodes touch the ground; dwell time is the key depth-quality proxy.
"""
import glob
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

import os
from pathlib import Path
SNAP = os.environ.get(
    "SNAP_DIR",
    str(Path.home() / ".cache/huggingface/hub"
        / "datasets--robot-learning-group47--eval1_final_90_chengming"
        / "snapshots/76a015f94fa1a3b6e8280d1aafec664db281ae5f")
)

PAN_IDX, LIFT_IDX, GRIPPER_IDX = 0, 1, 5

GRIPPER_OPEN_THRESH = 20.0    # gripper > this = open
GRIPPER_CLOSED_THRESH = 10.0  # gripper < this = closed after being open
LIFT_STABLE_TOL = 0.5         # degrees — lift within this of max = "at ground"


def analyze_episode(parquet_path: str) -> dict | None:
    df = pq.read_table(parquet_path).to_pandas()
    states = np.vstack(df["observation.state"].values)
    episode_idx = int(df["episode_index"].iloc[0])
    task_idx = int(df["task_index"].iloc[0])

    gripper = states[:, GRIPPER_IDX]
    lift = states[:, PAN_IDX * 0 + LIFT_IDX]  # lift_idx=1
    pan = states[:, PAN_IDX]

    # Find gripper-open start
    open_frames = np.where(gripper > GRIPPER_OPEN_THRESH)[0]
    if not len(open_frames):
        return None
    start_open = int(open_frames[0])

    # Find first close after open
    close_after = np.where(gripper[start_open:] < GRIPPER_CLOSED_THRESH)[0]
    if not len(close_after):
        return None
    grasp_frame = int(start_open + close_after[0])

    # Ground dwell: frames between gripper-open and grasp where lift is at/near its max
    approach_lift = lift[start_open:grasp_frame]
    max_lift = float(approach_lift.max()) if len(approach_lift) else float(lift[grasp_frame])
    stable_mask = approach_lift >= (max_lift - LIFT_STABLE_TOL)
    dwell_frames = int(stable_mask.sum())
    dwell_seconds = dwell_frames / 15.0  # dataset is 15 fps

    # Pan during the stable/ground phase (mean of those frames)
    stable_indices = np.where(stable_mask)[0]
    if len(stable_indices):
        pan_stable = float(pan[start_open + stable_indices[0]: start_open + stable_indices[-1] + 1].mean())
    else:
        pan_stable = float(pan[grasp_frame])

    return {
        "episode_index": episode_idx,
        "task_index": task_idx,
        "n_frames": len(states),
        "grasp_frame": grasp_frame,
        "lift_at_ground": max_lift,
        "dwell_frames": dwell_frames,
        "dwell_seconds": round(dwell_seconds, 3),
        "pan_at_ground": round(pan_stable, 3),
        "abs_pan": round(abs(pan_stable), 3),
    }


TASK_NAMES = {0: "red bowl", 1: "blue bowl", 2: "green bowl"}


def main():
    files = sorted(glob.glob(f"{SNAP}/data/chunk-000/file-*.parquet"))
    records = []
    for f in files:
        r = analyze_episode(f)
        if r is not None:
            records.append(r)
        else:
            df = pq.read_table(f).to_pandas()
            print(f"  WARNING: no grasp found in ep {df['episode_index'].iloc[0]}")

    df = pd.DataFrame(records)
    df["task_name"] = df["task_index"].map(TASK_NAMES)

    print(f"\n{'='*70}")
    print(f"  {len(df)} episodes analyzed  |  fps=15")
    print(f"{'='*70}")
    print("\nSummary:")
    print(df[["dwell_frames", "dwell_seconds", "pan_at_ground", "abs_pan"]].describe().round(3).to_string())

    # ── Rank by DWELL TIME (more = better) ────────────────────────────────
    df_dwell = df.sort_values("dwell_frames", ascending=False).reset_index(drop=True)
    df_dwell["dwell_rank"] = df_dwell.index + 1

    print(f"\n{'─'*72}")
    print("RANKING by DWELL TIME AT GROUND (more frames = better)")
    print(f"{'─'*72}")
    print(f"{'Rank':>4}  {'Ep':>4}  {'Task':<12}  {'dwell_frames':>12}  {'dwell_s':>8}  {'pan':>8}  {'lift@gnd':>9}")
    for _, row in df_dwell.iterrows():
        print(f"  {row.dwell_rank:>3}  {row.episode_index:>4}  {row.task_name:<12}  "
              f"{row.dwell_frames:>12}  {row.dwell_seconds:>8.2f}  {row.pan_at_ground:>8.3f}  {row.lift_at_ground:>9.2f}")

    # ── Rank by CENTERING (abs_pan smaller = better) ───────────────────────
    df_center = df.sort_values("abs_pan", ascending=True).reset_index(drop=True)
    df_center["center_rank"] = df_center.index + 1

    print(f"\n{'─'*72}")
    print("RANKING by CENTERING (|shoulder_pan| smaller = more centered)")
    print(f"{'─'*72}")
    print(f"{'Rank':>4}  {'Ep':>4}  {'Task':<12}  {'pan_at_ground':>13}  {'abs_pan':>8}  {'dwell_s':>8}")
    for _, row in df_center.iterrows():
        print(f"  {row.center_rank:>3}  {row.episode_index:>4}  {row.task_name:<12}  "
              f"{row.pan_at_ground:>13.3f}  {row.abs_pan:>8.3f}  {row.dwell_seconds:>8.2f}")

    # ── Combined rank ──────────────────────────────────────────────────────
    merged = df_dwell[["episode_index", "task_name", "dwell_rank", "dwell_frames", "dwell_seconds"]].merge(
        df_center[["episode_index", "center_rank", "pan_at_ground", "abs_pan", "lift_at_ground"]],
        on="episode_index"
    )
    merged["combined_score"] = (merged["dwell_rank"] + merged["center_rank"]) / 2
    merged = merged.sort_values("combined_score").reset_index(drop=True)
    merged["combined_rank"] = merged.index + 1

    print(f"\n{'─'*72}")
    print("COMBINED RANK (avg dwell_rank + center_rank, lower = better)")
    print(f"{'─'*72}")
    print(f"{'Rank':>4}  {'Ep':>4}  {'Task':<12}  {'score':>6}  {'dwell_r':>8}  {'center_r':>9}  "
          f"{'dwell_s':>8}  {'pan':>8}")
    for _, row in merged.iterrows():
        print(f"  {row.combined_rank:>3}  {row.episode_index:>4}  {row.task_name:<12}  "
              f"{row.combined_score:>6.1f}  {row.dwell_rank:>8.0f}  {row.center_rank:>9.0f}  "
              f"{row.dwell_seconds:>8.2f}  {row.pan_at_ground:>8.3f}")

    out_csv = "eval1_grasp_quality_ranking.csv"
    merged.to_csv(out_csv, index=False)
    print(f"\nSaved to {out_csv}")


if __name__ == "__main__":
    main()
