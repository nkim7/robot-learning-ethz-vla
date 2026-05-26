#!/usr/bin/env python3
"""
Build eval2_180_survey_position dataset.

For every episode in eval2_180_chengming_promptaug_clean_256_v3 we prepend a
synthetic "survey" sequence:
  initial_pose → lift_pose (arm rises, camera sees all 3 bowls)
  hold at lift_pose
  lift_pose → initial_pose (arm returns)

Then relabel each episode with a position-based task:
  "Put the banana into the left/middle/right bowl from the robot perspective."

The position is derived from layout_left_to_right + target_color in tasks.parquet.
Camera frames during the synthetic prefix are the first frame of the original
episode (camera is fixed; scene is already set up).

Output: local LeRobot v2 dataset ready to push to HF.

Usage:
  python build_eval2_survey_dataset.py \
      [--src robot-learning-group47/eval2_180_chengming_promptaug_clean_256_v3] \
      [--out /tmp/eval2_survey_position] \
      [--repo_id robot-learning-group47/eval2_180_survey_position_v1]
      [--push]
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
from huggingface_hub import snapshot_download, HfApi

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

INITIAL_POSE = np.array([-2.24, -0.48, -2.86, 75.34, -122.59, 0.38], dtype=np.float32)
LIFT_POSE    = np.array([ 7.43, 56.57, -82.95, 64.18, -121.27, 0.00], dtype=np.float32)

FPS = 15

# Synthetic prefix design (frames)
SURVEY_UP_FRAMES   = 30   # initial → lift
SURVEY_HOLD_FRAMES = 15   # hold at lift (model sees all bowls here)
SURVEY_DOWN_FRAMES = 30   # lift → initial
SURVEY_TOTAL       = SURVEY_UP_FRAMES + SURVEY_HOLD_FRAMES + SURVEY_DOWN_FRAMES  # 75

# New tasks
POSITION_TASKS = {
    "left":   "Put the banana into the left bowl from the robot perspective.",
    "middle": "Put the banana into the middle bowl from the robot perspective.",
    "right":  "Put the banana into the right bowl from the robot perspective.",
}
POSITION_TO_IDX = {"left": 0, "middle": 1, "right": 2}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def derive_position(layout_left_to_right: str, target_color: str) -> str:
    """'blue-green-red', 'blue' -> 'left'"""
    parts = [p.strip() for p in layout_left_to_right.split("-")]
    idx = parts.index(target_color)
    return ["left", "middle", "right"][idx]


def build_survey_states() -> np.ndarray:
    """
    Returns (SURVEY_TOTAL, 6) array of joint states for the synthetic prefix.
    Action at frame i = state at frame i+1 (last action repeats last state).
    """
    states = []
    # up
    for i in range(SURVEY_UP_FRAMES):
        alpha = (i + 1) / SURVEY_UP_FRAMES
        states.append(INITIAL_POSE * (1 - alpha) + LIFT_POSE * alpha)
    # hold
    for _ in range(SURVEY_HOLD_FRAMES):
        states.append(LIFT_POSE.copy())
    # down
    for i in range(SURVEY_DOWN_FRAMES):
        alpha = (i + 1) / SURVEY_DOWN_FRAMES
        states.append(LIFT_POSE * (1 - alpha) + INITIAL_POSE * alpha)
    return np.stack(states)  # (75, 6)


def survey_actions(survey_states: np.ndarray) -> np.ndarray:
    """Action at step t = state at step t+1; last step repeats."""
    actions = np.concatenate([survey_states[1:], survey_states[-1:]], axis=0)
    return actions


def run(cmd: list[str], **kwargs) -> None:
    result = subprocess.run(cmd, check=True, capture_output=True, **kwargs)
    return result


def extract_frame(video_path: Path, frame_idx: int, out_png: Path) -> None:
    run(["ffmpeg", "-y", "-i", str(video_path),
         "-vf", f"select=eq(n\\,{frame_idx})", "-vframes", "1",
         "-q:v", "2", str(out_png)])


def still_to_video(frame_png: Path, n_frames: int, fps: int, out_mp4: Path) -> None:
    """Repeat a single frame n_frames times into a video."""
    run([
        "ffmpeg", "-y",
        "-loop", "1", "-i", str(frame_png),
        "-t", str(n_frames / fps),
        "-r", str(fps),
        "-c:v", "libx264", "-pix_fmt", "yuv420p",
        "-preset", "fast", "-crf", "18",
        str(out_mp4),
    ])


def extract_video_segment(src: Path, start_frame: int, n_frames: int,
                           fps: int, out_mp4: Path) -> None:
    """Extract n_frames starting at start_frame from src video."""
    start_sec = start_frame / fps
    duration  = n_frames / fps
    run([
        "ffmpeg", "-y",
        "-ss", str(start_sec),
        "-i", str(src),
        "-t", str(duration),
        "-r", str(fps),
        "-c:v", "libx264", "-pix_fmt", "yuv420p",
        "-preset", "fast", "-crf", "18",
        str(out_mp4),
    ])


def concat_videos(*parts: Path, out_mp4: Path) -> None:
    """Concatenate any number of mp4s using ffmpeg concat demuxer."""
    with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as f:
        for p in parts:
            f.write(f"file '{p.resolve()}'\n")
        list_path = f.name
    run([
        "ffmpeg", "-y",
        "-f", "concat", "-safe", "0",
        "-i", list_path,
        "-c", "copy",
        str(out_mp4),
    ])
    Path(list_path).unlink()


def find_lift_peak_frame(states: np.ndarray) -> int:
    """Return the frame index where shoulder_lift (dim 1) is highest."""
    return int(np.argmax(states[:, 1]))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", default="robot-learning-group47/eval2_180_chengming_promptaug_clean_256_v3")
    parser.add_argument("--out", default="/tmp/eval2_survey_position")
    parser.add_argument("--repo_id", default="robot-learning-group47/eval2_180_survey_position_v1")
    parser.add_argument("--push", action="store_true")
    args = parser.parse_args()

    out_dir = Path(args.out)
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True)

    # --- Download source dataset ---
    print(f"Downloading {args.src} ...")
    src_dir = Path(snapshot_download(args.src, repo_type="dataset"))
    print(f"Source at: {src_dir}")

    # --- Load source metadata ---
    src_tasks = pd.read_parquet(src_dir / "meta" / "tasks.parquet")
    src_info = json.loads((src_dir / "meta" / "info.json").read_text())

    # Build task_index → position lookup
    task_to_position: dict[int, str] = {}
    for _, row in src_tasks.iterrows():
        pos = derive_position(row["layout_left_to_right"], row["target_color"])
        task_to_position[int(row["task_index"])] = pos

    # Load episodes parquet
    eps_parquet = next((src_dir / "meta" / "episodes").rglob("*.parquet"))
    src_episodes = pd.read_parquet(eps_parquet).sort_values("episode_index").reset_index(drop=True)

    survey_states_arr = build_survey_states()          # (75, 6)
    survey_actions_arr = survey_actions(survey_states_arr)  # (75, 6)

    # --- Prepare output directories ---
    (out_dir / "data" / "chunk-000").mkdir(parents=True)
    (out_dir / "videos" / "observation.images.front" / "chunk-000").mkdir(parents=True)
    (out_dir / "meta" / "episodes" / "chunk-000").mkdir(parents=True)

    # --- Write new tasks ---
    new_tasks_df = pd.DataFrame([
        {"task_index": POSITION_TO_IDX[pos], "task": task}
        for pos, task in POSITION_TASKS.items()
    ])
    new_tasks_df.to_parquet(out_dir / "meta" / "tasks.parquet", index=False)

    # Also write tasks.jsonl
    with open(out_dir / "meta" / "tasks.jsonl", "w") as f:
        for _, row in new_tasks_df.iterrows():
            f.write(json.dumps({"task_index": int(row["task_index"]), "task": row["task"]}) + "\n")

    # --- Process each episode ---
    all_rows = []
    new_episodes = []
    global_frame_idx = 0
    total_episodes = len(src_episodes)

    for _, ep_row in src_episodes.iterrows():
        ep_idx = int(ep_row["episode_index"])
        orig_task_idx = int(ep_row["task_index"])
        position = task_to_position[orig_task_idx]
        new_task_idx = POSITION_TO_IDX[position]

        print(f"  ep {ep_idx:3d}/{total_episodes}  {position:6s}  task_idx={new_task_idx}")

        # -- Load original data parquet --
        orig_parquet = src_dir / "data" / "chunk-000" / f"file-{ep_idx:06d}.parquet"
        if not orig_parquet.exists():
            # try 3-digit
            orig_parquet = src_dir / "data" / "chunk-000" / f"file-{ep_idx:03d}.parquet"
        orig_df = pd.read_parquet(orig_parquet)

        # columns: index, frame_index, episode_index, timestamp,
        #          action (list/array), observation.state (list/array), task_index
        # Normalise action/state to numpy
        def col_to_arr(col):
            v = orig_df[col].iloc[0]
            if isinstance(v, (list, np.ndarray)):
                return np.stack(orig_df[col].values)
            return orig_df[col].values.reshape(-1, 1)

        orig_states  = col_to_arr("observation.state")   # (N, 6)
        orig_actions = col_to_arr("action")               # (N, 6)
        orig_n = len(orig_df)

        # -- Build prefix rows --
        prefix_n = SURVEY_TOTAL
        ep_global_start = global_frame_idx

        for pf_i in range(prefix_n):
            ts = pf_i / FPS
            all_rows.append({
                "index": global_frame_idx,
                "frame_index": pf_i,
                "episode_index": ep_idx,
                "timestamp": ts,
                "observation.state": survey_states_arr[pf_i].tolist(),
                "action": survey_actions_arr[pf_i].tolist(),
                "task_index": new_task_idx,
            })
            global_frame_idx += 1

        # -- Append original rows (re-indexed) --
        for of_i in range(orig_n):
            ts = (prefix_n + of_i) / FPS
            all_rows.append({
                "index": global_frame_idx,
                "frame_index": prefix_n + of_i,
                "episode_index": ep_idx,
                "timestamp": ts,
                "observation.state": orig_states[of_i].tolist(),
                "action": orig_actions[of_i].tolist(),
                "task_index": new_task_idx,
            })
            global_frame_idx += 1

        ep_length = prefix_n + orig_n

        # -- Build video --
        # Find source video
        vid_chunk = int(ep_row["videos/observation.images.front/chunk_index"])
        vid_file  = int(ep_row["videos/observation.images.front/file_index"])
        src_vid = src_dir / "videos" / "observation.images.front" / f"chunk-{vid_chunk:03d}" / f"file-{vid_file:03d}.mp4"
        if not src_vid.exists():
            src_vid = src_dir / "videos" / "observation.images.front" / f"chunk-{vid_chunk:03d}" / f"file-{vid_file}.mp4"

        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            out_vid = out_dir / "videos" / "observation.images.front" / "chunk-000" / f"file-{ep_idx:03d}.mp4"

            # Frame 0: scene before anything moves (banana on table, bowls in position)
            frame0_png = td / "frame0.png"
            extract_frame(src_vid, 0, frame0_png)

            # Lift-peak frames: real camera footage from when arm is at lift pose
            # These frames show all 3 bowls from the elevated perspective — the
            # decision-critical view. Banana will be in gripper here, but bowl
            # layout is clearly visible and this matches inference conditions.
            peak_frame = find_lift_peak_frame(orig_states)
            hold_start = max(0, peak_frame - SURVEY_HOLD_FRAMES // 2)

            up_mp4   = td / "survey_up.mp4"
            hold_mp4 = td / "survey_hold.mp4"
            down_mp4 = td / "survey_down.mp4"

            still_to_video(frame0_png, SURVEY_UP_FRAMES,   FPS, up_mp4)
            extract_video_segment(src_vid, hold_start, SURVEY_HOLD_FRAMES, FPS, hold_mp4)
            still_to_video(frame0_png, SURVEY_DOWN_FRAMES, FPS, down_mp4)

            concat_videos(up_mp4, hold_mp4, down_mp4, src_vid, out_mp4=out_vid)

        # -- Save episode parquet --
        ep_df = pd.DataFrame(all_rows[ep_global_start:ep_global_start + ep_length])
        ep_df.to_parquet(
            out_dir / "data" / "chunk-000" / f"file-{ep_idx:06d}.parquet",
            index=False,
        )

        new_episodes.append({
            "episode_index": ep_idx,
            "task_index": new_task_idx,
            "length": ep_length,
            "num_frames": ep_length,
            "dataset_from_index": ep_global_start,
            "dataset_to_index": ep_global_start + ep_length,
            "videos/observation.images.front/chunk_index": 0,
            "videos/observation.images.front/file_index": ep_idx,
            "videos/observation.images.front/from_timestamp": 0.0,
            "videos/observation.images.front/to_timestamp": (ep_length - 1) / FPS,
        })

    # --- Write episodes parquet ---
    eps_df = pd.DataFrame(new_episodes)
    eps_df.to_parquet(out_dir / "meta" / "episodes" / "chunk-000" / "file-000.parquet", index=False)

    with open(out_dir / "meta" / "episodes.jsonl", "w") as f:
        for _, row in eps_df.iterrows():
            f.write(json.dumps(row.to_dict()) + "\n")

    # --- Copy stats.json from source (used for normalization) ---
    src_stats = src_dir / "meta" / "stats.json"
    if src_stats.exists():
        shutil.copy(src_stats, out_dir / "meta" / "stats.json")

    # --- Write info.json ---
    total_frames = sum(e["length"] for e in new_episodes)
    new_info = dict(src_info)
    new_info["total_episodes"] = total_episodes
    new_info["total_frames"] = total_frames
    new_info["fps"] = FPS
    # keep features as-is (same state/action/image shape)
    (out_dir / "meta" / "info.json").write_text(json.dumps(new_info, indent=2))

    print(f"\nDataset written to {out_dir}")
    print(f"  episodes: {total_episodes},  frames: {total_frames}")
    print(f"  prefix per episode: {SURVEY_TOTAL} frames ({SURVEY_TOTAL/FPS:.1f}s)")

    # --- Push to HF ---
    if args.push:
        print(f"\nPushing to {args.repo_id} ...")
        api = HfApi()
        api.create_repo(args.repo_id, repo_type="dataset", exist_ok=True)
        api.upload_folder(
            folder_path=str(out_dir),
            repo_id=args.repo_id,
            repo_type="dataset",
        )
        print("Done.")


if __name__ == "__main__":
    main()
