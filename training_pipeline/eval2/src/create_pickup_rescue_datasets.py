#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd


PICKUP_TASK = "Pick up the banana and lift it until all three bowls are visible."


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval1-source", type=Path, default=Path("/data/eval1_chengming_256h264"))
    parser.add_argument("--eval2-pickup", type=Path, default=Path("/data/lerobot_cache/robot-learning-group47/eval2_180_posegated_pickuplift_clean256_v1"))
    parser.add_argument("--eval2-split-csv", type=Path, default=Path("/data/eval2_180_2phase_inspect_posebased/eval2_180_split_candidates_posebased.csv"))
    parser.add_argument("--out-root", type=Path, default=Path("/data/lerobot_cache/robot-learning-group47"))
    parser.add_argument("--name-b", default="pickup_rescue_eval1eval2_clean256_v1")
    parser.add_argument("--name-c", default="pickup_rescue_eval2_besthalf_clean256_v1")
    parser.add_argument("--name-d", default="pickup_rescue_eval1_booster_clean256_v1")
    parser.add_argument("--eval1-lift-threshold", type=float, default=120.0)
    parser.add_argument("--eval1-gripper-max", type=float, default=10.0)
    parser.add_argument("--eval1-max-split", type=int, default=140)
    parser.add_argument("--only", nargs="*", choices=["b", "c", "d"], default=["b", "c", "d"])
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def clean_dir(path: Path, force: bool) -> None:
    if path.exists():
        if not force:
            raise FileExistsError(f"{path} exists; pass --force")
        shutil.rmtree(path)
    path.mkdir(parents=True)


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))


def episode_files(dataset: Path) -> list[Path]:
    return sorted((dataset / "data/chunk-000").glob("*.parquet"))


def episode_file(dataset: Path, episode_index: int) -> Path:
    for fmt in (f"file-{episode_index:06d}.parquet", f"file-{episode_index:03d}.parquet"):
        path = dataset / "data/chunk-000" / fmt
        if path.exists():
            return path
    raise FileNotFoundError((dataset, episode_index))


def video_file(dataset: Path, episode_index: int) -> Path:
    for fmt in (f"file-{episode_index:06d}.mp4", f"file-{episode_index:03d}.mp4"):
        path = dataset / "videos/observation.images.front/chunk-000" / fmt
        if path.exists():
            return path
    raise FileNotFoundError((dataset, episode_index))


def paired_video_file(dataset: Path, data_file: Path) -> Path:
    stem = data_file.stem
    candidates = [
        dataset / "videos/observation.images.front/chunk-000" / f"{stem}.mp4",
        dataset / "videos/observation.images.front/chunk-000" / f"file-{int(stem.split('-')[-1]):06d}.mp4",
        dataset / "videos/observation.images.front/chunk-000" / f"file-{int(stem.split('-')[-1]):03d}.mp4",
    ]
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError((dataset, data_file))


def load_episode_rows(dataset: Path) -> pd.DataFrame:
    files = sorted((dataset / "meta/episodes").rglob("*.parquet"))
    if files:
        return pd.concat([pd.read_parquet(p) for p in files], ignore_index=True)
    rows = []
    for p in episode_files(dataset):
        df = pd.read_parquet(p, columns=["episode_index", "index"])
        rows.append(
            {
                "episode_index": int(df["episode_index"].iloc[0]),
                "task_index": 0,
                "length": len(df),
                "num_frames": len(df),
                "dataset_from_index": int(df["index"].min()),
                "dataset_to_index": int(df["index"].max()) + 1,
                "videos/observation.images.front/chunk_index": 0,
                "videos/observation.images.front/file_index": int(df["episode_index"].iloc[0]),
                "videos/observation.images.front/from_timestamp": 0.0,
                "videos/observation.images.front/to_timestamp": float((len(df) - 1) / 15),
            }
        )
    return pd.DataFrame(rows)


def make_tasks(out: Path) -> None:
    tasks = pd.DataFrame([{"task_index": 0, "task": PICKUP_TASK, "phase": "pickup"}])
    tasks.to_parquet(out / "meta/tasks.parquet", index=False)
    write_jsonl(out / "meta/tasks.jsonl", tasks.to_dict("records"))


def base_info(source: Path) -> dict:
    return json.loads((source / "meta/info.json").read_text())


def patch_info(source: Path, out: Path, episodes: int, frames: int) -> None:
    info = base_info(source)
    info["total_episodes"] = int(episodes)
    info["total_frames"] = int(frames)
    info["total_tasks"] = 1
    info["splits"] = {"train": f"0:{episodes}"}
    (out / "meta/info.json").write_text(json.dumps(info, indent=2) + "\n")
    src_stats = source / "meta/stats.json"
    if src_stats.exists():
        shutil.copy2(src_stats, out / "meta/stats.json")


def clip_video(src: Path, dst: Path, start_frame: int, num_frames: int, fps: int) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg",
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        str(src),
        "-ss",
        f"{start_frame / fps:.6f}",
        "-frames:v",
        str(num_frames),
        "-vf",
        f"fps={fps}",
        "-an",
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        str(dst),
    ]
    subprocess.run(cmd, check=True)


def eval1_split(df: pd.DataFrame, lift_threshold: float, gripper_max: float, max_split: int) -> int:
    state = np.stack(df["observation.state"].to_numpy())
    hits = np.where((state[:, 1] >= lift_threshold) & (np.abs(state[:, 5]) <= gripper_max))[0]
    if len(hits) == 0:
        return min(max_split, len(df) - 1)
    return min(int(hits[0]), max_split, len(df) - 1)


def init_out(out: Path, source_for_info: Path, force: bool) -> None:
    clean_dir(out, force)
    (out / "meta").mkdir(parents=True, exist_ok=True)
    (out / "meta/episodes/chunk-000").mkdir(parents=True, exist_ok=True)
    (out / "data/chunk-000").mkdir(parents=True, exist_ok=True)
    (out / "videos/observation.images.front/chunk-000").mkdir(parents=True, exist_ok=True)
    make_tasks(out)
    if (source_for_info / "meta/stats.json").exists():
        shutil.copy2(source_for_info / "meta/stats.json", out / "meta/stats.json")


def add_clip(
    *,
    out: Path,
    rows: list[pd.Series],
    jsonl_rows: list[dict],
    src_row: pd.Series,
    df: pd.DataFrame,
    src_video: Path,
    new_ep: int,
    global_index: int,
    fps: int,
    source_name: str,
    source_episode: int,
    split_idx: int | None,
) -> tuple[int, int]:
    n = len(df)
    df = df.copy().reset_index(drop=True)
    df["episode_index"] = new_ep
    df["task_index"] = 0
    df["index"] = range(global_index, global_index + n)
    df["frame_index"] = range(n)
    df["timestamp"] = [i / fps for i in range(n)]
    df.to_parquet(out / "data/chunk-000" / f"file-{new_ep:06d}.parquet", index=False)

    clip_video(src_video, out / "videos/observation.images.front/chunk-000" / f"file-{new_ep:03d}.mp4", 0, n, fps)

    ep_row = src_row.copy()
    ep_row["episode_index"] = new_ep
    ep_row["task_index"] = 0
    ep_row["length"] = n
    ep_row["num_frames"] = n
    ep_row["dataset_from_index"] = global_index
    ep_row["dataset_to_index"] = global_index + n
    ep_row["videos/observation.images.front/chunk_index"] = 0
    ep_row["videos/observation.images.front/file_index"] = new_ep
    ep_row["videos/observation.images.front/from_timestamp"] = 0.0
    ep_row["videos/observation.images.front/to_timestamp"] = (n - 1) / fps
    rows.append(ep_row)
    jsonl_rows.append(
        {
            "episode_index": new_ep,
            "tasks": [PICKUP_TASK],
            "length": n,
            "source": source_name,
            "source_episode_index": int(source_episode),
            "split_idx": None if split_idx is None else int(split_idx),
        }
    )
    return global_index + n, new_ep + 1


def finalize(out: Path, source_for_info: Path, rows: list[pd.Series], jsonl_rows: list[dict], frames: int) -> None:
    ep_df = pd.DataFrame(rows)
    int_cols = [
        c
        for c in ep_df.columns
        if c in {"episode_index", "task_index", "length", "num_frames", "dataset_from_index", "dataset_to_index"}
        or c.endswith("/chunk_index")
        or c.endswith("/file_index")
    ]
    for c in int_cols:
        ep_df[c] = ep_df[c].fillna(0)
        ep_df[c] = ep_df[c].astype("int64")
    ep_df.to_parquet(out / "meta/episodes/chunk-000/file-000.parquet", index=False)
    write_jsonl(out / "meta/episodes.jsonl", jsonl_rows)
    patch_info(source_for_info, out, len(ep_df), frames)


def append_eval1(
    *,
    out: Path,
    eval1: Path,
    rows: list[pd.Series],
    jsonl_rows: list[dict],
    global_index: int,
    new_ep: int,
    lift_threshold: float,
    gripper_max: float,
    max_split: int,
) -> tuple[int, int, dict]:
    fps = int(base_info(eval1).get("fps", 15))
    ep_rows = {int(r["episode_index"]): r for _, r in load_episode_rows(eval1).iterrows()}
    splits = []
    for p in episode_files(eval1):
        df_full = pd.read_parquet(p)
        src_ep = int(df_full["episode_index"].iloc[0])
        split = eval1_split(df_full, lift_threshold, gripper_max, max_split)
        df = df_full.iloc[: split + 1]
        global_index, new_ep = add_clip(
            out=out,
            rows=rows,
            jsonl_rows=jsonl_rows,
            src_row=ep_rows[src_ep],
            df=df,
            src_video=paired_video_file(eval1, p),
            new_ep=new_ep,
            global_index=global_index,
            fps=fps,
            source_name="eval1",
            source_episode=src_ep,
            split_idx=split,
        )
        splits.append(split)
    return global_index, new_ep, {"eval1_episodes": len(splits), "eval1_split_stats": pd.Series(splits).describe().to_dict()}


def append_eval2_pickup(
    *,
    out: Path,
    eval2: Path,
    rows: list[pd.Series],
    jsonl_rows: list[dict],
    global_index: int,
    new_ep: int,
    selected_source_episodes: set[int] | None = None,
) -> tuple[int, int, dict]:
    fps = int(base_info(eval2).get("fps", 15))
    ep_rows = {int(r["episode_index"]): r for _, r in load_episode_rows(eval2).iterrows()}
    used = []
    for p in episode_files(eval2):
        df = pd.read_parquet(p)
        src_ep = int(df["episode_index"].iloc[0])
        if selected_source_episodes is not None and src_ep not in selected_source_episodes:
            continue
        global_index, new_ep = add_clip(
            out=out,
            rows=rows,
            jsonl_rows=jsonl_rows,
            src_row=ep_rows[src_ep],
            df=df,
            src_video=video_file(eval2, src_ep),
            new_ep=new_ep,
            global_index=global_index,
            fps=fps,
            source_name="eval2",
            source_episode=src_ep,
            split_idx=None,
        )
        used.append(src_ep)
    return global_index, new_ep, {"eval2_episodes": len(used)}


def select_eval2_besthalf(split_csv: Path, n_per_color: int = 30) -> set[int]:
    df = pd.read_csv(split_csv)
    keep_rows = []
    for _, group in df.sort_values("pose_dist_min").groupby("target_color"):
        keep_rows.append(group.head(n_per_color))
    selected = pd.concat(keep_rows, ignore_index=True)
    return set(int(x) for x in selected["episode_index"])


def build(args: argparse.Namespace, key: str, out_name: str, mode: str) -> dict:
    out = args.out_root / out_name
    init_out(out, args.eval2_pickup, args.force)
    rows: list[pd.Series] = []
    jsonl_rows: list[dict] = []
    global_index = 0
    new_ep = 0
    summary = {"name": out_name, "mode": mode}

    if mode in {"eval1eval2", "eval1"}:
        global_index, new_ep, s = append_eval1(
            out=out,
            eval1=args.eval1_source,
            rows=rows,
            jsonl_rows=jsonl_rows,
            global_index=global_index,
            new_ep=new_ep,
            lift_threshold=args.eval1_lift_threshold,
            gripper_max=args.eval1_gripper_max,
            max_split=args.eval1_max_split,
        )
        summary.update(s)
    if mode == "eval1eval2":
        global_index, new_ep, s = append_eval2_pickup(
            out=out,
            eval2=args.eval2_pickup,
            rows=rows,
            jsonl_rows=jsonl_rows,
            global_index=global_index,
            new_ep=new_ep,
        )
        summary.update(s)
    if mode == "eval2besthalf":
        selected = select_eval2_besthalf(args.eval2_split_csv)
        global_index, new_ep, s = append_eval2_pickup(
            out=out,
            eval2=args.eval2_pickup,
            rows=rows,
            jsonl_rows=jsonl_rows,
            global_index=global_index,
            new_ep=new_ep,
            selected_source_episodes=selected,
        )
        summary.update(s)
        summary["selected_eval2_source_episodes"] = sorted(selected)

    finalize(out, args.eval2_pickup, rows, jsonl_rows, global_index)
    summary["total_episodes"] = len(rows)
    summary["total_frames"] = int(global_index)
    (out / "meta/pickup_rescue_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def main() -> None:
    args = parse_args()
    outputs = []
    if "b" in args.only:
        outputs.append(build(args, "b", args.name_b, "eval1eval2"))
    if "c" in args.only:
        outputs.append(build(args, "c", args.name_c, "eval2besthalf"))
    if "d" in args.only:
        outputs.append(build(args, "d", args.name_d, "eval1"))
    print(json.dumps(outputs, indent=2))


if __name__ == "__main__":
    main()
