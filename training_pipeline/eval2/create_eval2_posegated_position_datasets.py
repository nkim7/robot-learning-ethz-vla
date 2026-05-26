#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from pathlib import Path

import pandas as pd


PICKUP_TASK = "Pick up the banana and lift it until all three bowls are visible."
POSITION_TASKS = {
    "left": "Put the banana into the left bowl from the robot perspective.",
    "middle": "Put the banana into the middle bowl from the robot perspective.",
    "right": "Put the banana into the right bowl from the robot perspective.",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create Eval2 pose-gated position-policy datasets.")
    parser.add_argument("--source", type=Path, default=Path("/data/eval2_180_chengming_promptaug_clean_256_v3"))
    parser.add_argument(
        "--split-csv",
        type=Path,
        default=Path("/data/eval2_180_2phase_inspect_posebased/eval2_180_split_candidates_posebased.csv"),
    )
    parser.add_argument(
        "--out-root",
        type=Path,
        default=Path("/data/lerobot_cache/robot-learning-group47"),
    )
    parser.add_argument("--uniform-name", default="eval2_180_posegated_position_uniform_clean256_v1")
    parser.add_argument("--pickup-name", default="eval2_180_posegated_pickuplift_clean256_v1")
    parser.add_argument("--place-name", default="eval2_180_posegated_position_place_clean256_v1")
    parser.add_argument("--min-split", type=int, default=105)
    parser.add_argument("--max-split", type=int, default=130)
    parser.add_argument("--min-phase1", type=int, default=90)
    parser.add_argument("--min-phase2", type=int, default=35)
    parser.add_argument("--gripper-min", type=float, default=2.5)
    parser.add_argument("--gripper-max", type=float, default=10.0)
    parser.add_argument("--force", action="store_true")
    parser.add_argument(
        "--only",
        nargs="*",
        choices=["uniform", "pickup", "place"],
        default=["uniform", "pickup", "place"],
        help="Dataset kinds to build.",
    )
    parser.add_argument(
        "--clip-videos",
        action="store_true",
        help="Write one clipped video per output episode and reset timestamps. Needed for clipped phase-2 datasets.",
    )
    return parser.parse_args()


def clean_dir(path: Path, force: bool) -> None:
    if path.exists():
        if not force:
            raise FileExistsError(f"{path} exists; pass --force to overwrite")
        shutil.rmtree(path)
    path.mkdir(parents=True)


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows))


def read_source_episode(source: Path, episode_index: int) -> pd.DataFrame:
    for fmt in (f"file-{episode_index:06d}.parquet", f"file-{episode_index:03d}.parquet"):
        path = source / "data/chunk-000" / fmt
        if path.exists():
            return pd.read_parquet(path)
    raise FileNotFoundError(f"episode {episode_index}")


def target_position(layout_left_to_right: str, target_color: str) -> str:
    colors = layout_left_to_right.split("-")
    return ["left", "middle", "right"][colors.index(target_color)]


def make_tasks(kind: str) -> pd.DataFrame:
    if kind == "uniform":
        rows = [{"task_index": 0, "task": PICKUP_TASK, "phase": "pickup", "target_position": ""}]
        rows += [
            {"task_index": i, "task": POSITION_TASKS[pos], "phase": "place", "target_position": pos}
            for i, pos in enumerate(["left", "middle", "right"], start=1)
        ]
    elif kind == "pickup":
        rows = [{"task_index": 0, "task": PICKUP_TASK, "phase": "pickup", "target_position": ""}]
    elif kind == "place":
        rows = [
            {"task_index": i, "task": POSITION_TASKS[pos], "phase": "place", "target_position": pos}
            for i, pos in enumerate(["left", "middle", "right"])
        ]
    else:
        raise ValueError(kind)
    return pd.DataFrame(rows)


def copy_static(source: Path, out: Path, tasks: pd.DataFrame, copy_videos: bool) -> None:
    (out / "meta").mkdir(parents=True, exist_ok=True)
    for name in ["stats.json"]:
        src = source / "meta" / name
        if src.exists():
            shutil.copy2(src, out / "meta" / name)
    tasks.to_parquet(out / "meta/tasks.parquet", index=False)
    write_jsonl(out / "meta/tasks.jsonl", tasks.to_dict("records"))
    if copy_videos and (source / "videos").exists():
        shutil.copytree(source / "videos", out / "videos", ignore=shutil.ignore_patterns(".cache"))


def clip_video(source: Path, out: Path, src_ep: int, start_frame: int, num_frames: int, fps: int) -> None:
    src = source / "videos/observation.images.front/chunk-000" / f"file-{src_ep:03d}.mp4"
    if not src.exists():
        src = source / "videos/observation.images.front/chunk-000" / f"file-{src_ep:06d}.mp4"
    if not src.exists():
        raise FileNotFoundError(src)
    out.parent.mkdir(parents=True, exist_ok=True)
    start_s = start_frame / fps
    cmd = [
        "ffmpeg",
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        str(src),
        "-ss",
        f"{start_s:.6f}",
        "-frames:v",
        str(num_frames),
        "-vf",
        f"fps={fps}",
        "-an",
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        str(out),
    ]
    subprocess.run(cmd, check=True)


def patch_info(source: Path, out: Path, total_episodes: int, total_frames: int, total_tasks: int) -> None:
    info = json.loads((source / "meta/info.json").read_text())
    info["total_episodes"] = int(total_episodes)
    info["total_frames"] = int(total_frames)
    info["total_tasks"] = int(total_tasks)
    info["splits"] = {"train": f"0:{total_episodes}"}
    (out / "meta/info.json").write_text(json.dumps(info, indent=2) + "\n")


def task_index_for(kind: str, phase: str, position: str) -> int:
    if kind == "pickup":
        return 0
    if kind == "place":
        return {"left": 0, "middle": 1, "right": 2}[position]
    if phase == "pickup":
        return 0
    return {"left": 1, "middle": 2, "right": 3}[position]


def add_clip(
    *,
    out: Path,
    kind: str,
    rows: list[pd.Series],
    jsonl_rows: list[dict],
    src_ep_row: pd.Series,
    src_df: pd.DataFrame,
    src_ep: int,
    clip: slice,
    phase: str,
    position: str,
    global_index: int,
    new_ep: int,
    split: int,
    layout: str,
    target_color: str,
    clip_videos: bool,
    source: Path,
    fps: int,
) -> tuple[int, int]:
    start_frame = 0 if clip.start is None else int(clip.start)
    df = src_df.iloc[clip].copy().reset_index(drop=True)
    n = len(df)
    if n <= 0:
        raise ValueError((src_ep, phase, clip))
    task_index = task_index_for(kind, phase, position)
    first_ts = 0.0 if clip_videos else float(df["timestamp"].iloc[0])
    last_ts = (n - 1) / fps if clip_videos else float(df["timestamp"].iloc[-1])
    df["episode_index"] = new_ep
    df["task_index"] = task_index
    df["index"] = range(global_index, global_index + n)
    df["frame_index"] = range(n)
    if clip_videos:
        df["timestamp"] = [i / fps for i in range(n)]
    df.to_parquet(out / "data/chunk-000" / f"file-{new_ep:06d}.parquet", index=False)

    ep_row = src_ep_row.copy()
    ep_row["episode_index"] = new_ep
    ep_row["task_index"] = task_index
    ep_row["length"] = n
    ep_row["num_frames"] = n
    ep_row["dataset_from_index"] = global_index
    ep_row["dataset_to_index"] = global_index + n
    if clip_videos:
        ep_row["videos/observation.images.front/chunk_index"] = 0
        ep_row["videos/observation.images.front/file_index"] = new_ep
    ep_row["videos/observation.images.front/from_timestamp"] = first_ts
    ep_row["videos/observation.images.front/to_timestamp"] = last_ts
    if clip_videos:
        clip_video(
            source,
            out / "videos/observation.images.front/chunk-000" / f"file-{new_ep:03d}.mp4",
            src_ep=src_ep,
            start_frame=start_frame,
            num_frames=n,
            fps=fps,
        )
    rows.append(ep_row)
    jsonl_rows.append(
        {
            "episode_index": int(new_ep),
            "tasks": [PICKUP_TASK if phase == "pickup" else POSITION_TASKS[position]],
            "length": int(n),
            "source_episode_index": int(src_ep),
            "phase": phase,
            "split_idx": int(split),
            "target_position": position,
            "target_color": target_color,
            "layout_left_to_right": layout,
        }
    )
    return global_index + n, new_ep + 1


def build_dataset(
    *,
    source: Path,
    out: Path,
    kind: str,
    clean_rows: pd.DataFrame,
    episodes: pd.DataFrame,
    force: bool,
    clip_videos: bool,
) -> dict:
    clean_dir(out, force)
    tasks = make_tasks(kind)
    info = json.loads((source / "meta/info.json").read_text())
    fps = int(info.get("fps", 15))
    copy_static(source, out, tasks, copy_videos=not clip_videos)
    (out / "data/chunk-000").mkdir(parents=True, exist_ok=True)
    (out / "meta/episodes/chunk-000").mkdir(parents=True, exist_ok=True)
    if clip_videos:
        (out / "videos/observation.images.front/chunk-000").mkdir(parents=True, exist_ok=True)

    src_ep_rows = {int(r["episode_index"]): r for _, r in episodes.iterrows()}
    new_episode_rows: list[pd.Series] = []
    jsonl_rows: list[dict] = []
    global_index = 0
    new_ep = 0

    for r in clean_rows.itertuples(index=False):
        src_ep = int(r.episode_index)
        src_df = read_source_episode(source, src_ep)
        split = int(r.final_split)
        position = str(r.target_position)
        src_ep_row = src_ep_rows[src_ep].copy()
        if kind in {"uniform", "pickup"}:
            global_index, new_ep = add_clip(
                out=out,
                kind=kind,
                rows=new_episode_rows,
                jsonl_rows=jsonl_rows,
                src_ep_row=src_ep_row,
                src_df=src_df,
                src_ep=src_ep,
                clip=slice(0, split + 1),
                phase="pickup",
                position=position,
                global_index=global_index,
                new_ep=new_ep,
                split=split,
                layout=r.layout_left_to_right,
                target_color=r.target_color,
                clip_videos=clip_videos,
                source=source,
                fps=fps,
            )
        if kind in {"uniform", "place"}:
            global_index, new_ep = add_clip(
                out=out,
                kind=kind,
                rows=new_episode_rows,
                jsonl_rows=jsonl_rows,
                src_ep_row=src_ep_row,
                src_df=src_df,
                src_ep=src_ep,
                clip=slice(split, len(src_df)),
                phase="place",
                position=position,
                global_index=global_index,
                new_ep=new_ep,
                split=split,
                layout=r.layout_left_to_right,
                target_color=r.target_color,
                clip_videos=clip_videos,
                source=source,
                fps=fps,
            )

    episode_df = pd.DataFrame(new_episode_rows)
    int_cols = [
        c
        for c in episode_df.columns
        if c in {"episode_index", "task_index", "length", "num_frames", "dataset_from_index", "dataset_to_index"}
        or c.endswith("/chunk_index")
        or c.endswith("/file_index")
    ]
    for c in int_cols:
        episode_df[c] = episode_df[c].astype("int64")
    episode_df.to_parquet(out / "meta/episodes/chunk-000/file-000.parquet", index=False)
    write_jsonl(out / "meta/episodes.jsonl", jsonl_rows)
    patch_info(source, out, len(episode_df), global_index, len(tasks))
    return {
        "name": out.name,
        "kind": kind,
        "episodes": int(len(episode_df)),
        "frames": int(global_index),
        "tasks": int(len(tasks)),
    }


def verify(out: Path) -> None:
    info = json.loads((out / "meta/info.json").read_text())
    episodes = pd.concat([pd.read_parquet(p) for p in sorted((out / "meta/episodes").rglob("*.parquet"))])
    files = sorted((out / "data").rglob("*.parquet"))
    assert len(episodes) == info["total_episodes"], (len(episodes), info["total_episodes"])
    assert len(files) == info["total_episodes"], (len(files), info["total_episodes"])
    frames = sum(len(pd.read_parquet(p, columns=["index"])) for p in files)
    assert frames == info["total_frames"], (frames, info["total_frames"])


def main() -> None:
    args = parse_args()
    split_df = pd.read_csv(args.split_csv)
    episodes = pd.concat(
        [pd.read_parquet(p) for p in sorted((args.source / "meta/episodes").rglob("*.parquet"))],
        ignore_index=True,
    )

    clean = split_df.copy()
    clean["target_position"] = clean.apply(lambda r: target_position(r.layout_left_to_right, r.target_color), axis=1)
    clean["final_split"] = clean["split_idx"].clip(args.min_split, args.max_split).astype(int)
    keep = []
    reasons = []
    for r in clean.itertuples(index=False):
        df = read_source_episode(args.source, int(r.episode_index))
        state = list(df["observation.state"].iloc[int(r.final_split)])
        gripper = float(state[5])
        phase1_len = int(r.final_split) + 1
        phase2_len = len(df) - int(r.final_split)
        ok = True
        reason = []
        if phase1_len < args.min_phase1:
            ok = False
            reason.append("phase1_short")
        if phase2_len < args.min_phase2:
            ok = False
            reason.append("phase2_short")
        if not (args.gripper_min <= gripper <= args.gripper_max):
            ok = False
            reason.append(f"gripper_{gripper:.2f}")
        keep.append(ok)
        reasons.append(",".join(reason) if reason else "kept")
    clean["kept"] = keep
    clean["filter_reason"] = reasons
    kept = clean[clean["kept"]].reset_index(drop=True)

    outputs = {
        "uniform": args.out_root / args.uniform_name,
        "pickup": args.out_root / args.pickup_name,
        "place": args.out_root / args.place_name,
    }
    summaries = []
    for kind, out in outputs.items():
        if kind not in args.only:
            continue
        summaries.append(
            build_dataset(
                source=args.source,
                out=out,
                kind=kind,
                clean_rows=kept,
                episodes=episodes,
                force=args.force,
                clip_videos=args.clip_videos,
            )
        )
        verify(out)

    audit = {
        "source": str(args.source),
        "split_csv": str(args.split_csv),
        "total_source_episodes": int(len(clean)),
        "kept_episodes": int(len(kept)),
        "dropped_episodes": int((~clean["kept"]).sum()),
        "drop_reasons": clean.loc[~clean["kept"], ["episode_index", "filter_reason"]].to_dict("records"),
        "target_position_counts": kept["target_position"].value_counts().sort_index().to_dict(),
        "layout_position_counts": kept.groupby(["layout_left_to_right", "target_position"]).size().unstack(fill_value=0).to_dict(),
        "split_stats": kept["final_split"].describe().to_dict(),
        "outputs": summaries,
    }
    for kind, out in outputs.items():
        if kind not in args.only:
            continue
        (out / "meta/posegated_summary.json").write_text(json.dumps(audit, indent=2) + "\n")
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
