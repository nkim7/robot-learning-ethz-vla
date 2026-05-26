#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Expand Eval2 trajectories so every source_bucket trajectory is trained with every prompt augmentation."
    )
    parser.add_argument(
        "--source",
        type=Path,
        default=Path("/data/eval2_180_chengming_promptaug_clean_256_v3"),
        help="Source LeRobot dataset with prompt-augmented tasks metadata.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("/data/lerobot_cache/robot-learning-group47/eval2_180_full_promptaug_clean256_v1"),
        help="Output dataset path.",
    )
    parser.add_argument(
        "--hub-repo",
        default="robot-learning-group47/eval2_180_full_promptaug_clean256_v1",
        help="Destination HF dataset repo for --push-to-hub.",
    )
    parser.add_argument("--force", action="store_true", help="Overwrite output directory.")
    parser.add_argument("--push-to-hub", action="store_true", help="Upload the dataset to Hugging Face Hub.")
    return parser.parse_args()


def clean_output(path: Path, force: bool) -> None:
    if path.exists():
        if not force:
            raise FileExistsError(f"{path} exists; pass --force to overwrite")
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def read_source_episode(source: Path, episode_index: int) -> pd.DataFrame:
    candidates = [
        source / "data/chunk-000" / f"file-{episode_index:06d}.parquet",
        source / "data/chunk-000" / f"file-{episode_index:03d}.parquet",
        source / "data/chunk-000" / f"file-{episode_index}.parquet",
    ]
    for path in candidates:
        if path.exists():
            return pd.read_parquet(path)
    raise FileNotFoundError(f"No source parquet found for episode {episode_index}")


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows))


def patch_info(source: Path, out: Path, total_episodes: int, total_frames: int, total_tasks: int) -> None:
    info = json.loads((source / "meta/info.json").read_text())
    info["total_episodes"] = total_episodes
    info["total_frames"] = total_frames
    info["total_tasks"] = total_tasks
    info["splits"] = {"train": f"0:{total_episodes}"}
    (out / "meta/info.json").write_text(json.dumps(info, indent=2) + "\n")


def copy_static_metadata(source: Path, out: Path) -> None:
    meta_out = out / "meta"
    meta_out.mkdir(parents=True, exist_ok=True)
    for name in ["stats.json", "tasks.parquet", "tasks.jsonl"]:
        src = source / "meta" / name
        if src.exists():
            shutil.copy2(src, meta_out / name)
    videos_src = source / "videos"
    if videos_src.exists():
        shutil.copytree(videos_src, out / "videos", ignore=shutil.ignore_patterns(".cache"))


def create_dataset(source: Path, out: Path, force: bool) -> None:
    clean_output(out, force)
    copy_static_metadata(source, out)

    tasks = pd.read_parquet(source / "meta/tasks.parquet").reset_index(drop=True)
    episodes = pd.concat(
        [pd.read_parquet(p) for p in sorted((source / "meta/episodes").rglob("*.parquet"))],
        ignore_index=True,
    )

    task_to_bucket = dict(zip(tasks["task_index"].astype(int), tasks["source_bucket"].astype(str)))
    tasks_by_bucket = {
        bucket: group.sort_values("task_index").reset_index(drop=True)
        for bucket, group in tasks.groupby("source_bucket", sort=True)
    }

    data_dir = out / "data/chunk-000"
    episode_dir = out / "meta/episodes/chunk-000"
    data_dir.mkdir(parents=True, exist_ok=True)
    episode_dir.mkdir(parents=True, exist_ok=True)

    new_episode_rows: list[pd.Series] = []
    jsonl_rows: list[dict] = []
    total_frames = 0
    global_index = 0
    new_ep = 0

    for _, src_ep_row in episodes.sort_values("episode_index").iterrows():
        src_ep = int(src_ep_row["episode_index"])
        src_task = int(src_ep_row["task_index"])
        bucket = task_to_bucket[src_task]
        src_df = read_source_episode(source, src_ep)
        prompt_rows = tasks_by_bucket[bucket]

        for task_row in prompt_rows.itertuples(index=False):
            task_index = int(task_row.task_index)
            df = src_df.copy()
            n = len(df)
            df["episode_index"] = new_ep
            df["task_index"] = task_index
            df["index"] = range(global_index, global_index + n)
            df.to_parquet(data_dir / f"file-{new_ep:06d}.parquet", index=False)

            ep_row = src_ep_row.copy()
            ep_row["episode_index"] = new_ep
            ep_row["task_index"] = task_index
            ep_row["length"] = n
            ep_row["num_frames"] = n
            ep_row["dataset_from_index"] = global_index
            ep_row["dataset_to_index"] = global_index + n
            new_episode_rows.append(ep_row)
            jsonl_rows.append(
                {
                    "episode_index": new_ep,
                    "tasks": [task_row.task],
                    "length": n,
                    "source_episode_index": src_ep,
                    "source_bucket": bucket,
                    "layout_code": task_row.layout_code,
                    "target_color": task_row.target_color,
                }
            )

            global_index += n
            total_frames += n
            new_ep += 1

    episode_df = pd.DataFrame(new_episode_rows)
    int_cols = [
        col
        for col in episode_df.columns
        if col in {"episode_index", "task_index", "length", "num_frames", "dataset_from_index", "dataset_to_index"}
        or col.endswith("/chunk_index")
        or col.endswith("/file_index")
    ]
    for col in int_cols:
        episode_df[col] = episode_df[col].astype("int64")
    episode_df.to_parquet(episode_dir / "file-000.parquet", index=False)
    write_jsonl(out / "meta/episodes.jsonl", jsonl_rows)
    patch_info(source, out, total_episodes=new_ep, total_frames=total_frames, total_tasks=len(tasks))

    summary = {
        "source": str(source),
        "total_source_episodes": int(len(episodes)),
        "total_tasks": int(len(tasks)),
        "total_expanded_episodes": int(new_ep),
        "total_frames": int(total_frames),
        "episodes_per_source_bucket": episodes.assign(
            source_bucket=episodes["task_index"].map(task_to_bucket)
        )["source_bucket"].value_counts().sort_index().to_dict(),
        "tasks_per_source_bucket": tasks["source_bucket"].value_counts().sort_index().to_dict(),
    }
    (out / "meta/full_promptaug_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


def verify_dataset(out: Path) -> None:
    info = json.loads((out / "meta/info.json").read_text())
    episodes = pd.concat(
        [pd.read_parquet(p) for p in sorted((out / "meta/episodes").rglob("*.parquet"))],
        ignore_index=True,
    )
    data_files = sorted((out / "data").rglob("*.parquet"))
    if len(episodes) != info["total_episodes"]:
        raise AssertionError((len(episodes), info["total_episodes"]))
    if len(data_files) != info["total_episodes"]:
        raise AssertionError((len(data_files), info["total_episodes"]))
    frame_count = sum(len(pd.read_parquet(p, columns=["index"])) for p in data_files)
    if frame_count != info["total_frames"]:
        raise AssertionError((frame_count, info["total_frames"]))
    print(f"verified episodes={len(episodes)} frames={frame_count} tasks={info['total_tasks']}")


def push_to_hub(out: Path, repo_id: str) -> None:
    from huggingface_hub import HfApi

    api = HfApi()
    api.create_repo(repo_id=repo_id, repo_type="dataset", exist_ok=True)
    api.upload_folder(
        folder_path=str(out),
        repo_id=repo_id,
        repo_type="dataset",
        ignore_patterns=[".cache/*", "**/.cache/*"],
    )
    print(f"uploaded {out} to {repo_id}")


def main() -> None:
    args = parse_args()
    create_dataset(args.source, args.out, args.force)
    verify_dataset(args.out)
    if args.push_to_hub:
        push_to_hub(args.out, args.hub_repo)


if __name__ == "__main__":
    main()
