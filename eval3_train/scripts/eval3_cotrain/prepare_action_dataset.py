#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

from cotrain_common import (
    ACTION_LMR_DATASET_ROOT,
    ACTION_TOY_DATASET_ROOT,
    PREPARED_ACTION_DATASET_ROOT,
)


VIDEO_KEY = "observation.images.front"
MANIFEST_NAME = "eval3_cotrain_action_manifest.json"


def load_info(root: Path) -> dict:
    with (root / "meta" / "info.json").open() as f:
        return json.load(f)


def load_tasks(root: Path) -> pd.DataFrame:
    return pd.read_parquet(root / "meta" / "tasks.parquet")


def task_lookup(root: Path) -> dict[int, str]:
    tasks = load_tasks(root).reset_index()
    return {int(row.task_index): str(row.task) for row in tasks.itertuples(index=False)}


def source_episode_meta(root: Path, episode_index: int) -> pd.DataFrame:
    return pd.read_parquet(root / "meta" / "episodes" / "chunk-000" / f"file-{episode_index:03d}.parquet")


def source_data(root: Path, episode_index: int) -> pd.DataFrame:
    return pd.read_parquet(root / "data" / "chunk-000" / f"file-{episode_index:03d}.parquet")


def replace_symlink(src: Path, dst: Path) -> None:
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    os.symlink(src, dst)


def scalar_stats(values: np.ndarray) -> dict:
    values = values.astype(np.float64)
    return {
        "min": [float(np.min(values))],
        "max": [float(np.max(values))],
        "mean": [float(np.mean(values))],
        "std": [float(np.std(values))],
        "count": [int(values.size)],
        "q01": [float(np.quantile(values, 0.01))],
        "q10": [float(np.quantile(values, 0.10))],
        "q50": [float(np.quantile(values, 0.50))],
        "q90": [float(np.quantile(values, 0.90))],
        "q99": [float(np.quantile(values, 0.99))],
    }


def vector_stats(values: np.ndarray) -> dict:
    values = values.astype(np.float64)
    return {
        "min": np.min(values, axis=0).tolist(),
        "max": np.max(values, axis=0).tolist(),
        "mean": np.mean(values, axis=0).tolist(),
        "std": np.std(values, axis=0).tolist(),
        "count": [int(values.shape[0])],
        "q01": np.quantile(values, 0.01, axis=0).tolist(),
        "q10": np.quantile(values, 0.10, axis=0).tolist(),
        "q50": np.quantile(values, 0.50, axis=0).tolist(),
        "q90": np.quantile(values, 0.90, axis=0).tolist(),
        "q99": np.quantile(values, 0.99, axis=0).tolist(),
    }


def image_stats_from_sources(toy_root: Path) -> dict:
    # Image normalization is IDENTITY for this policy; keep valid channel stats
    # from the TOY dataset so LeRobot metadata remains complete.
    with (toy_root / "meta" / "stats.json").open() as f:
        return json.load(f)["observation.images.front"]


def compute_stats(all_data: list[pd.DataFrame], toy_root: Path) -> dict:
    df = pd.concat(all_data, ignore_index=True)
    action = np.stack(df["action"].to_numpy())
    state = np.stack(df["observation.state"].to_numpy())
    return {
        "episode_index": scalar_stats(df["episode_index"].to_numpy()),
        "task_index": scalar_stats(df["task_index"].to_numpy()),
        "observation.images.front": image_stats_from_sources(toy_root),
        "observation.state": vector_stats(state),
        "index": scalar_stats(df["index"].to_numpy()),
        "frame_index": scalar_stats(df["frame_index"].to_numpy()),
        "timestamp": scalar_stats(df["timestamp"].to_numpy()),
        "action": vector_stats(action),
    }


def create_dataset(
    toy_root: Path,
    lmr_root: Path,
    output_root: Path,
    overwrite: bool,
    toy_repeats: int,
    lmr_repeats: int,
) -> None:
    manifest_path = output_root / MANIFEST_NAME
    if output_root.exists():
        if not overwrite and manifest_path.exists():
            print(f"Prepared action dataset already exists: {output_root}")
            return
        if not overwrite:
            raise FileExistsError(
                f"{output_root} already exists but is not a prepared co-train dataset. "
                "Pass --overwrite to replace it."
            )
        shutil.rmtree(output_root)

    toy_info = load_info(toy_root)
    lmr_info = load_info(lmr_root)
    toy_tasks = task_lookup(toy_root)
    lmr_tasks = task_lookup(lmr_root)

    all_task_names: list[str] = []
    for lookup in (toy_tasks, lmr_tasks):
        for task in lookup.values():
            if task not in all_task_names:
                all_task_names.append(task)
    task_to_new_index = {task: index for index, task in enumerate(all_task_names)}

    (output_root / "meta" / "episodes" / "chunk-000").mkdir(parents=True)
    (output_root / "data" / "chunk-000").mkdir(parents=True)
    (output_root / "videos" / VIDEO_KEY / "chunk-000").mkdir(parents=True)

    total_frames = 0
    new_episode_index = 0
    all_episode_meta = []
    all_data = []

    source_specs = (
        ("toy", toy_root, toy_info, toy_tasks, toy_repeats),
        ("lmr", lmr_root, lmr_info, lmr_tasks, lmr_repeats),
    )
    for source_name, root, info, lookup, repeats in source_specs:
        for repeat_index in range(repeats):
            for old_episode_index in range(int(info["total_episodes"])):
                data = source_data(root, old_episode_index).copy()
                meta = source_episode_meta(root, old_episode_index).copy()

                old_task_index = int(data["task_index"].iloc[0])
                task_name = lookup[old_task_index]
                new_task_index = task_to_new_index[task_name]
                length = len(data)

                data["episode_index"] = new_episode_index
                data["index"] = np.arange(total_frames, total_frames + length, dtype=np.int64)
                data["task_index"] = new_task_index
                data["frame_index"] = np.arange(length, dtype=np.int64)

                meta["episode_index"] = new_episode_index
                meta["tasks"] = [[task_name]]
                meta["length"] = length
                meta["data/chunk_index"] = 0
                meta["data/file_index"] = new_episode_index
                meta["dataset_from_index"] = total_frames
                meta["dataset_to_index"] = total_frames + length
                meta[f"videos/{VIDEO_KEY}/chunk_index"] = 0
                meta[f"videos/{VIDEO_KEY}/file_index"] = new_episode_index
                meta[f"videos/{VIDEO_KEY}/from_timestamp"] = 0.0
                meta[f"videos/{VIDEO_KEY}/to_timestamp"] = float(
                    data["timestamp"].iloc[-1] + 1.0 / info["fps"]
                )
                meta["meta/episodes/chunk_index"] = 0
                meta["meta/episodes/file_index"] = new_episode_index

                dst_data = output_root / "data" / "chunk-000" / f"file-{new_episode_index:03d}.parquet"
                dst_meta = (
                    output_root
                    / "meta"
                    / "episodes"
                    / "chunk-000"
                    / f"file-{new_episode_index:03d}.parquet"
                )
                data.to_parquet(dst_data, index=False)
                meta.to_parquet(dst_meta, index=False)

                src_video = (
                    root
                    / "videos"
                    / VIDEO_KEY
                    / "chunk-000"
                    / f"file-{old_episode_index:03d}.mp4"
                )
                dst_video = (
                    output_root
                    / "videos"
                    / VIDEO_KEY
                    / "chunk-000"
                    / f"file-{new_episode_index:03d}.mp4"
                )
                replace_symlink(src_video.resolve(), dst_video)

                meta["source_dataset"] = source_name
                meta["source_episode_index"] = old_episode_index
                meta["source_repeat_index"] = repeat_index
                all_episode_meta.append(meta)
                all_data.append(data)
                total_frames += length
                new_episode_index += 1

    tasks_df = pd.DataFrame(
        {"task": all_task_names, "task_index": list(range(len(all_task_names)))}
    ).set_index("task")
    tasks_df.to_parquet(output_root / "meta" / "tasks.parquet")

    info_out = dict(toy_info)
    info_out["repo_id"] = None
    info_out["total_episodes"] = new_episode_index
    info_out["total_frames"] = total_frames
    info_out["total_tasks"] = len(all_task_names)
    info_out["splits"] = {"train": f"0:{new_episode_index}"}
    info_out["data_files_size_in_mb"] = int(
        sum(path.stat().st_size for path in (output_root / "data").rglob("*.parquet")) / 1_000_000
    )
    info_out["video_files_size_in_mb"] = int(
        sum(path.stat().st_size for path in (output_root / "videos").rglob("*.mp4")) / 1_000_000
    )
    with (output_root / "meta" / "info.json").open("w") as f:
        json.dump(info_out, f, indent=2)

    with (output_root / "meta" / "stats.json").open("w") as f:
        json.dump(compute_stats(all_data, toy_root), f, indent=2)

    merged_episode_meta = pd.concat(all_episode_meta, ignore_index=True)
    merged_episode_meta.to_parquet(output_root / "meta" / "episodes.parquet", index=False)

    manifest = {
        "toy_dataset": str(toy_root),
        "lmr_dataset": str(lmr_root),
        "output_dataset": str(output_root),
        "total_episodes": new_episode_index,
        "total_frames": total_frames,
        "toy_repeats": toy_repeats,
        "lmr_repeats": lmr_repeats,
        "tasks": all_task_names,
    }
    with manifest_path.open("w") as f:
        json.dump(manifest, f, indent=2)
    print(json.dumps(manifest, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--toy-root", type=Path, default=ACTION_TOY_DATASET_ROOT)
    parser.add_argument("--lmr-root", type=Path, default=ACTION_LMR_DATASET_ROOT)
    parser.add_argument("--output-root", type=Path, default=PREPARED_ACTION_DATASET_ROOT)
    parser.add_argument("--toy-repeats", type=int, default=1)
    parser.add_argument("--lmr-repeats", type=int, default=1)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    create_dataset(
        args.toy_root,
        args.lmr_root,
        args.output_root,
        args.overwrite,
        args.toy_repeats,
        args.lmr_repeats,
    )


if __name__ == "__main__":
    main()
