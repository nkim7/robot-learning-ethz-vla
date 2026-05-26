#!/usr/bin/env python3
import json
import re
import shutil
import sys
from pathlib import Path

import pandas as pd


def read_json(path: Path):
    return json.loads(path.read_text())


def write_json(path: Path, obj):
    path.write_text(json.dumps(obj, indent=2) + "\n")


def remove_path(path: Path):
    if not path.exists():
        return

    print(f"Removing: {path}")

    if path.is_dir():
        shutil.rmtree(path)
    else:
        path.unlink()


def file_index(path: Path):
    """
    Extract index from file-003.parquet or file-003.mp4.
    """
    match = re.search(r"file-(\d+)", path.name)
    if not match:
        return -1
    return int(match.group(1))


def update_info_json(info_path: Path, remaining_episode_count: int, remaining_rows: int):
    if not info_path.exists():
        return

    try:
        info = read_json(info_path)
    except Exception as e:
        print(f"Warning: could not read {info_path}: {e}")
        return

    if "total_episodes" in info:
        info["total_episodes"] = remaining_episode_count

    if "total_frames" in info:
        info["total_frames"] = remaining_rows

    if "splits" in info and isinstance(info["splits"], dict):
        info["splits"]["train"] = f"0:{remaining_episode_count}"

    write_json(info_path, info)
    print(f"Updated: {info_path}")


def remove_empty_dirs(root: Path):
    for base_name in ["data", "videos", "meta"]:
        base = root / base_name
        if not base.exists():
            continue

        for d in sorted([x for x in base.rglob("*") if x.is_dir()], reverse=True):
            try:
                d.rmdir()
                print(f"Removed empty dir: {d}")
            except OSError:
                pass


def main():
    if len(sys.argv) != 2:
        print("Usage: delete_latest_eval1_episode.py DATASET_ROOT")
        sys.exit(1)

    root = Path(sys.argv[1]).expanduser().resolve()

    if not root.exists():
        print(f"Dataset root does not exist: {root}")
        sys.exit(1)

    print("")
    print("========================================")
    print("Deleting latest LeRobot v3 episode")
    print(f"Dataset root: {root}")
    print("========================================")

    data_files = sorted(
        root.glob("data/chunk-*/file-*.parquet"),
        key=file_index,
    )

    if not data_files:
        print("No data/chunk-*/file-*.parquet files found.")
        return

    latest_data_file = data_files[-1]

    rel = latest_data_file.relative_to(root)
    chunk_dir = rel.parts[1]           # chunk-000
    file_name = rel.name               # file-003.parquet
    file_stem = latest_data_file.stem  # file-003

    latest_episode_index = None
    latest_rows = 0

    try:
        df_latest = pd.read_parquet(latest_data_file)
        latest_rows = len(df_latest)

        if "episode_index" in df_latest.columns and len(df_latest) > 0:
            latest_episode_index = int(df_latest["episode_index"].max())
    except Exception as e:
        print(f"Warning: could not read latest data file {latest_data_file}: {e}")

    print(f"Latest data file: {latest_data_file}")
    print(f"Chunk: {chunk_dir}")
    print(f"File stem: {file_stem}")
    print(f"Rows in latest file: {latest_rows}")

    if latest_episode_index is not None:
        print(f"Episode index inside latest file: {latest_episode_index}")

    # 1. Delete main data parquet.
    remove_path(latest_data_file)

    # 2. Delete matching videos:
    # videos/{video_key}/chunk-000/file-003.mp4
    videos_dir = root / "videos"
    if videos_dir.exists():
        for video_file in videos_dir.glob(f"*/{chunk_dir}/{file_stem}.*"):
            remove_path(video_file)

    # 3. Delete matching meta episode parquet:
    # meta/episodes/chunk-000/file-003.parquet
    remove_path(root / "meta" / "episodes" / chunk_dir / file_name)

    # 4. Delete matching episode stats parquet if present.
    remove_path(root / "meta" / "episodes_stats" / chunk_dir / file_name)
    remove_path(root / "meta" / "episode_stats" / chunk_dir / file_name)

    # 5. Remove empty directories.
    remove_empty_dirs(root)

    # 6. Recompute remaining episode count and frame count from remaining data parquet files.
    remaining_data_files = sorted(root.glob("data/chunk-*/file-*.parquet"), key=file_index)

    remaining_eps = set()
    remaining_rows = 0

    for p in remaining_data_files:
        try:
            df = pd.read_parquet(p)
        except Exception as e:
            print(f"Warning: could not read {p}: {e}")
            continue

        remaining_rows += len(df)

        if "episode_index" in df.columns:
            remaining_eps.update(int(x) for x in df["episode_index"].dropna().unique())

    remaining_episode_count = len(remaining_eps)

    # 7. Update meta/info.json.
    update_info_json(root / "meta" / "info.json", remaining_episode_count, remaining_rows)

    print("")
    print("Deletion complete.")
    print(f"Deleted data file: {rel}")
    print(f"Deleted matching videos: videos/*/{chunk_dir}/{file_stem}.*")
    print(f"Deleted matching meta: meta/episodes/{chunk_dir}/{file_name}")
    print(f"Remaining episode indices: {sorted(remaining_eps)}")
    print(f"Remaining episode count: {remaining_episode_count}")
    print(f"Remaining frames: {remaining_rows}")
    print("")


if __name__ == "__main__":
    main()
