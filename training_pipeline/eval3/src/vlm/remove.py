import json
import shutil
import time
from pathlib import Path
from collections import defaultdict

import pandas as pd
import torch

from lerobot.datasets.lerobot_dataset import LeRobotDataset


# ---------------------------------------------------------------------
# Config for NEW VIDEO DATASET
# ---------------------------------------------------------------------

REPO_ID = "robot-learning-group47/eval3_toy_video"

OLD_ROOT = Path("data/eval3_toy_video")
NEW_ROOT = Path("data/eval3_toy_video_new")

NUM_LAST_EPISODES_TO_REMOVE = 1


# ---------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------

def data_dir(root: Path) -> Path:
    return root / "data"


def episode_metadata_dir(root: Path) -> Path:
    return root / "meta" / "episodes"


def find_all_parquets(folder: Path):
    if not folder.exists():
        return []
    return sorted(folder.rglob("*.parquet"))


def parquet_sort_key(path: Path):
    stem = path.stem
    digits = "".join(ch for ch in stem if ch.isdigit())

    if digits:
        return int(digits)

    return str(path)


def get_data_parquet_files(root: Path):
    files = find_all_parquets(data_dir(root))

    if not files:
        raise FileNotFoundError(f"No data parquet files found under {data_dir(root)}")

    return sorted(files, key=parquet_sort_key)


def get_episode_metadata_files(root: Path):
    files = find_all_parquets(episode_metadata_dir(root))

    if not files:
        raise FileNotFoundError(
            f"No episode metadata parquet files found under {episode_metadata_dir(root)}"
        )

    return sorted(files, key=parquet_sort_key)


# ---------------------------------------------------------------------
# Dataset helpers
# ---------------------------------------------------------------------

def get_episode_ranges(ds):
    if hasattr(ds, "episode_data_index"):
        edi = ds.episode_data_index

        if "from" in edi and "to" in edi:
            return [
                (int(edi["from"][ep]), int(edi["to"][ep]))
                for ep in range(len(edi["from"]))
            ]

        if "from_idx" in edi and "to_idx" in edi:
            return [
                (int(edi["from_idx"][ep]), int(edi["to_idx"][ep]))
                for ep in range(len(edi["from_idx"]))
            ]

    episode_to_indices = defaultdict(list)

    for i in range(len(ds)):
        item = ds[i]

        if "episode_index" not in item:
            raise RuntimeError("Cannot find episode_index in dataset frames.")

        ep = item["episode_index"]

        if isinstance(ep, torch.Tensor):
            ep = ep.item()

        episode_to_indices[int(ep)].append(i)

    return [
        (min(indices), max(indices) + 1)
        for ep, indices in sorted(episode_to_indices.items())
    ]


def read_parquets(files):
    if not files:
        raise FileNotFoundError("No parquet files found.")

    dfs = []

    for p in files:
        dfs.append(pd.read_parquet(p))

    return pd.concat(dfs, ignore_index=True)


# ---------------------------------------------------------------------
# Episode detection inside parquet
# ---------------------------------------------------------------------

def get_episode_indices_in_parquet(path: Path):
    df = pd.read_parquet(path, columns=["episode_index"])

    if "episode_index" not in df.columns:
        raise RuntimeError(f"{path} has no episode_index column.")

    return sorted(set(int(x) for x in df["episode_index"].tolist()))


def data_file_episode_map(root: Path):
    files = get_data_parquet_files(root)
    mapping = {}

    print("Inspecting data parquet episode layout:")

    for p in files:
        eps = get_episode_indices_in_parquet(p)
        mapping[p] = eps
        print(f"  {p}: episodes={eps}")

    return mapping


def data_is_one_file_per_episode(root: Path, num_episodes: int):
    """
    True if every episode is stored in exactly one parquet file.
    Works for:
      data/chunk-000/file-000.parquet
      data/chunk-000/file-001.parquet
      ...
    """
    mapping = data_file_episode_map(root)

    episode_to_files = defaultdict(list)

    for p, eps in mapping.items():
        for ep in eps:
            episode_to_files[ep].append(p)

    for ep in range(num_episodes):
        if ep not in episode_to_files:
            print(f"Episode {ep} is not covered by any data parquet.")
            return False

        if len(episode_to_files[ep]) != 1:
            print(f"Episode {ep} appears in multiple data parquets: {episode_to_files[ep]}")
            return False

    for p, eps in mapping.items():
        relevant = [ep for ep in eps if ep < num_episodes]
        if len(relevant) > 1:
            print(f"{p} contains multiple episodes {relevant}; treating as merged/chunked.")
            return False

    return True


# ---------------------------------------------------------------------
# Data cleanup
# ---------------------------------------------------------------------

def remove_last_episode_data_files(new_root: Path, keep_episodes: int, num_episodes: int):
    """
    Delete parquet files whose episode_index belongs only to removed episodes.
    """
    mapping = data_file_episode_map(new_root)
    removed_eps = set(range(keep_episodes, num_episodes))

    print(f"Removed episode indices: {sorted(removed_eps)}")

    for p, eps in mapping.items():
        eps_set = set(eps)

        if eps_set and eps_set.issubset(removed_eps):
            print(f"Deleting removed episode data parquet: {p} episodes={eps}")
            p.unlink()

        elif eps_set & removed_eps:
            raise RuntimeError(
                f"Data parquet {p} contains both kept and removed episodes: {eps}. "
                f"This is a merged/chunked file and should be trimmed by rows instead."
            )

    kept_files = get_data_parquet_files(new_root)

    print("Data parquet files after cleanup:")
    for p in kept_files:
        print(f"  {p}")

    return kept_files


def trim_merged_data_parquet(new_root: Path, keep_episodes: int):
    """
    Fallback for one merged/chunked parquet containing multiple episodes.
    """
    files = get_data_parquet_files(new_root)
    df = read_parquets(files)

    if "episode_index" not in df.columns:
        raise RuntimeError("Data parquet has no episode_index column.")

    kept = df[df["episode_index"].astype(int) < keep_episodes].copy()

    if len(kept) == 0:
        raise RuntimeError("After filtering, kept data has 0 rows.")

    if "index" in kept.columns:
        kept["index"] = list(range(len(kept)))

    if "frame_index" in kept.columns:
        kept["frame_index"] = kept.groupby("episode_index").cumcount()

    ddir = data_dir(new_root)
    shutil.rmtree(ddir)

    out_dir = ddir / "chunk-000"
    out_dir.mkdir(parents=True, exist_ok=True)

    out_file = out_dir / "file-000.parquet"
    kept.to_parquet(out_file, index=False)

    print(f"Wrote trimmed merged data parquet: {out_file}")
    print(f"Rows: {len(kept)}")

    return [out_file]


# ---------------------------------------------------------------------
# Episode metadata cleanup
# ---------------------------------------------------------------------

def trim_episode_metadata(new_root: Path, keep_episodes: int):
    files = get_episode_metadata_files(new_root)

    print("Episode metadata files before cleanup:")
    for p in files:
        print(f"  {p}")

    episodes_df = read_parquets(files)

    if len(episodes_df) < keep_episodes:
        raise RuntimeError(
            f"Episode metadata has {len(episodes_df)} rows, "
            f"but need to keep {keep_episodes}."
        )

    kept = episodes_df.iloc[:keep_episodes].copy().reset_index(drop=True)

    if "episode_index" in kept.columns:
        kept["episode_index"] = list(range(keep_episodes))

    ep_dir = episode_metadata_dir(new_root)
    shutil.rmtree(ep_dir)

    out_dir = ep_dir / "chunk-000"
    out_dir.mkdir(parents=True, exist_ok=True)

    out_file = out_dir / "file-000.parquet"
    kept.to_parquet(out_file, index=False)

    print(f"Wrote trimmed episode metadata parquet: {out_file}")
    print(kept)


# ---------------------------------------------------------------------
# Video / media cleanup
# ---------------------------------------------------------------------

def remove_removed_episode_media(new_root: Path, keep_episodes: int, num_episodes: int):
    """
    Best-effort cleanup for video/image files from removed episodes.

    Supports common LeRobot video paths such as:
      videos/chunk-000/observation.images.front/episode_000000.mp4
      videos/chunk-000/observation.images.front/file-000.mp4
      videos/...
    """
    removed_eps = list(range(keep_episodes, num_episodes))

    media_exts = {
        ".mp4", ".avi", ".mov", ".mkv",
        ".png", ".jpg", ".jpeg", ".webp",
    }

    candidate_roots = [
        new_root / "videos",
        new_root / "images",
        new_root / "media",
    ]

    patterns = []

    for ep in removed_eps:
        patterns.extend([
            f"{ep}",
            f"{ep:03d}",
            f"{ep:04d}",
            f"{ep:05d}",
            f"{ep:06d}",
            f"episode_{ep}",
            f"episode_{ep:03d}",
            f"episode_{ep:04d}",
            f"episode_{ep:05d}",
            f"episode_{ep:06d}",
            f"file-{ep}",
            f"file-{ep:03d}",
            f"file-{ep:04d}",
            f"file-{ep:05d}",
            f"file-{ep:06d}",
        ])

    for base in candidate_roots:
        if not base.exists():
            continue

        for p in sorted(base.rglob("*")):
            if not p.is_file():
                continue

            if p.suffix.lower() not in media_exts:
                continue

            name = p.stem

            if any(pattern == name for pattern in patterns):
                print(f"Deleting removed episode media file: {p}")
                p.unlink()

    # Remove empty directories after deleting media.
    for base in candidate_roots:
        if not base.exists():
            continue

        for d in sorted(base.rglob("*"), reverse=True):
            if d.is_dir():
                try:
                    d.rmdir()
                except OSError:
                    pass


# ---------------------------------------------------------------------
# info.json update
# ---------------------------------------------------------------------

def update_info_json(root: Path, num_episodes: int, num_frames: int | None):
    info_path = root / "meta" / "info.json"

    if not info_path.exists():
        print("No meta/info.json found. Skipping.")
        return

    with info_path.open("r") as f:
        info = json.load(f)

    for key in ["total_episodes", "num_episodes", "episodes"]:
        if key in info:
            print(f"Updating info.json {key}: {info[key]} -> {num_episodes}")
            info[key] = int(num_episodes)

    if num_frames is not None:
        for key in ["total_frames", "num_frames", "frames"]:
            if key in info:
                print(f"Updating info.json {key}: {info[key]} -> {num_frames}")
                info[key] = int(num_frames)

    with info_path.open("w") as f:
        json.dump(info, f, indent=2)


# ---------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------

def count_rows_in_data(root: Path):
    total = 0

    for p in get_data_parquet_files(root):
        df = pd.read_parquet(p)
        total += len(df)

    return total


def validate_dataset(root: Path, expected_episodes: int):
    print("Validating cleaned dataset...")

    print("Data parquet files:")
    data_files = get_data_parquet_files(root)
    for p in data_files:
        print(f"  {p}")

    print("Episode metadata files:")
    meta_files = get_episode_metadata_files(root)
    for p in meta_files:
        print(f"  {p}")

    for p in data_files + meta_files:
        pd.read_parquet(p)

    data_df = read_parquets(data_files)

    if "episode_index" not in data_df.columns:
        raise RuntimeError("Data does not contain episode_index.")

    remaining_eps = sorted(set(int(x) for x in data_df["episode_index"].tolist()))
    expected_eps = list(range(expected_episodes))

    print(f"Remaining episode_index values in data: {remaining_eps}")

    if remaining_eps != expected_eps:
        raise RuntimeError(
            f"Expected data episode_index values {expected_eps}, got {remaining_eps}"
        )

    ds = LeRobotDataset(REPO_ID, root=root)
    ranges = get_episode_ranges(ds)

    print(f"Loaded frames: {len(ds)}")
    print(f"Loaded episodes: {len(ranges)}")

    if len(ranges) != expected_episodes:
        raise RuntimeError(
            f"Expected {expected_episodes} episodes, got {len(ranges)}"
        )

    print("\nFeatures:")
    for k, v in ds.features.items():
        print(k, ":", v)

    image_keys = [k for k in ds.features.keys() if "image" in k.lower()]

    if not image_keys:
        raise RuntimeError(
            "No image/video feature found after cleanup. "
            "This is wrong for eval3_toy_video."
        )

    print("Image/video features found:", image_keys)
    print("Validation OK.")


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main():
    if not OLD_ROOT.exists():
        raise FileNotFoundError(f"Cannot find dataset root: {OLD_ROOT}")

    if NEW_ROOT.exists():
        print(f"Removing old temp folder: {NEW_ROOT}")
        shutil.rmtree(NEW_ROOT)

    print(f"Loading dataset from {OLD_ROOT}")
    old_ds = LeRobotDataset(REPO_ID, root=OLD_ROOT)

    episode_ranges = get_episode_ranges(old_ds)
    num_episodes = len(episode_ranges)

    print(f"Found episodes: {num_episodes}")
    print(f"Found frames: {len(old_ds)}")

    if num_episodes <= NUM_LAST_EPISODES_TO_REMOVE:
        raise RuntimeError(
            f"Dataset has only {num_episodes} episodes. "
            f"Cannot remove {NUM_LAST_EPISODES_TO_REMOVE}. "
            f"Delete whole dataset and restart instead."
        )

    keep_episodes = num_episodes - NUM_LAST_EPISODES_TO_REMOVE

    print(f"Keeping episodes: 0 to {keep_episodes - 1}")
    print(f"Removing episodes: {list(range(keep_episodes, num_episodes))}")

    backup_root = Path(
        f"data/eval3_toy_video_backup_before_delete_{time.strftime('%Y%m%d_%H%M%S')}"
    )

    print(f"Creating backup: {backup_root}")
    shutil.copytree(OLD_ROOT, backup_root)

    print(f"Copying old dataset to temp folder: {NEW_ROOT}")
    shutil.copytree(OLD_ROOT, NEW_ROOT)

    if data_is_one_file_per_episode(NEW_ROOT, num_episodes):
        print("Detected one data parquet per episode by content. Keeping original files.")
        remove_last_episode_data_files(
            NEW_ROOT,
            keep_episodes=keep_episodes,
            num_episodes=num_episodes,
        )
    else:
        print("Detected merged/chunked data parquet. Trimming rows.")
        trim_merged_data_parquet(
            NEW_ROOT,
            keep_episodes=keep_episodes,
        )

    trim_episode_metadata(
        NEW_ROOT,
        keep_episodes=keep_episodes,
    )

    remove_removed_episode_media(
        NEW_ROOT,
        keep_episodes=keep_episodes,
        num_episodes=num_episodes,
    )

    num_frames_after = count_rows_in_data(NEW_ROOT)

    update_info_json(
        NEW_ROOT,
        num_episodes=keep_episodes,
        num_frames=num_frames_after,
    )

    validate_dataset(
        NEW_ROOT,
        expected_episodes=keep_episodes,
    )

    print("Replacing old dataset with cleaned dataset...")
    shutil.rmtree(OLD_ROOT)
    shutil.move(str(NEW_ROOT), str(OLD_ROOT))

    print("=" * 80)
    print("Done. Latest bad episode removed.")
    print(f"Backup saved at: {backup_root}")
    print(f"Cleaned dataset: {OLD_ROOT}")
    print("=" * 80)


if __name__ == "__main__":
    main()