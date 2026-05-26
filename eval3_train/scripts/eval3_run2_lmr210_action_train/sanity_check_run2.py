#!/usr/bin/env python3
from __future__ import annotations

import pandas as pd

from run2_common import (
    DATASET_DIR,
    DATASET_REPO_ID,
    EXPECTED_BASE_MODEL,
    VLM_LORA_DIR,
    configure_writable_hf_cache,
    require_vlm_lora,
)


def main() -> None:
    configure_writable_hf_cache()
    adapter = require_vlm_lora()

    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    ds = LeRobotDataset(
        repo_id=DATASET_REPO_ID,
        root=DATASET_DIR,
        video_backend="pyav",
    )

    tasks = pd.read_parquet(DATASET_DIR / "meta" / "tasks.parquet")

    print("Run 2 sanity check")
    print(f"  dataset root: {DATASET_DIR}")
    print(f"  dataset repo_id: {DATASET_REPO_ID}")
    print(f"  episodes: {ds.num_episodes}")
    print(f"  frames: {ds.num_frames}")
    print(f"  tasks: {list(tasks.index)}")
    print(f"  VLM LoRA: {VLM_LORA_DIR}")
    print(f"  VLM LoRA base: {adapter.get('base_model_name_or_path')}")
    print(f"  expected base: {EXPECTED_BASE_MODEL}")

    if ds.num_episodes != 300:
        raise SystemExit(f"ERROR: expected 300 merged episodes, got {ds.num_episodes}")
    if ds.num_frames != 67212:
        raise SystemExit(f"ERROR: expected 67212 merged frames, got {ds.num_frames}")
    if len(tasks) != 6:
        raise SystemExit(f"ERROR: expected 6 merged tasks, got {len(tasks)}")

    print("Sanity check passed.")


if __name__ == "__main__":
    main()
