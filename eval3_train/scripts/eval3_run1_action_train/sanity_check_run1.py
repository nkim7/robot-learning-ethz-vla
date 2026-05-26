#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import torch

from run1_common import (
    DATASET_DIR,
    DATASET_REPO_ID,
    EXPECTED_BASE_MODEL,
    VLM_LORA_DIR,
    configure_writable_hf_cache,
    die,
    lerobot_train_executable,
    require_vlm_lora,
)


def print_cuda() -> None:
    print("CUDA")
    print(f"  available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        idx = torch.cuda.current_device()
        props = torch.cuda.get_device_properties(idx)
        print(f"  device: cuda:{idx}")
        print(f"  name: {props.name}")
        print(f"  total memory GiB: {props.total_memory / 1024**3:.2f}")
    else:
        print("  GPU unavailable in this process. Training must be launched where cuda is visible.")
        die("CUDA is required for Run 1 training but is not available in this process.")


def print_dataset() -> None:
    print("\nDataset")
    if not DATASET_DIR.exists():
        die(f"Local dataset missing: {DATASET_DIR}")

    info_path = DATASET_DIR / "meta" / "info.json"
    tasks_path = DATASET_DIR / "meta" / "tasks.parquet"
    if not info_path.exists():
        die(f"Missing dataset metadata: {info_path}")
    if not tasks_path.exists():
        die(f"Missing dataset tasks file: {tasks_path}")

    info = json.loads(info_path.read_text())
    features = info.get("features", {})
    print(f"  local path: {DATASET_DIR}")
    print(f"  repo id: {DATASET_REPO_ID}")
    print(f"  total episodes: {info.get('total_episodes')}")
    print(f"  total frames/samples: {info.get('total_frames')}")
    print(f"  feature keys: {list(features.keys())}")
    print(f"  action shape: {features.get('action', {}).get('shape')}")
    print(f"  observation.state shape: {features.get('observation.state', {}).get('shape')}")
    image_keys = [k for k, v in features.items() if v.get("dtype") in {"image", "video"}]
    print(f"  image keys: {image_keys}")

    tasks = pd.read_parquet(tasks_path)
    print("  task strings:")
    for task in list(tasks.index[:8]):
        print(f"    - {task}")

    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    ds = LeRobotDataset(
        repo_id=DATASET_REPO_ID,
        root=DATASET_DIR,
        video_backend="pyav",
    )
    sample = ds[0]
    print(f"  LeRobotDataset frames: {ds.num_frames}")
    print(f"  LeRobotDataset episodes: {ds.num_episodes}")
    print("  sample keys/shapes:")
    for key, value in sample.items():
        shape = getattr(value, "shape", None)
        print(f"    {key}: {shape if shape is not None else type(value).__name__}")


def print_lora() -> None:
    print("\nFrozen VLM LoRA")
    adapter = require_vlm_lora()
    print(f"  checkpoint: {VLM_LORA_DIR}")
    print("  adapter_config.json: present")
    print(f"  adapter base model: {adapter.get('base_model_name_or_path')}")
    print(f"  expected base model: {EXPECTED_BASE_MODEL}")


def print_base_cache_hint() -> None:
    print("\nBase model cache")
    local_candidates = [
        Path("checkpoint/base_models/HuggingFaceTB__SmolVLM2-2.2B-Instruct"),
        Path("/tmp/hf-home/hub"),
        Path.home() / ".cache" / "huggingface" / "hub",
    ]
    hits = [p for p in local_candidates if p.exists()]
    if hits:
        for hit in hits:
            print(f"  found cache/path candidate: {hit}")
    else:
        print("  no local base model cache candidate found; it will be downloaded on first model load")


def print_lerobot_cli() -> None:
    print("\nLeRobot CLI")
    exe = lerobot_train_executable()
    if not exe:
        die("lerobot-train CLI is not available in PATH")
    print(f"  lerobot-train: {exe}")
    try:
        from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig
        from lerobot.scripts import lerobot_train

        print(f"  SmolVLA config: {SmolVLAConfig.__name__}")
        print(f"  train module: {lerobot_train.__file__}")
    except Exception as exc:
        die(f"Could not import LeRobot SmolVLA training modules: {exc}")


def main() -> None:
    configure_writable_hf_cache()
    print_cuda()
    print_dataset()
    print_lora()
    print_base_cache_hint()
    print_lerobot_cli()
    print("\nSanity check passed.")


if __name__ == "__main__":
    main()
