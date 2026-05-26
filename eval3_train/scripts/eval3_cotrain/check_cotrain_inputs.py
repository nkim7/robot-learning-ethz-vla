#!/usr/bin/env python3
from __future__ import annotations

import json
import os
from pathlib import Path

from cotrain_common import (
    ACTION_LMR_DATASET_ROOT,
    ACTION_TOY_DATASET_ROOT,
    GROUNDING_DATA_DIR,
    PREPARED_ACTION_DATASET_ROOT,
    ROBOT_DATASET_ROOT,
    START_ACTION_CHECKPOINT,
    START_VLM_LORA,
    require_path,
)


def count_jsonl(path):
    with path.open() as f:
        return sum(1 for line in f if line.strip())


def main() -> None:
    start_action_checkpoint = Path(os.environ.get("START_ACTION_CHECKPOINT", START_ACTION_CHECKPOINT))
    start_vlm_lora = Path(os.environ.get("START_VLM_LORA", START_VLM_LORA))
    action_toy_dataset_root = Path(os.environ.get("ACTION_TOY_DATASET_ROOT", ACTION_TOY_DATASET_ROOT))
    action_lmr_dataset_root = Path(os.environ.get("ACTION_LMR_DATASET_ROOT", ACTION_LMR_DATASET_ROOT))
    prepared_default = Path(os.environ.get("PREPARED_ACTION_DATASET_ROOT", PREPARED_ACTION_DATASET_ROOT))
    robot_dataset_root = Path(os.environ.get("ROBOT_DATASET_ROOT", prepared_default or ROBOT_DATASET_ROOT))
    grounding_data_dir = Path(os.environ.get("GROUNDING_DATA_DIR", GROUNDING_DATA_DIR))

    require_path(start_action_checkpoint, "starting action checkpoint")
    require_path(start_vlm_lora / "adapter_model.safetensors", "starting VLM LoRA weights")
    require_path(action_toy_dataset_root / "meta" / "info.json", "TOY action dataset")
    require_path(action_lmr_dataset_root / "meta" / "info.json", "L/M/R action dataset")
    require_path(robot_dataset_root / "meta" / "info.json", "robot dataset")
    require_path(grounding_data_dir / "train.jsonl", "grounding train split")
    require_path(grounding_data_dir / "eval.jsonl", "grounding eval split")

    print(
        json.dumps(
            {
                "start_action_checkpoint": str(start_action_checkpoint),
                "start_vlm_lora": str(start_vlm_lora),
                "action_toy_dataset_root": str(action_toy_dataset_root),
                "action_lmr_dataset_root": str(action_lmr_dataset_root),
                "robot_dataset_root": str(robot_dataset_root),
                "grounding_data_dir": str(grounding_data_dir),
                "grounding_train_examples": count_jsonl(grounding_data_dir / "train.jsonl"),
                "grounding_eval_examples": count_jsonl(grounding_data_dir / "eval.jsonl"),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
