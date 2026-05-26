#!/usr/bin/env python3
"""Training launcher for SmolVLA bridge training (celebrity→position E2E).

Applies in order:
  1. run1_lora_patch   — loads VLM LoRA frozen (EVAL3_VLM_LORA_TRAINABLE=0)
  2. bridge_loss_patch — adds spatial_head + align/spatial bridge losses
  3. lerobot_train_main — standard LeRobot training loop
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

_this_dir = Path(__file__).resolve().parent
_root_dir = _this_dir.parents[1]
_run1_dir = _root_dir / "scripts" / "eval3_run1_action_train"

for _d in [str(_this_dir), str(_run1_dir)]:
    if _d not in sys.path:
        sys.path.insert(0, _d)


def main() -> None:
    os.environ.setdefault("HF_HOME", "/tmp/hf-home")
    os.environ.setdefault("HF_DATASETS_CACHE", "/tmp/hf-datasets-cache")
    os.environ.setdefault("HF_HUB_CACHE", "/tmp/hf-home/hub")
    os.environ.setdefault("EVAL3_RUN1_PATCH_ACTIVE", "1")
    # Ensure VLM LoRA is loaded frozen for bridge training
    os.environ.setdefault("EVAL3_VLM_LORA_TRAINABLE", "0")

    from run1_lora_patch import apply_run1_vlm_lora_patch
    apply_run1_vlm_lora_patch()

    from bridge_loss_patch import apply_bridge_loss_patch
    apply_bridge_loss_patch()

    from lerobot.scripts.lerobot_train import main as lerobot_train_main
    lerobot_train_main()


if __name__ == "__main__":
    main()
