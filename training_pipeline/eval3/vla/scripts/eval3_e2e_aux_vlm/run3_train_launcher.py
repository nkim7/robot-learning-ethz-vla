#!/usr/bin/env python3
"""Training launcher for E2E co-training with auxiliary VLM Q&A loss.

Applies in order:
  1. run1_lora_patch  — loads VLM LoRA (trainable when EVAL3_VLM_LORA_TRAINABLE=1)
  2. aux_vlm_loss_patch — adds Q&A cross-entropy loss to policy forward
  3. lerobot_train_main — standard LeRobot training loop
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

# Make run1 patch and this directory importable
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

    from run1_lora_patch import apply_run1_vlm_lora_patch
    apply_run1_vlm_lora_patch()

    from aux_vlm_loss_patch import apply_aux_vlm_loss_patch
    apply_aux_vlm_loss_patch()

    from lerobot.scripts.lerobot_train import main as lerobot_train_main
    lerobot_train_main()


if __name__ == "__main__":
    main()
