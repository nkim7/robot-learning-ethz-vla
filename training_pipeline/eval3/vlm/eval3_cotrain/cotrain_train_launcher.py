#!/usr/bin/env python3
"""Cycle N action training launcher for Eval 3 cotrain.

Injects a frozen (or trainable) VLM LoRA adapter into SmolVLA before training,
using the same monkey-patch approach as eval3_run1/run2 launchers.

Key env vars
------------
EVAL3_VLM_LORA_PATH       Path to the VLM LoRA adapter directory.
                           Default: checkpoints/eval3_cotrain/cycle_05_vlm_lora
EVAL3_EXPECTED_VLM_BASE   Expected base model declared in adapter_config.json.
                           Default: HuggingFaceTB/SmolVLM2-2.2B-Instruct
EVAL3_VLM_LORA_TRAINABLE  '1' to also train LoRA weights (cotrain VLM step).
                           Default: '0' (frozen)
EVAL3_VLM_LORA_LR         LR for LoRA params when trainable. Default: 1e-6

All other config is passed via CLI args forwarded to lerobot_train.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

# Make sibling modules importable
_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from cotrain_common import configure_writable_hf_cache, require_vlm_lora  # noqa: E402
from cotrain_lora_patch import apply_run1_vlm_lora_patch as apply_vlm_lora_patch  # noqa: E402


def main() -> None:
    configure_writable_hf_cache()
    require_vlm_lora()  # fail fast before downloading anything
    apply_vlm_lora_patch()

    from lerobot.scripts.lerobot_train import main as lerobot_train_main
    lerobot_train_main()


if __name__ == "__main__":
    main()
