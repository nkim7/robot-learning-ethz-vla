#!/usr/bin/env python3
from __future__ import annotations

import os

from run1_common import configure_writable_hf_cache
from run1_lora_patch import apply_run1_vlm_lora_patch


def main() -> None:
    configure_writable_hf_cache()
    os.environ.setdefault("EVAL3_RUN1_PATCH_ACTIVE", "1")
    apply_run1_vlm_lora_patch()

    from lerobot.scripts.lerobot_train import main as lerobot_train_main

    lerobot_train_main()


if __name__ == "__main__":
    main()
