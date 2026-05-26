#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from hf_common import MODEL_DIR, has_any, print_tree

FULL_WEIGHT_PATTERNS = (
    "model.safetensors",
    "model-*.safetensors",
    "model.safetensors.index.json",
    "pytorch_model.bin",
    "pytorch_model-*.bin",
    "pytorch_model.bin.index.json",
    "*.gguf",
)


def classify_checkpoint(path: Path) -> str:
    adapter_config = path / "adapter_config.json"
    config = path / "config.json"

    if adapter_config.exists():
        print("Checkpoint type: PEFT/LoRA adapter")
        try:
            adapter = json.loads(adapter_config.read_text())
        except json.JSONDecodeError as exc:
            print(f"adapter_config.json exists but could not be parsed: {exc}")
            return "adapter"

        base = adapter.get("base_model_name_or_path")
        if base:
            print(f"Base model from adapter_config.json: {base}")
        else:
            print("adapter_config.json does not include base_model_name_or_path.")
            print("Pass --base-model to the inference script.")
        return "adapter"

    if config.exists() and has_any(path, FULL_WEIGHT_PATTERNS):
        print("Checkpoint type: full fine-tuned model")
        return "full"

    print("Checkpoint type: unclear")
    print_tree(path)
    print("\nExpected one of:")
    print("  - adapter_config.json for a PEFT/LoRA adapter")
    print("  - config.json plus model weights for a full fine-tuned model")
    print("Model weight patterns checked:")
    for pattern in FULL_WEIGHT_PATTERNS:
        print(f"  - {pattern}")
    return "unclear"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, default=MODEL_DIR)
    args = parser.parse_args()

    if not args.checkpoint.exists():
        raise SystemExit(f"Checkpoint path does not exist: {args.checkpoint}")

    classify_checkpoint(args.checkpoint)


if __name__ == "__main__":
    main()
