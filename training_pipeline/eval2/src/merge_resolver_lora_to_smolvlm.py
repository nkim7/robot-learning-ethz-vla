#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForImageTextToText, AutoProcessor


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Merge the Eval2 resolver LoRA into SmolVLM2.")
    parser.add_argument("--base-id", default="HuggingFaceTB/SmolVLM2-500M-Video-Instruct")
    parser.add_argument(
        "--adapter",
        type=Path,
        default=Path("/data/resolver_outputs/eval2_smolvlm_lora_layout_temporal_aug_v2/best"),
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("/data/models/smolvlm2-500m-eval2-grounded-v1"),
    )
    parser.add_argument("--hub-repo", default="robot-learning-group47/smolvlm2-500m-eval2-grounded-v1")
    parser.add_argument("--push-to-hub", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    print(f"loading base {args.base_id}")
    model = AutoModelForImageTextToText.from_pretrained(
        args.base_id,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
    )
    print(f"loading adapter {args.adapter}")
    model = PeftModel.from_pretrained(model, str(args.adapter))
    print("merging")
    merged = model.merge_and_unload()
    print(f"saving to {args.out}")
    merged.save_pretrained(args.out, safe_serialization=True)
    AutoProcessor.from_pretrained(str(args.adapter)).save_pretrained(args.out)

    if args.push_to_hub:
        print(f"pushing to {args.hub_repo}")
        merged.push_to_hub(args.hub_repo)
        AutoProcessor.from_pretrained(str(args.adapter)).push_to_hub(args.hub_repo)


if __name__ == "__main__":
    main()
