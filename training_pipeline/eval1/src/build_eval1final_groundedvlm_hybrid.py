#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import torch
from transformers import AutoModelForImageTextToText, AutoProcessor

from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig  # noqa: F401
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build Eval1-final SmolVLA with a grounded SmolVLM2 backbone.")
    parser.add_argument(
        "--base-policy",
        default="robot-learning-group47/eval1_clean_256_chunk50_lightingonly_smolvla_v1",
    )
    parser.add_argument(
        "--grounded-vlm",
        default="/data/models/smolvlm2-500m-eval2-grounded-v1",
        help="Local path or HF repo for the merged grounded SmolVLM2.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("/data/lerobot_outputs/policies/eval1final_groundedvlm_smolvla_init_v1"),
    )
    parser.add_argument("--hub-repo", default="robot-learning-group47/eval1final_groundedvlm_smolvla_init_v1")
    parser.add_argument("--push-to-hub", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    print(f"loading base policy {args.base_policy}")
    policy = SmolVLAPolicy.from_pretrained(args.base_policy)
    print(f"loading grounded VLM {args.grounded_vlm}")
    grounded_vlm = AutoModelForImageTextToText.from_pretrained(
        args.grounded_vlm,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
    )
    grounded_processor = AutoProcessor.from_pretrained(args.grounded_vlm)

    policy.model.vlm_with_expert.vlm = grounded_vlm
    policy.model.vlm_with_expert.processor = grounded_processor
    policy.model.vlm_with_expert.config = grounded_vlm.config
    policy.config.vlm_model_name = str(args.grounded_vlm)
    policy.config.load_vlm_weights = True
    policy.config.freeze_vision_encoder = True
    policy.config.train_expert_only = True

    policy.model.vlm_with_expert.set_requires_grad()
    policy.model.set_requires_grad()

    print(f"saving hybrid policy to {args.out}")
    policy.save_pretrained(args.out)

    if args.push_to_hub:
        print(f"pushing to {args.hub_repo}")
        policy.push_to_hub(args.hub_repo)


if __name__ == "__main__":
    main()
