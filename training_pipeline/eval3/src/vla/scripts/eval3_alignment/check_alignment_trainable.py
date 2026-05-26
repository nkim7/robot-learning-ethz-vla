#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from collections import Counter
from pathlib import Path

import torch


ROOT_DIR = Path(__file__).resolve().parents[2]
RUN1_SCRIPT_DIR = ROOT_DIR / "scripts" / "eval3_run1_action_train"
RUN2_SCRIPT_DIR = ROOT_DIR / "scripts" / "eval3_run2_lmr210_action_train"


def configure_writable_hf_cache() -> None:
    os.environ.setdefault("HF_HOME", "/tmp/hf-home")
    os.environ.setdefault("HF_DATASETS_CACHE", "/tmp/hf-datasets-cache")
    os.environ.setdefault("HF_HUB_CACHE", "/tmp/hf-home/hub")
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")


def ensure_run2_patch_importable() -> None:
    for script_dir in (RUN1_SCRIPT_DIR, RUN2_SCRIPT_DIR):
        script_dir_str = str(script_dir)
        if script_dir_str not in sys.path:
            sys.path.insert(0, script_dir_str)


def group_name(name: str) -> str:
    if "vlm_with_expert.vlm" in name or re.search(r"\bvlm\.", name):
        if "lora_" in name or ".lora" in name or "adapter" in name:
            return "vlm_lora"
        return "vlm_backbone"
    if "vlm_with_expert.lm_expert" in name:
        return "action_expert"
    if "state_proj" in name:
        return "state_proj"
    if any(k in name for k in ("action_in_proj", "action_out_proj", "action_time_mlp")):
        return "action_projection"
    return name.split(".")[0]


def has_trainable(policy: torch.nn.Module, predicate) -> bool:
    return any(param.requires_grad for name, param in policy.named_parameters() if predicate(name))


def has_any(policy: torch.nn.Module, predicate) -> bool:
    return any(True for name, _ in policy.named_parameters() if predicate(name))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--action-checkpoint", type=Path, required=True)
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--dataset-repo-id", required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--lr", type=float, default=5e-6)
    args = parser.parse_args()

    configure_writable_hf_cache()
    ensure_run2_patch_importable()
    os.environ["EVAL3_VLM_LORA_PATH"] = str(args.adapter.resolve())
    os.environ["EVAL3_VLM_LORA_TRAINABLE"] = "0"
    os.environ.setdefault("EVAL3_EXPECTED_VLM_BASE", "HuggingFaceTB/SmolVLM2-2.2B-Instruct")

    from run1_lora_patch import apply_run1_vlm_lora_patch

    apply_run1_vlm_lora_patch()

    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from lerobot.policies.factory import make_policy
    from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig

    ds = LeRobotDataset(
        repo_id=args.dataset_repo_id,
        root=args.dataset_root,
        video_backend="pyav",
    )
    cfg = SmolVLAConfig(
        pretrained_path=args.action_checkpoint,
        device=args.device,
        use_amp=True,
        use_peft=True,
        push_to_hub=False,
        train_expert_only=True,
        train_state_proj=True,
        freeze_vision_encoder=True,
        optimizer_lr=args.lr,
        vlm_model_name=os.environ["EVAL3_EXPECTED_VLM_BASE"],
        load_vlm_weights=True,
        num_vlm_layers=-1,
    )

    policy = make_policy(cfg=cfg, ds_meta=ds.meta)

    total = sum(p.numel() for p in policy.parameters())
    trainable = sum(p.numel() for p in policy.parameters() if p.requires_grad)
    counts = Counter()
    trainable_names: list[str] = []
    for name, param in policy.named_parameters():
        if param.requires_grad:
            counts[group_name(name)] += param.numel()
            trainable_names.append(name)

    lora_exists = has_any(policy, lambda n: "lora_" in n or ".lora" in n or "adapter" in n)
    lora_trainable = has_trainable(policy, lambda n: "lora_" in n or ".lora" in n or "adapter" in n)
    vlm_trainable = has_trainable(
        policy, lambda n: "vlm_with_expert.vlm" in n or re.search(r"\bvlm\.", n)
    )
    expert_trainable = has_trainable(policy, lambda n: "vlm_with_expert.lm_expert" in n)
    state_trainable = has_trainable(policy, lambda n: "state_proj" in n)
    action_proj_trainable = has_trainable(
        policy, lambda n: any(k in n for k in ("action_in_proj", "action_out_proj", "action_time_mlp"))
    )

    summary = {
        "action_checkpoint": str(args.action_checkpoint),
        "adapter": str(args.adapter),
        "dataset_root": str(args.dataset_root),
        "dataset_repo_id": args.dataset_repo_id,
        "total_parameters": total,
        "trainable_parameters": trainable,
        "trainable_percentage": 100.0 * trainable / total if total else 0.0,
        "trainable_groups": dict(counts.most_common()),
        "checks": {
            "vlm_backbone_frozen": not vlm_trainable,
            "lora_adapter_present": lora_exists,
            "lora_adapter_frozen": not lora_trainable,
            "action_expert_trainable": expert_trainable,
            "state_projection_trainable": state_trainable,
            "action_projection_trainable": action_proj_trainable,
        },
        "first_trainable_names": trainable_names[:120],
    }

    print(json.dumps(summary, indent=2))
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(summary, indent=2) + "\n")

    errors = []
    if lora_trainable:
        errors.append("LoRA adapter is trainable")
    if vlm_trainable:
        errors.append("VLM/base vision parameters are trainable")
    if not lora_exists:
        errors.append("LoRA adapter was not loaded")
    if not expert_trainable:
        errors.append("action expert has no trainable parameters")
    if not state_trainable:
        errors.append("state projection has no trainable parameters")
    if not action_proj_trainable:
        errors.append("action projection/output head has no trainable parameters")

    if errors:
        raise SystemExit("ERROR: " + "; ".join(errors))


if __name__ == "__main__":
    main()
