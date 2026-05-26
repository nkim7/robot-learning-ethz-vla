#!/usr/bin/env python3
from __future__ import annotations

import re
from collections import Counter

import torch

from run2_common import (
    DATASET_DIR,
    DATASET_REPO_ID,
    EXPECTED_BASE_MODEL,
    configure_writable_hf_cache,
    ensure_run1_patch_importable,
)


def group_name(name: str) -> str:
    if "vlm_with_expert.vlm" in name or re.search(r"\bvlm\.", name):
        if "lora_" in name or "adapter" in name:
            return "vlm_lora"
        return "vlm_backbone"
    if "vlm_with_expert.lm_expert" in name:
        return "action_expert"
    if "state_proj" in name:
        return "state_proj"
    if "action_" in name:
        return "action_projection"
    return name.split(".")[0]


def has_trainable(policy: torch.nn.Module, predicate) -> bool:
    return any(param.requires_grad for name, param in policy.named_parameters() if predicate(name))


def has_any(policy: torch.nn.Module, predicate) -> bool:
    return any(True for name, _ in policy.named_parameters() if predicate(name))


def main() -> None:
    configure_writable_hf_cache()
    ensure_run1_patch_importable()

    from run1_lora_patch import apply_run1_vlm_lora_patch

    apply_run1_vlm_lora_patch()

    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from lerobot.policies.factory import make_policy
    from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig

    ds = LeRobotDataset(
        repo_id=DATASET_REPO_ID,
        root=DATASET_DIR,
        video_backend="pyav",
    )
    cfg = SmolVLAConfig(
        device="cuda",
        use_amp=True,
        use_peft=True,
        chunk_size=50,
        n_action_steps=50,
        n_obs_steps=1,
        tokenizer_max_length=48,
        empty_cameras=2,
        train_expert_only=True,
        train_state_proj=True,
        freeze_vision_encoder=True,
        optimizer_lr=3e-5,
        vlm_model_name=EXPECTED_BASE_MODEL,
        load_vlm_weights=True,
        num_vlm_layers=-1,
    )
    policy = make_policy(cfg=cfg, ds_meta=ds.meta)

    total = sum(p.numel() for p in policy.parameters())
    trainable = sum(p.numel() for p in policy.parameters() if p.requires_grad)
    pct = 100.0 * trainable / total if total else 0.0

    counts = Counter()
    trainable_names: list[str] = []
    for name, param in policy.named_parameters():
        if param.requires_grad:
            counts[group_name(name)] += param.numel()
            trainable_names.append(name)

    print(f"total parameters: {total:,}")
    print(f"trainable parameters: {trainable:,}")
    print(f"trainable percentage: {pct:.6f}%")
    print("trainable module groups:")
    for group, count in counts.most_common():
        print(f"  {group}: {count:,}")

    lora_exists = has_any(policy, lambda n: "lora_" in n or "adapter" in n)
    lora_trainable = has_trainable(policy, lambda n: "lora_" in n or "adapter" in n)
    vlm_trainable = has_trainable(
        policy, lambda n: "vlm_with_expert.vlm" in n or re.search(r"\bvlm\.", n)
    )
    expert_trainable = has_trainable(policy, lambda n: "vlm_with_expert.lm_expert" in n)
    state_trainable = has_trainable(policy, lambda n: "state_proj" in n)

    print("\nexplicit checks:")
    print(f"  VLM backbone frozen: {not vlm_trainable}")
    print(f"  LoRA adapter present: {lora_exists}")
    print(f"  LoRA adapter frozen: {not lora_trainable}")
    print(f"  action expert trainable: {expert_trainable}")
    print(f"  state projection trainable: {state_trainable}")

    if lora_trainable:
        raise SystemExit("ERROR: LoRA adapter is trainable; Run 2 must keep it frozen.")
    if vlm_trainable:
        raise SystemExit("ERROR: VLM parameters are trainable; Run 2 must keep VLM frozen.")
    if not expert_trainable:
        raise SystemExit("ERROR: action expert has no trainable parameters.")
    if not state_trainable:
        raise SystemExit("ERROR: state projection has no trainable parameters.")

    print("\nTrainable-parameter check passed.")


if __name__ == "__main__":
    main()
