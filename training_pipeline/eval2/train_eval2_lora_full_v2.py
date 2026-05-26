#!/usr/bin/env python3
"""Ablation: ordinary full-trajectory BC with LoRA text pathway open.

Run name: eval2_strict10_lora_expert_full_v2

This is a direct BC baseline with:
  - strict10 backbone + eval1 expert init
  - LoRA r=16 on text q/v projections (text pathway open)
  - All action projections + expert trainable
  - Target-slot aux head (for diagnostic comparison with counterfactual run)
  - Standard uniform sampling from full dataset (no counterfactual tricks)

Stop criterion: probe at 6k; if slot_acc ≈ 100% but action not prompt-conditioned,
the ablation has confirmed state leakage → stop early.
"""

from __future__ import annotations

import argparse
import random
import shutil
import time
from pathlib import Path

import pandas as pd
import torch
import torch.nn.functional as F
from peft import LoraConfig, get_peft_model
from torch import nn
from torch.utils.data import DataLoader, Dataset, Sampler

from lerobot.datasets.factory import resolve_delta_timestamps
from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
from lerobot.policies.factory import make_pre_post_processors
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
from lerobot.utils.constants import ACTION

SLOTS = ("left", "middle", "right")
SLOT_TO_ID = {s: i for i, s in enumerate(SLOTS)}


class FullTrajectoryDataset(Dataset):
    """Full-trajectory BC dataset with per-episode slot labels for aux loss."""

    def __init__(self, base: LeRobotDataset, target_df: pd.DataFrame):
        self.base = base
        self.ep_slot: dict[int, int] = {}
        for _, row in target_df.drop_duplicates("episode_index").iterrows():
            self.ep_slot[int(row["episode_index"])] = SLOT_TO_ID[str(row["target_slot"])]

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, idx: int) -> dict:
        item = dict(self.base[idx])
        ep = int(item["episode_index"])
        item["target_slot_id"] = torch.tensor(self.ep_slot.get(ep, 0), dtype=torch.long)
        return item


def collate_with_slot(items: list[dict]) -> dict:
    slot_ids = torch.stack([item.pop("target_slot_id") for item in items])
    batch = torch.utils.data.default_collate(items)
    batch["target_slot_id"] = slot_ids
    return batch


def add_slot_head_hook(policy, slot_head: nn.Module, storage: dict):
    raw = policy.get_base_model() if hasattr(policy, "get_base_model") else policy
    action_out = raw.model.action_out_proj

    def hook(_module, inputs, _output):
        storage["slot_logits"] = slot_head(inputs[0].float().mean(dim=1))

    return action_out.register_forward_hook(hook)


def copy_processors(src: Path, dst: Path) -> None:
    for name in [
        "policy_preprocessor.json",
        "policy_preprocessor_step_5_normalizer_processor.safetensors",
        "policy_postprocessor.json",
        "policy_postprocessor_step_0_unnormalizer_processor.safetensors",
        "config.json",
    ]:
        if (src / name).exists():
            shutil.copy(src / name, dst / name)


def save_checkpoint(out_dir: Path, step: int, policy, slot_head: nn.Module, init_path: Path) -> Path:
    ckpt = out_dir / "checkpoints" / f"{step:06d}" / "pretrained_model"
    ckpt.mkdir(parents=True, exist_ok=True)
    policy.save_pretrained(ckpt)
    policy.config.save_pretrained(ckpt)
    copy_processors(init_path, ckpt)
    torch.save(slot_head.state_dict(), ckpt / "target_slot_head.pt")
    return ckpt


def main() -> None:
    parser = argparse.ArgumentParser(description="Full-trajectory BC ablation with LoRA open")
    parser.add_argument("--init", default="/home/shadeform/outputs/smolvla-eval2-strict10-eval1-init")
    parser.add_argument("--dataset", default="robot-learning-group47/eval2_fixed_bowl_fixed_banana_108")
    parser.add_argument("--dataset-root", default="/home/shadeform/.cache/lerobot/robot-learning-group47/eval2_fixed_bowl_fixed_banana_108")
    parser.add_argument("--target-index", default="/home/shadeform/data/eval2_fixed_target_branching/target_branching_index.csv")
    parser.add_argument("--out-dir", default="/home/shadeform/outputs/eval2_strict10_lora_expert_full_v2")
    parser.add_argument("--steps", type=int, default=10000, help="Stop at 6k if probe collapsed")
    parser.add_argument("--batch-size", type=int, default=24)
    parser.add_argument("--lambda-slot", type=float, default=0.075)
    parser.add_argument("--lr", type=float, default=5e-6)
    parser.add_argument("--save-every", type=int, default=2000)
    parser.add_argument("--log-every", type=int, default=25)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--wandb-project", default="lerobot")
    parser.add_argument("--wandb-name", default="eval2_strict10_lora_expert_full_v2")
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    init_path = Path(args.init)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    wandb_run = None
    if args.wandb:
        import wandb
        wandb_run = wandb.init(project=args.wandb_project, name=args.wandb_name, config=vars(args))

    print("Loading policy...", flush=True)
    policy = SmolVLAPolicy.from_pretrained(args.init)
    policy.config.device = device

    ds_meta = LeRobotDatasetMetadata(args.dataset, root=args.dataset_root)
    delta_ts = resolve_delta_timestamps(policy.config, ds_meta)
    base_ds = LeRobotDataset(
        args.dataset, root=args.dataset_root,
        delta_timestamps=delta_ts, video_backend="pyav", tolerance_s=1e-4,
    )
    target_df = pd.read_csv(args.target_index)
    dataset = FullTrajectoryDataset(base_ds, target_df)

    loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=4, collate_fn=collate_with_slot, drop_last=True,
    )

    processor_kwargs = {
        "preprocessor_overrides": {
            "device_processor": {"device": device},
            "normalizer_processor": {
                "stats": base_ds.meta.stats,
                "features": {**policy.config.input_features, **policy.config.output_features},
                "norm_map": policy.config.normalization_mapping,
            },
            "rename_observations_processor": {
                "rename_map": {"observation.images.front": "observation.images.camera1"}
            },
        },
        "postprocessor_overrides": {
            "unnormalizer_processor": {
                "stats": base_ds.meta.stats,
                "features": policy.config.output_features,
                "norm_map": policy.config.normalization_mapping,
            },
        },
    }
    preprocessor, _ = make_pre_post_processors(policy.config, pretrained_path=args.init, **processor_kwargs)

    lora_cfg = LoraConfig(
        r=16,
        target_modules=r"model\.vlm_with_expert\.vlm\.model\.text_model\.layers\.[0-9]+\.self_attn\.(q|v)_proj",
        modules_to_save=[
            "lm_expert", "state_proj", "action_in_proj",
            "action_out_proj", "action_time_mlp_in", "action_time_mlp_out",
        ],
        bias="none",
    )
    policy = get_peft_model(policy, lora_cfg).to(device)
    raw = policy.get_base_model()
    slot_head = nn.Linear(raw.model.vlm_with_expert.expert_hidden_size, 3).to(device)
    storage: dict = {}
    hook_handle = add_slot_head_hook(policy, slot_head, storage)

    params = [p for p in policy.parameters() if p.requires_grad] + list(slot_head.parameters())
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=1e-10, betas=(0.9, 0.95), eps=1e-8)
    print(f"Trainable params: {sum(p.numel() for p in params):,}", flush=True)

    policy.train()
    slot_head.train()
    start = time.time()
    step = 0

    for epoch in range(1000):
        for batch in loader:
            step += 1
            for key in list(base_ds.meta.camera_keys):
                if key in batch and batch[key].dtype == torch.uint8:
                    batch[key] = batch[key].float() / 255.0

            target_slot_id = batch.pop("target_slot_id").to(device)
            batch = preprocessor(batch)
            storage.clear()
            action_loss, output = policy.forward(batch)

            slot_logits = storage["slot_logits"]
            slot_loss = F.cross_entropy(slot_logits, target_slot_id)
            slot_acc = (slot_logits.argmax(dim=-1) == target_slot_id).float().mean()
            loss = action_loss + args.lambda_slot * slot_loss

            opt.zero_grad(set_to_none=True)
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(params, 10.0)
            opt.step()

            if step % args.log_every == 0 or step == 1:
                elapsed = time.time() - start
                metrics = {
                    "loss": float(loss.item()),
                    "action_loss": float(action_loss.item()),
                    "slot_loss": float(slot_loss.item()),
                    "slot_acc": float(slot_acc.item()),
                    "grad_norm": float(grad_norm.item()),
                    "steps_per_s": step / max(elapsed, 1e-6),
                }
                print(
                    f"step={step} loss={metrics['loss']:.4f} "
                    f"action={metrics['action_loss']:.4f} "
                    f"slot={metrics['slot_loss']:.4f} slot_acc={metrics['slot_acc']*100:.1f}% "
                    f"grad={metrics['grad_norm']:.2f}",
                    flush=True,
                )
                if wandb_run is not None:
                    import wandb
                    wandb.log(metrics, step=step)

            if step % args.save_every == 0 or step == args.steps:
                ckpt = save_checkpoint(out_dir, step, policy, slot_head, init_path)
                print(f"saved {ckpt}", flush=True)

            if step >= args.steps:
                break
        if step >= args.steps:
            break

    hook_handle.remove()
    if wandb_run is not None:
        wandb_run.finish()
    print("Done.", flush=True)


if __name__ == "__main__":
    main()
