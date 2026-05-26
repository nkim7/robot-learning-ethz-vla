#!/usr/bin/env python3
"""Stage 2 target-routing VLA training for Eval2.

This is a single-GPU custom loop because stock `lerobot-train` cannot attach the
target-slot auxiliary loss to an action-relevant latent or use same-layout
left/middle/right grouped frame batches.

Model input remains image/state + original prompt. target_slot is used only as a
training label for the auxiliary head.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import shutil
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterator

import pandas as pd
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, Dataset, Sampler

from eval2_layout_utils import COLOR_NAMES, SLOTS, layout_id, normalize_layout, prompt_type, target_slot
from lerobot.datasets.factory import resolve_delta_timestamps
from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
from lerobot.policies.factory import make_pre_post_processors
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
from lerobot.utils.constants import ACTION


SLOT_TO_ID = {slot: i for i, slot in enumerate(SLOTS)}
FAMILY_WEIGHTS = {"ordinal": 0.35, "relative": 0.40, "negation": 0.25}


COMBOS = [
    ("bgr", "blue", 0, 5),
    ("bgr", "green", 6, 11),
    ("bgr", "red", 12, 17),
    ("brg", "blue", 18, 23),
    ("brg", "green", 24, 29),
    ("brg", "red", 30, 35),
    ("gbr", "blue", 36, 41),
    ("gbr", "green", 42, 47),
    ("gbr", "red", 48, 53),
    ("grb", "blue", 54, 59),
    ("grb", "green", 60, 65),
    ("grb", "red", 66, 71),
    ("rbg", "blue", 72, 77),
    ("rbg", "green", 78, 83),
    ("rbg", "red", 84, 89),
    ("rgb", "blue", 90, 95),
    ("rgb", "green", 96, 101),
    ("rgb", "red", 102, 107),
]


def augmented_episode_layout(episode_index: int):
    cursor = 0
    for layout_code, color, orig_start, _orig_end in COMBOS:
        prompts_per_demo = 8 if color == normalize_layout(layout_code)[1] else 6
        next_cursor = cursor + 6 * prompts_per_demo
        if cursor <= episode_index < next_cursor:
            layout = normalize_layout(layout_code)
            orig_episode = orig_start + ((episode_index - cursor) // prompts_per_demo)
            return layout, layout_id(layout), color, target_slot(layout, color), orig_episode
        cursor = next_cursor
    raise ValueError(f"Episode {episode_index} is outside the known Eval2 augmented mapping")


def task_lookup(meta) -> dict[int, str]:
    tasks = meta.tasks.reset_index()
    return {int(row["task_index"]): str(row["task"]) for _, row in tasks.iterrows()}


class Stage2RoutingDataset(Dataset):
    def __init__(self, base: LeRobotDataset, target_index: Path):
        self.base = base
        self.target_df = pd.read_csv(target_index).reset_index(drop=True)
        self.branch_n = len(self.target_df)
        self.ep_slot = {}
        for _, row in self.target_df.drop_duplicates("episode_index").iterrows():
            self.ep_slot[int(row["episode_index"])] = SLOT_TO_ID[str(row["target_slot"])]

    def __len__(self) -> int:
        return self.branch_n + len(self.base)

    def __getitem__(self, idx: int) -> dict:
        if idx < self.branch_n:
            row = self.target_df.iloc[idx]
            item = dict(self.base[int(row["global_index"])])
            item["target_slot_id"] = torch.tensor(SLOT_TO_ID[str(row["target_slot"])], dtype=torch.long)
            item["stage2_sample_type"] = "branch"
            item["stage2_layout_id"] = str(row["layout_id"])
            return item
        base_idx = idx - self.branch_n
        item = dict(self.base[base_idx])
        ep = int(item["episode_index"])
        item["target_slot_id"] = torch.tensor(self.ep_slot[ep], dtype=torch.long)
        item["stage2_sample_type"] = "full"
        item["stage2_layout_id"] = ""
        return item


class Stage2RoutingBatchSampler(Sampler[list[int]]):
    def __init__(
        self,
        dataset: Stage2RoutingDataset,
        batch_size: int,
        branch_fraction: float = 0.7,
        seed: int = 42,
        steps: int = 1000,
    ) -> None:
        if batch_size < 6:
            raise ValueError("batch_size should be at least 6")
        self.dataset = dataset
        self.batch_size = batch_size
        self.steps = steps
        self.rng = random.Random(seed)
        branch_groups = max(1, round(batch_size * branch_fraction / 3))
        self.branch_per_batch = min(batch_size, branch_groups * 3)
        self.full_per_batch = batch_size - self.branch_per_batch
        self.layout_pools: dict[str, dict[str, list[int]]] = {}
        for layout_id, grp in dataset.target_df.groupby("layout_id"):
            pools = {}
            ok = True
            for slot in SLOTS:
                rows = grp.index[grp["target_slot"] == slot].tolist()
                if not rows:
                    ok = False
                pools[slot] = rows
            if ok:
                self.layout_pools[str(layout_id)] = pools
        self.layout_ids = sorted(self.layout_pools)
        if not self.layout_ids:
            raise ValueError("No complete layout groups found")

    def __len__(self) -> int:
        return self.steps

    def __iter__(self) -> Iterator[list[int]]:
        groups_per_batch = self.branch_per_batch // 3
        for _ in range(self.steps):
            batch = []
            layouts = [self.rng.choice(self.layout_ids) for _ in range(groups_per_batch)]
            for layout in layouts:
                pools = self.layout_pools[layout]
                for slot in SLOTS:
                    batch.append(self.rng.choice(pools[slot]))
            for _ in range(self.full_per_batch):
                batch.append(self.dataset.branch_n + self.rng.randrange(len(self.dataset.base)))
            self.rng.shuffle(batch)
            yield batch[: self.batch_size]


class Stage2AugmentedDataset(Dataset):
    def __init__(self, base: LeRobotDataset):
        self.base = base
        self.task_by_idx = task_lookup(base.meta)
        episode_col = list(base.hf_dataset["episode_index"])
        task_col = list(base.hf_dataset["task_index"])

        self.ep_slot: dict[int, int] = {}
        self.ep_family: dict[int, str] = {}
        self.ep_orig: dict[int, int] = {}
        self.frame_task_idx: list[int] = [int(x) for x in task_col]
        self.pools: dict[str, dict[str, list[int]]] = {
            family: {slot: [] for slot in SLOTS} for family in FAMILY_WEIGHTS
        }
        seen_eps = set()
        family_counts = Counter()
        slot_counts = Counter()

        for idx, (ep_value, task_value) in enumerate(zip(episode_col, task_col)):
            ep = int(ep_value)
            task_idx = int(task_value)
            if ep not in seen_eps:
                _layout, _layout_name, _color, slot, orig_episode = augmented_episode_layout(ep)
                family = prompt_type(self.task_by_idx[task_idx])
                self.ep_slot[ep] = SLOT_TO_ID[slot]
                self.ep_family[ep] = family
                self.ep_orig[ep] = orig_episode
                seen_eps.add(ep)
                family_counts[family] += 1
                slot_counts[slot] += 1
            family = self.ep_family[ep]
            if family in self.pools:
                slot = SLOTS[self.ep_slot[ep]]
                self.pools[family][slot].append(idx)

        missing = [
            f"{family}/{slot}"
            for family, slot_pools in self.pools.items()
            for slot, rows in slot_pools.items()
            if not rows
        ]
        if missing:
            raise ValueError(f"Missing balanced sampling pools: {missing}")

        print(f"Augmented episode families: {dict(family_counts)}")
        print(f"Augmented target slots:      {dict(slot_counts)}")
        self.family_counts = family_counts
        self.slot_counts = slot_counts

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, idx: int) -> dict:
        item = dict(self.base[idx])
        ep = int(item["episode_index"])
        item["target_slot_id"] = torch.tensor(self.ep_slot[ep], dtype=torch.long)
        item["stage2_sample_type"] = self.ep_family[ep]
        item["stage2_layout_id"] = ""
        return item

    def describe_index(self, idx: int) -> dict[str, str | int]:
        ep = int(self.base.hf_dataset[idx]["episode_index"])
        slot = SLOTS[self.ep_slot[ep]]
        return {
            "task": self.task_by_idx[self.frame_task_idx[idx]],
            "family": self.ep_family[ep],
            "target_position": slot,
            "episode_index": ep,
            "orig_episode_index": self.ep_orig[ep],
        }


class BalancedOfficialPromptBatchSampler(Sampler[list[int]]):
    def __init__(
        self,
        dataset: Stage2AugmentedDataset,
        batch_size: int,
        steps: int,
        seed: int = 42,
        family_weights: dict[str, float] | None = None,
    ) -> None:
        if batch_size % len(SLOTS) != 0:
            raise ValueError(f"batch_size must be divisible by {len(SLOTS)}, got {batch_size}")
        self.dataset = dataset
        self.batch_size = batch_size
        self.steps = steps
        self.rng = random.Random(seed)
        weights = family_weights or FAMILY_WEIGHTS
        self.families = list(weights)
        total = sum(weights.values())
        self.weights = [weights[f] / total for f in self.families]

    def __len__(self) -> int:
        return self.steps

    def __iter__(self) -> Iterator[list[int]]:
        per_slot = self.batch_size // len(SLOTS)
        for _ in range(self.steps):
            batch = []
            for slot in SLOTS:
                for _ in range(per_slot):
                    family = self.rng.choices(self.families, weights=self.weights, k=1)[0]
                    batch.append(self.rng.choice(self.dataset.pools[family][slot]))
            self.rng.shuffle(batch)
            yield batch


def collate_with_metadata(items: list[dict]) -> dict:
    meta = {
        "target_slot_id": torch.stack([item.pop("target_slot_id") for item in items]),
        "stage2_sample_type": [item.pop("stage2_sample_type") for item in items],
        "stage2_layout_id": [item.pop("stage2_layout_id") for item in items],
    }
    batch = torch.utils.data.default_collate(items)
    batch.update(meta)
    return batch


def add_slot_head_hook(policy, slot_head: nn.Module, storage: dict):
    raw = policy.get_base_model() if hasattr(policy, "get_base_model") else policy
    action_out = raw.model.action_out_proj

    def hook(_module, inputs, _output):
        suffix_out = inputs[0].float()
        storage["slot_logits"] = slot_head(suffix_out.mean(dim=1))

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


def save_stage2_checkpoint(out_dir: Path, step: int, policy, slot_head: nn.Module, init_path: Path) -> Path:
    ckpt = out_dir / "checkpoints" / f"{step:06d}" / "pretrained_model"
    ckpt.mkdir(parents=True, exist_ok=True)
    policy.save_pretrained(ckpt)
    policy.config.save_pretrained(ckpt)
    copy_processors(init_path, ckpt)
    torch.save(slot_head.state_dict(), ckpt / "target_slot_head.pt")
    return ckpt


def set_stage2_trainable(policy) -> None:
    policy.model.vlm_with_expert.train_expert_only = True
    policy.model.vlm_with_expert.freeze_vision_encoder = True
    policy.model.vlm_with_expert.set_requires_grad()
    if hasattr(policy, "set_requires_grad"):
        policy.set_requires_grad()
    elif hasattr(policy.model, "state_proj"):
        for params in policy.model.state_proj.parameters():
            params.requires_grad = True
    for params in policy.model.vlm_with_expert.get_vlm_model().connector.parameters():
        params.requires_grad = True


def print_dry_inspection(args, policy, dataset, sampler, out_dir: Path) -> None:
    trainable = [(name, p.numel()) for name, p in policy.named_parameters() if p.requires_grad]
    trainable_names = [name for name, _ in trainable]
    frozen_vlm = all(
        not p.requires_grad
        for name, p in policy.named_parameters()
        if "vlm_with_expert.vlm" in name and "connector" not in name
    )
    frozen_lora = all(
        not p.requires_grad for name, p in policy.named_parameters() if "lora" in name.lower()
    )
    frozen_vision = all(
        not p.requires_grad for name, p in policy.named_parameters() if "vision_model" in name
    )
    connector_trainable = any("connector" in name for name in trainable_names)
    expert_trainable = any("lm_expert" in name for name in trainable_names)
    state_action_trainable = any(
        key in name
        for name in trainable_names
        for key in ("state_proj", "action_in_proj", "action_out_proj", "action_time_mlp")
    )

    print("\n=== Stage 2 Dry Inspection ===")
    print(f"STAGE1_CKPT: {args.init}")
    print(f"--policy.path: {args.init}")
    print("Full Stage 1 policy load: SmolVLAPolicy.from_pretrained(--policy.path), not fresh SmolVLA")
    print(f"Output directory: {out_dir}")
    print(f"Output directory exists: {out_dir.exists()}")
    print(f"Output overlaps Stage 1: {str(out_dir).startswith('/home/shadeform/outputs/eval2_stage1_position_bootstrap')}")
    print("\nFrozen checks:")
    print(f"  VLM backbone: {frozen_vlm}")
    print(f"  VLM LoRA:     {frozen_lora}")
    print(f"  vision encoder: {frozen_vision}")
    print("\nTrainable checks:")
    print(f"  connector / projection / adapter: {connector_trainable}")
    print(f"  action expert intended/trainable: {expert_trainable}")
    print(f"  state/action projections:         {state_action_trainable}")
    print(f"\nTrainable parameter tensors ({sum(n for _, n in trainable):,} params):")
    for name, _count in trainable[:200]:
        print(f"  {name}")
    if len(trainable_names) > 200:
        print(f"  ... {len(trainable_names) - 200} more")

    print("\nStage 2 sampler distribution:")
    print("  ordinal: 35%")
    print("  relative: 40%")
    print("  exclusion/negation: 25%")
    print("  color_direct: 0%")
    print("\nTarget-position distribution:")
    print("  left / middle / right = balanced per batch")
    print(f"  available episode slots: {dict(dataset.slot_counts)}")

    print("\nFirst sampled batches:")
    for batch_i, batch in zip(range(3), iter(sampler)):
        counts = Counter()
        print(f"  Batch {batch_i}:")
        for idx in batch[:9]:
            row = dataset.describe_index(idx)
            counts[(row["family"], row["target_position"])] += 1
            print(
                f"    family={row['family']:<8s} target={row['target_position']:<6s} "
                f"orig_ep={row['orig_episode_index']:<3d} task={row['task']}"
            )
        print(f"    shown_counts={dict(counts)}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--init", "--policy.path", dest="init", default="/home/shadeform/outputs/eval2_stage1_position_bootstrap/checkpoints/015000/pretrained_model")
    parser.add_argument("--dataset", default="robot-learning-group47/eval2_compositional_augmented")
    parser.add_argument("--dataset-root", default="/home/shadeform/lerobot_cache/robot-learning-group47/eval2_compositional_augmented")
    parser.add_argument("--target-index", default="/home/shadeform/data/eval2_fixed_target_branching/target_branching_index.csv")
    parser.add_argument("--out-dir", default="/home/shadeform/outputs/eval2_stage2_compositional_015000_balanced_r2")
    parser.add_argument("--steps", type=int, default=40000)
    parser.add_argument("--batch-size", type=int, default=30)
    parser.add_argument("--branch-fraction", type=float, default=0.7)
    parser.add_argument("--lambda-slot", type=float, default=0.1)
    parser.add_argument("--lr", type=float, default=3e-5)
    parser.add_argument("--save-every", type=int, default=5000)
    parser.add_argument("--log-every", type=int, default=25)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--balanced-augmented", action="store_true")
    parser.add_argument("--dry-inspect", action="store_true")
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--wandb-project", default="lerobot")
    parser.add_argument("--wandb-name", default="smolvla-eval2-stage2-target-routing")
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    init_path = Path(args.init)
    out_dir = Path(args.out_dir)
    if not args.dry_inspect:
        out_dir.mkdir(parents=True, exist_ok=True)
    wandb_run = None
    if args.wandb:
        import wandb

        wandb_run = wandb.init(project=args.wandb_project, name=args.wandb_name, config=vars(args))

    print("Loading dataset/meta")
    policy = SmolVLAPolicy.from_pretrained(args.init)
    policy.config.device = device
    policy.config.train_expert_only = True
    policy.config.freeze_vision_encoder = True
    policy.config.train_state_proj = True
    ds_meta = LeRobotDatasetMetadata(args.dataset, root=args.dataset_root)
    delta_ts = resolve_delta_timestamps(policy.config, ds_meta)
    base_ds = LeRobotDataset(
        args.dataset,
        root=args.dataset_root,
        delta_timestamps=delta_ts,
        video_backend="pyav",
        tolerance_s=1e-4,
    )
    if args.balanced_augmented:
        dataset = Stage2AugmentedDataset(base_ds)
        sampler = BalancedOfficialPromptBatchSampler(
            dataset,
            batch_size=args.batch_size,
            steps=args.steps,
            seed=args.seed,
        )
    else:
        dataset = Stage2RoutingDataset(base_ds, Path(args.target_index))
        sampler = Stage2RoutingBatchSampler(
            dataset,
            batch_size=args.batch_size,
            branch_fraction=args.branch_fraction,
            steps=args.steps,
            seed=args.seed,
        )
    loader = DataLoader(dataset, batch_sampler=sampler, num_workers=4, collate_fn=collate_with_metadata)

    processor_kwargs = {
        "preprocessor_overrides": {
            "device_processor": {"device": device},
            "normalizer_processor": {
                "stats": base_ds.meta.stats,
                "features": {**policy.config.input_features, **policy.config.output_features},
                "norm_map": policy.config.normalization_mapping,
            },
            "rename_observations_processor": {"rename_map": {}},
        },
        "postprocessor_overrides": {
            "unnormalizer_processor": {
                "stats": base_ds.meta.stats,
                "features": policy.config.output_features,
                "norm_map": policy.config.normalization_mapping,
            },
        },
    }
    preprocessor, _postprocessor = make_pre_post_processors(
        policy.config,
        pretrained_path=args.init,
        **processor_kwargs,
    )

    set_stage2_trainable(policy)
    if args.dry_inspect:
        if not args.balanced_augmented:
            raise ValueError("--dry-inspect currently expects --balanced-augmented")
        print_dry_inspection(args, policy, dataset, sampler, out_dir)
        return
    policy = policy.to(device)
    raw = policy
    slot_head = nn.Linear(raw.model.vlm_with_expert.expert_hidden_size, 3).to(device)
    storage: dict = {}
    hook_handle = add_slot_head_hook(policy, slot_head, storage)

    params = [p for p in policy.parameters() if p.requires_grad] + list(slot_head.parameters())
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=1e-10, betas=(0.9, 0.95), eps=1e-8)
    total_trainable = sum(p.numel() for p in params)
    print(f"Trainable params incl slot head: {total_trainable:,}")
    if args.balanced_augmented:
        print("Batch composition: official families ordinal=35% relative=40% negation=25%; slots balanced 1:1:1")
    else:
        print(f"Batch composition: branch={sampler.branch_per_batch} full={sampler.full_per_batch}")

    policy.train()
    slot_head.train()
    start = time.time()
    for step, batch in enumerate(loader, start=1):
        for key in list(base_ds.meta.camera_keys):
            if key in batch and batch[key].dtype == torch.uint8:
                batch[key] = batch[key].float() / 255.0
        target_slot_id = batch.pop("target_slot_id").to(device)
        sample_types = batch.pop("stage2_sample_type")
        batch.pop("stage2_layout_id")
        batch = preprocessor(batch)
        storage.clear()
        action_loss, output = policy.forward(batch)
        if "slot_logits" not in storage:
            raise RuntimeError("Slot hook did not fire; action_out_proj latent was not captured")
        slot_logits = storage["slot_logits"]
        slot_loss = F.cross_entropy(slot_logits, target_slot_id)
        slot_acc = (slot_logits.argmax(dim=-1) == target_slot_id).float().mean()
        loss = action_loss + args.lambda_slot * slot_loss
        opt.zero_grad(set_to_none=True)
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(params, 10.0)
        opt.step()

        if step % args.log_every == 0 or step == 1:
            branch_count = sum(1 for x in sample_types if x == "branch")
            elapsed = time.time() - start
            metrics = {
                "loss": float(loss.item()),
                "action_loss": float(action_loss.item()),
                "slot_loss": float(slot_loss.item()),
                "slot_acc": float(slot_acc.item()),
                "grad_norm": float(grad_norm.item()),
                "official_count": sum(1 for x in sample_types if x in FAMILY_WEIGHTS),
                "other_count": sum(1 for x in sample_types if x not in FAMILY_WEIGHTS),
                "steps_per_s": step / max(elapsed, 1e-6),
            }
            print(
                f"step={step} loss={metrics['loss']:.4f} action={metrics['action_loss']:.4f} "
                f"slot={metrics['slot_loss']:.4f} slot_acc={metrics['slot_acc']*100:.1f}% "
                f"official/other={metrics['official_count']}/{metrics['other_count']} "
                f"grad={metrics['grad_norm']:.2f}",
                flush=True,
            )
            if wandb_run is not None:
                import wandb

                wandb.log(metrics, step=step)
        if step % args.save_every == 0 or step == args.steps:
            ckpt = save_stage2_checkpoint(out_dir, step, policy, slot_head, init_path)
            print(f"saved {ckpt}", flush=True)
        if step >= args.steps:
            break
    hook_handle.remove()
    if wandb_run is not None:
        wandb_run.finish()


if __name__ == "__main__":
    main()
