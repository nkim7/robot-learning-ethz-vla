#!/usr/bin/env python3
"""Eval2 Stage 2B: VLM-side relational repair inside SmolVLA.

This run is intentionally not connector/action-only. It starts from the Stage 1
motor checkpoint, freezes the action expert and main VLM weights, trains VLM
text q/v LoRA plus a target-slot auxiliary head, and emphasizes relative prompt
target-slot correctness.
"""

from __future__ import annotations

import argparse
import random
import shutil
import time
from collections import Counter
from pathlib import Path
from typing import Iterator

import torch
import torch.nn.functional as F
from peft import LoraConfig, get_peft_model
from torch import nn
from torch.utils.data import DataLoader, Dataset, Sampler

from eval2_layout_utils import SLOTS, layout_id, normalize_layout, prompt_type, target_slot
from lerobot.datasets.factory import resolve_delta_timestamps
from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
from lerobot.policies.factory import make_pre_post_processors
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy


SLOT_TO_ID = {slot: i for i, slot in enumerate(SLOTS)}
FAMILY_WEIGHTS = {"relative": 0.70, "ordinal": 0.15, "negation": 0.10, "color": 0.05}
DEFAULT_INIT = "/home/shadeform/outputs/eval2_stage2_compositional_015000_balanced_r2/checkpoints/020000/pretrained_model"
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
        layout = normalize_layout(layout_code)
        prompts_per_demo = 8 if color == layout[1] else 6
        next_cursor = cursor + 6 * prompts_per_demo
        if cursor <= episode_index < next_cursor:
            orig_episode = orig_start + ((episode_index - cursor) // prompts_per_demo)
            return layout, layout_id(layout), color, target_slot(layout, color), orig_episode
        cursor = next_cursor
    raise ValueError(f"Episode {episode_index} outside Eval2 augmented mapping")


def task_lookup(meta) -> dict[int, str]:
    tasks = meta.tasks.reset_index()
    return {int(row["task_index"]): str(row["task"]) for _, row in tasks.iterrows()}


class RelationalRepairDataset(Dataset):
    def __init__(self, base: LeRobotDataset):
        self.base = base
        self.task_by_idx = task_lookup(base.meta)
        episode_col = list(base.hf_dataset["episode_index"])
        task_col = [int(x) for x in base.hf_dataset["task_index"]]
        self.frame_task_idx = task_col
        self.ep_slot: dict[int, int] = {}
        self.ep_family: dict[int, str] = {}
        self.ep_layout: dict[int, str] = {}
        self.pools: dict[str, dict[str, list[int]]] = {
            family: {slot: [] for slot in SLOTS} for family in FAMILY_WEIGHTS
        }
        seen_eps = set()
        family_counts = Counter()
        layout_counts = Counter()
        for idx, (ep_value, task_idx) in enumerate(zip(episode_col, task_col)):
            ep = int(ep_value)
            if ep not in seen_eps:
                layout, lid, color, slot, _orig = augmented_episode_layout(ep)
                family = prompt_type(self.task_by_idx[task_idx])
                self.ep_slot[ep] = SLOT_TO_ID[slot]
                self.ep_family[ep] = family
                self.ep_layout[ep] = lid
                seen_eps.add(ep)
                family_counts[family] += 1
                layout_counts[lid] += 1
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
            raise ValueError(f"Missing sampling pools: {missing}")
        print(f"Episode family counts: {dict(family_counts)}")
        print(f"Episode layout counts: {dict(layout_counts)}")

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, idx: int) -> dict:
        item = dict(self.base[idx])
        ep = int(item["episode_index"])
        item["target_slot_id"] = torch.tensor(self.ep_slot[ep], dtype=torch.long)
        item["repair_family"] = self.ep_family[ep]
        item["repair_layout"] = self.ep_layout[ep]
        return item


class WeightedFamilySlotSampler(Sampler[list[int]]):
    def __init__(self, dataset: RelationalRepairDataset, batch_size: int, steps: int, seed: int = 42):
        if batch_size % len(SLOTS) != 0:
            raise ValueError(f"batch_size must be divisible by {len(SLOTS)}")
        self.dataset = dataset
        self.batch_size = batch_size
        self.steps = steps
        self.rng = random.Random(seed)
        self.families = list(FAMILY_WEIGHTS)
        total = sum(FAMILY_WEIGHTS.values())
        self.weights = [FAMILY_WEIGHTS[f] / total for f in self.families]

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


def collate(items: list[dict]) -> dict:
    meta = {
        "target_slot_id": torch.stack([item.pop("target_slot_id") for item in items]),
        "repair_family": [item.pop("repair_family") for item in items],
        "repair_layout": [item.pop("repair_layout") for item in items],
    }
    batch = torch.utils.data.default_collate(items)
    batch.update(meta)
    return batch


def add_slot_head_hook(policy, slot_head: nn.Module, storage: dict):
    raw = policy.get_base_model() if hasattr(policy, "get_base_model") else policy

    def hook(_module, inputs, _output):
        suffix_out = inputs[0].float()
        storage["slot_logits"] = slot_head(suffix_out.mean(dim=1))

    return raw.model.action_out_proj.register_forward_hook(hook)


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


def save_ckpt(out_dir: Path, step: int, policy, slot_head: nn.Module, init_path: Path) -> Path:
    ckpt = out_dir / "checkpoints" / f"{step:06d}" / "pretrained_model"
    ckpt.mkdir(parents=True, exist_ok=True)
    policy.save_pretrained(ckpt)
    policy.config.save_pretrained(ckpt)
    copy_processors(init_path, ckpt)
    torch.save(slot_head.state_dict(), ckpt / "target_slot_head.pt")
    return ckpt


def maybe_load_slot_head(slot_head: nn.Module, init_path: Path, device: str) -> None:
    slot_head_path = init_path / "target_slot_head.pt"
    if not slot_head_path.exists():
        print("No existing target_slot_head.pt found; starting slot head from random init")
        return
    slot_head.load_state_dict(torch.load(slot_head_path, map_location=device))
    print(f"Warm-started target slot head from {slot_head_path}")


def configure_trainability(policy, train_connector: bool) -> None:
    raw = policy.get_base_model() if hasattr(policy, "get_base_model") else policy
    for p in raw.parameters():
        p.requires_grad = False
    for name, p in policy.named_parameters():
        if "lora_" in name:
            p.requires_grad = True
    if train_connector:
        for p in raw.model.vlm_with_expert.get_vlm_model().connector.parameters():
            p.requires_grad = True
    raw.model.vlm_with_expert.get_vlm_model().vision_model.eval()
    raw.model.vlm_with_expert.lm_expert.eval()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--init", "--policy.path", dest="init", default=DEFAULT_INIT)
    parser.add_argument("--dataset", default="robot-learning-group47/eval2_compositional_augmented")
    parser.add_argument("--dataset-root", default="/home/shadeform/lerobot_cache/robot-learning-group47/eval2_compositional_augmented")
    parser.add_argument("--out-dir", default="/home/shadeform/outputs/eval2_stage2b_vlm_relational_repair")
    parser.add_argument("--steps", type=int, default=12000)
    parser.add_argument("--batch-size", type=int, default=30)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--lambda-slot", type=float, default=0.5)
    parser.add_argument("--save-every", type=int, default=2000)
    parser.add_argument("--log-every", type=int, default=25)
    parser.add_argument("--lora-rank", type=int, default=16)
    parser.add_argument("--train-connector", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dry-inspect", action="store_true")
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--wandb-project", default="lerobot")
    parser.add_argument("--wandb-name", default="eval2-stage2b-vlm-relational-repair")
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    init_path = Path(args.init)
    out_dir = Path(args.out_dir)
    if not args.dry_inspect:
        out_dir.mkdir(parents=True, exist_ok=True)
    wandb_run = None
    if args.wandb and not args.dry_inspect:
        import wandb
        wandb_run = wandb.init(project=args.wandb_project, name=args.wandb_name, config=vars(args))

    print(f"Loading Stage1 policy: {args.init}")
    policy = SmolVLAPolicy.from_pretrained(args.init)
    policy.config.device = device
    policy.config.train_expert_only = True
    policy.config.freeze_vision_encoder = True
    ds_meta = LeRobotDatasetMetadata(args.dataset, root=args.dataset_root)
    delta_ts = resolve_delta_timestamps(policy.config, ds_meta)
    base_ds = LeRobotDataset(args.dataset, root=args.dataset_root, delta_timestamps=delta_ts, video_backend="pyav")
    dataset = RelationalRepairDataset(base_ds)
    sampler = WeightedFamilySlotSampler(dataset, batch_size=args.batch_size, steps=args.steps, seed=args.seed)
    loader = DataLoader(dataset, batch_sampler=sampler, num_workers=4, collate_fn=collate)

    lora_cfg = LoraConfig(
        r=args.lora_rank,
        lora_alpha=args.lora_rank * 2,
        target_modules=r"model\.vlm_with_expert\.vlm\.model\.text_model\.layers\.[0-9]+\.self_attn\.(q|v)_proj",
        lora_dropout=0.05,
        bias="none",
    )
    policy = get_peft_model(policy, lora_cfg)
    configure_trainability(policy, train_connector=args.train_connector)
    policy = policy.to(device)
    raw = policy.get_base_model()
    slot_head = nn.Linear(raw.model.vlm_with_expert.expert_hidden_size, len(SLOTS)).to(device)
    maybe_load_slot_head(slot_head, init_path, device)
    storage: dict = {}
    hook = add_slot_head_hook(policy, slot_head, storage)
    params = [p for p in policy.parameters() if p.requires_grad] + list(slot_head.parameters())
    print("Train setup:")
    print(f"  init: {args.init}")
    print(f"  family weights: {FAMILY_WEIGHTS}")
    print(f"  lambda_slot: {args.lambda_slot}")
    print("  frozen: vision encoder, base VLM weights except LoRA, action expert")
    print(f"  train_connector: {args.train_connector}")
    print(f"  trainable params incl slot head: {sum(p.numel() for p in params):,}")
    if args.dry_inspect:
        for name, p in policy.named_parameters():
            if p.requires_grad:
                print(f"  trainable: {name}")
        return

    preprocessor, _postprocessor = make_pre_post_processors(
        policy.config,
        pretrained_path=args.init,
        preprocessor_overrides={
            "device_processor": {"device": device},
            "normalizer_processor": {
                "stats": base_ds.meta.stats,
                "features": {**policy.config.input_features, **policy.config.output_features},
                "norm_map": policy.config.normalization_mapping,
            },
            "rename_observations_processor": {"rename_map": {}},
        },
    )
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=1e-4, betas=(0.9, 0.95), eps=1e-8)
    policy.train()
    slot_head.train()
    start = time.time()
    for step, batch in enumerate(loader, start=1):
        for key in list(base_ds.meta.camera_keys):
            if key in batch and batch[key].dtype == torch.uint8:
                batch[key] = batch[key].float() / 255.0
        target_slot_id = batch.pop("target_slot_id").to(device)
        families = batch.pop("repair_family")
        batch.pop("repair_layout")
        batch = preprocessor(batch)
        storage.clear()
        action_loss, _output = policy.forward(batch)
        if "slot_logits" not in storage:
            raise RuntimeError("Slot head hook did not fire")
        slot_loss = F.cross_entropy(storage["slot_logits"], target_slot_id)
        slot_acc = (storage["slot_logits"].argmax(dim=-1) == target_slot_id).float().mean()
        loss = action_loss + args.lambda_slot * slot_loss
        opt.zero_grad(set_to_none=True)
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        if step % args.log_every == 0 or step == 1:
            counts = Counter(families)
            metrics = {
                "loss": float(loss.item()),
                "action_loss": float(action_loss.item()),
                "slot_loss": float(slot_loss.item()),
                "slot_acc": float(slot_acc.item()),
                "grad_norm": float(grad_norm.item()),
                "steps_per_s": step / max(time.time() - start, 1e-6),
            }
            print(
                f"step={step} loss={metrics['loss']:.4f} action={metrics['action_loss']:.4f} "
                f"slot={metrics['slot_loss']:.4f} slot_acc={metrics['slot_acc']*100:.1f}% "
                f"families={dict(counts)} grad={metrics['grad_norm']:.2f}",
                flush=True,
            )
            if wandb_run is not None:
                import wandb
                wandb.log({**metrics, **{f"batch_family/{k}": v for k, v in counts.items()}}, step=step)
        if step % args.save_every == 0 or step == args.steps:
            ckpt = save_ckpt(out_dir, step, policy, slot_head, init_path)
            print(f"saved {ckpt}", flush=True)
        if step >= args.steps:
            break
    hook.remove()
    if wandb_run is not None:
        wandb_run.finish()


if __name__ == "__main__":
    main()
