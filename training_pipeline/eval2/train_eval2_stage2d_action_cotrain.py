#!/usr/bin/env python3
"""Approach D — Eval2 Stage2-20k Relation-Heavy Action Co-Training.

Trains a LoRA + action-expert adaptation starting from the Stage2-20k full policy
checkpoint.  Loss = L_action only.  No auxiliary slot head.  No counterfactual
action-mismatched prompts.

Prompt mix per step (action-consistent only):
  relative-to-color : 60 %
  ordinal           : 15 %
  exclusion         : 10 %
  direct color      : 10 %
  position simple   :  5 %

Trainable:
  VLM q/v LoRA  (LR 1e-5)
  lm_expert     (LR 2e-5)
  state_proj, action_in_proj, action_out_proj,
  action_time_mlp_in, action_time_mlp_out  (LR 2e-5)

Frozen:
  vision encoder
  base VLM weights (except LoRA)
  connector (v1)
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import shutil
import sys
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import torch
from peft import LoraConfig, get_peft_model

sys.path.insert(0, "/home/shadeform")

from eval2_layout_utils import COLOR_NAMES, SLOTS, normalize_layout
from lerobot.datasets.factory import resolve_delta_timestamps
from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
from lerobot.policies.factory import make_pre_post_processors
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

# ──────────────────────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────────────────────

LAYOUT_CODES = ("bgr", "brg", "gbr", "grb", "rbg", "rgb")
COLORS = ("blue", "green", "red")
SLOT_NAMES = ("left", "middle", "right")
SLOT_TO_IDX = {s: i for i, s in enumerate(SLOT_NAMES)}
COLOR_FULL = {"b": "blue", "g": "green", "r": "red"}

# Prompt-family sampling weights (must sum to 1.0)
FAMILY_WEIGHTS = {
    "relative":  0.60,
    "ordinal":   0.15,
    "exclusion": 0.10,
    "direct":    0.10,
    "position":  0.05,
}
FAMILY_ORDER = list(FAMILY_WEIGHTS.keys())
FAMILY_CUMULATIVE = []
_acc = 0.0
for _fam in FAMILY_ORDER:
    _acc += FAMILY_WEIGHTS[_fam]
    FAMILY_CUMULATIVE.append(_acc)

DEFAULT_DATASET      = "robot-learning-group47/eval2_108_position_only"
DEFAULT_DATASET_ROOT = "/home/shadeform/lerobot_cache/robot-learning-group47/eval2_108_position_only"
DEFAULT_INIT         = (
    "/home/shadeform/outputs/eval2_stage2_compositional_015000_balanced_r2"
    "/checkpoints/020000/pretrained_model"
)
DEFAULT_OUT          = "/home/shadeform/outputs/eval2_stage2_20k_relation_cotrain_lora_action_v1"
COMBO_CSV            = "combo_episode_map_eval2_fixed_bowl_fixed_banana_108.csv"

# ──────────────────────────────────────────────────────────────────────────────
# Episode-level metadata
# ──────────────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class EpisodeMeta:
    episode_index: int
    layout: list        # [left_color, mid_color, right_color]
    layout_code: str    # e.g. "bgr"
    target_color: str
    target_slot: str    # "left" | "middle" | "right"
    target_slot_idx: int


def load_episode_meta(dataset_root: Path) -> dict[int, EpisodeMeta]:
    csv_path = dataset_root / COMBO_CSV
    ep_meta: dict[int, EpisodeMeta] = {}
    with csv_path.open() as f:
        for row in csv.DictReader(f):
            layout_code = row["bowl_order_left_to_right"].strip().lower()
            layout = [COLOR_FULL[ch] for ch in layout_code]
            target_color = row["target_bowl_color"].strip().lower()
            target_slot_idx = layout.index(target_color)
            target_slot = SLOT_NAMES[target_slot_idx]
            for ep in range(int(row["episode_start"]), int(row["episode_end"]) + 1):
                ep_meta[ep] = EpisodeMeta(
                    episode_index=ep,
                    layout=layout,
                    layout_code=layout_code,
                    target_color=target_color,
                    target_slot=target_slot,
                    target_slot_idx=target_slot_idx,
                )
    return ep_meta


# ──────────────────────────────────────────────────────────────────────────────
# Action-consistent prompt generation
# ──────────────────────────────────────────────────────────────────────────────

def valid_relative_prompts(meta: EpisodeMeta) -> list[str]:
    """Return all relative prompts that point to meta.target_color."""
    layout = meta.layout
    ti = meta.target_slot_idx
    prompts = []
    # "right of X" → bowl at X_idx+1
    if ti > 0:
        anchor = layout[ti - 1]
        prompts.append(
            f"Put the banana into the bowl on the right of the {anchor} bowl "
            f"from the robot perspective."
        )
    # "left of X" → bowl at X_idx-1
    if ti < 2:
        anchor = layout[ti + 1]
        prompts.append(
            f"Put the banana into the bowl on the left of the {anchor} bowl "
            f"from the robot perspective."
        )
    return prompts


def ordinal_prompt(meta: EpisodeMeta) -> str:
    ordinals = ("1st", "2nd", "3rd")
    return (
        f"Put the banana into the {ordinals[meta.target_slot_idx]} bowl "
        f"from the left from the robot perspective."
    )


def exclusion_prompt(meta: EpisodeMeta) -> str:
    excluded = [c for c in COLORS if c != meta.target_color]
    return (
        f"Put the banana into the bowl that is not {excluded[0]} "
        f"and not {excluded[1]}."
    )


def direct_color_prompt(meta: EpisodeMeta) -> str:
    return f"Put the banana into the {meta.target_color} bowl."


def position_prompt(meta: EpisodeMeta) -> str:
    return f"Pick up the banana and put it in the {meta.target_slot} bowl."


def sample_prompt(meta: EpisodeMeta, rng: random.Random) -> tuple[str, str]:
    """Sample a (family, prompt) pair that is action-consistent with meta."""
    roll = rng.random()
    family = FAMILY_ORDER[-1]
    for f, thresh in zip(FAMILY_ORDER, FAMILY_CUMULATIVE):
        if roll < thresh:
            family = f
            break

    if family == "relative":
        options = valid_relative_prompts(meta)
        if not options:
            # edge case: this target has no valid relative prompts — fall back to ordinal
            return "ordinal", ordinal_prompt(meta)
        return family, rng.choice(options)
    elif family == "ordinal":
        return family, ordinal_prompt(meta)
    elif family == "exclusion":
        return family, exclusion_prompt(meta)
    elif family == "direct":
        return family, direct_color_prompt(meta)
    else:
        return family, position_prompt(meta)


# ──────────────────────────────────────────────────────────────────────────────
# Frame pool (balanced across layout×target combos)
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class FrameRecord:
    global_idx: int     # index into LeRobotDataset
    episode_index: int
    frame_index: int
    meta: EpisodeMeta


def build_frame_pool(
    ds: LeRobotDataset,
    ep_meta: dict[int, EpisodeMeta],
    frame_min: int,
    frame_max: int,
) -> dict[str, list[FrameRecord]]:
    """Group frames by layout_code+target_color for balanced sampling."""
    ep_vals  = list(ds.hf_dataset["episode_index"])
    fr_vals  = list(ds.hf_dataset["frame_index"])
    pool: dict[str, list[FrameRecord]] = {}
    for global_idx, (ep_val, fr_val) in enumerate(zip(ep_vals, fr_vals)):
        ep  = int(ep_val)
        fr  = int(fr_val)
        if fr < frame_min or fr > frame_max:
            continue
        meta = ep_meta.get(ep)
        if meta is None:
            continue
        key = f"{meta.layout_code}_{meta.target_color[0]}"
        if key not in pool:
            pool[key] = []
        pool[key].append(FrameRecord(global_idx, ep, fr, meta))
    return pool


class BalancedActionSampler:
    """Round-robin over layout×target groups, random frame within each group."""

    def __init__(self, pool: dict[str, list[FrameRecord]], seed: int):
        self.pool = pool
        self.keys = sorted(pool.keys())
        self.ptrs = {k: 0 for k in self.keys}
        self.rng  = random.Random(seed)
        # Shuffle each group once
        for k in self.keys:
            self.rng.shuffle(self.pool[k])

    def next_batch(self, n: int) -> list[tuple[FrameRecord, str, str]]:
        """Return n (record, family, prompt) tuples."""
        out: list[tuple[FrameRecord, str, str]] = []
        key_cycle = [self.keys[i % len(self.keys)] for i in range(n)]
        # Spread evenly across groups, then shuffle order within batch
        self.rng.shuffle(key_cycle)
        for key in key_cycle:
            frames = self.pool[key]
            idx = self.ptrs[key] % len(frames)
            rec = frames[idx]
            self.ptrs[key] += 1
            family, prompt = sample_prompt(rec.meta, self.rng)
            out.append((rec, family, prompt))
        return out


# ──────────────────────────────────────────────────────────────────────────────
# Batch construction
# ──────────────────────────────────────────────────────────────────────────────

def fetch_item(ds: LeRobotDataset, rec: FrameRecord, prompt: str) -> dict:
    item = dict(ds[rec.global_idx])
    item["task"] = prompt
    item.pop("task_index", None)
    return item


def collate_batch(items: list[dict]) -> dict:
    from torch.utils.data import default_collate
    tasks = [item.pop("task") for item in items]
    batch = default_collate(items)
    batch["task"] = tasks
    return batch


# ──────────────────────────────────────────────────────────────────────────────
# Model configuration
# ──────────────────────────────────────────────────────────────────────────────

def configure_trainability(policy, raw) -> None:
    """Freeze everything, then unfreeze LoRA + action expert + projections."""
    # Freeze all base model params
    for p in raw.parameters():
        p.requires_grad = False

    # Unfreeze LoRA
    for name, p in policy.named_parameters():
        if "lora_" in name:
            p.requires_grad = True

    # Unfreeze action expert (lm_expert)
    for p in raw.vlm_with_expert.lm_expert.parameters():
        p.requires_grad = True

    # Unfreeze projections
    for submodule in [
        raw.state_proj,
        raw.action_in_proj,
        raw.action_out_proj,
        raw.action_time_mlp_in,
        raw.action_time_mlp_out,
    ]:
        for p in submodule.parameters():
            p.requires_grad = True

    # Explicitly keep vision encoder eval/frozen
    raw.vlm_with_expert.get_vlm_model().vision_model.eval()

    # Connector stays frozen in v1
    raw.vlm_with_expert.get_vlm_model().connector.eval()


def build_param_groups(policy, raw, lora_lr: float, action_lr: float) -> list[dict]:
    lora_params = [
        p for n, p in policy.named_parameters()
        if "lora_" in n and p.requires_grad
    ]
    action_params = (
        list(raw.vlm_with_expert.lm_expert.parameters()) +
        list(raw.state_proj.parameters()) +
        list(raw.action_in_proj.parameters()) +
        list(raw.action_out_proj.parameters()) +
        list(raw.action_time_mlp_in.parameters()) +
        list(raw.action_time_mlp_out.parameters())
    )
    # Deduplicate (lm_expert params were already requires_grad=True)
    action_params = [p for p in action_params if p.requires_grad]

    lora_ids  = {id(p) for p in lora_params}
    action_params_dedup = [p for p in action_params if id(p) not in lora_ids]

    n_lora   = sum(p.numel() for p in lora_params)
    n_action = sum(p.numel() for p in action_params_dedup)
    print(f"  LoRA params:   {n_lora:,} (lr={lora_lr})")
    print(f"  Action params: {n_action:,} (lr={action_lr})")
    return [
        {"params": lora_params,          "lr": lora_lr,   "name": "lora"},
        {"params": action_params_dedup,  "lr": action_lr, "name": "action"},
    ]


def get_cosine_lr(step: int, warmup: int, total: int, base: float, min_lr: float) -> float:
    if step <= warmup:
        return base * max(step, 1) / max(warmup, 1)
    progress = (step - warmup) / max(total - warmup, 1)
    return min_lr + 0.5 * (base - min_lr) * (1.0 + math.cos(math.pi * progress))


def update_lr(optimizer, step: int, warmup: int, total: int,
              lora_base: float, action_base: float,
              lora_min: float, action_min: float) -> None:
    lrs = [
        get_cosine_lr(step, warmup, total, lora_base,   lora_min),
        get_cosine_lr(step, warmup, total, action_base, action_min),
    ]
    for pg, lr in zip(optimizer.param_groups, lrs):
        pg["lr"] = lr


# ──────────────────────────────────────────────────────────────────────────────
# Checkpoint save
# ──────────────────────────────────────────────────────────────────────────────

def copy_processors(src: Path, dst: Path) -> None:
    for name in (
        "policy_preprocessor.json",
        "policy_preprocessor_step_5_normalizer_processor.safetensors",
        "policy_postprocessor.json",
        "policy_postprocessor_step_0_unnormalizer_processor.safetensors",
        "config.json",
    ):
        if (src / name).exists():
            shutil.copy(src / name, dst / name)


def save_ckpt(out_dir: Path, step: int, policy, init_path: Path) -> None:
    ckpt = out_dir / "checkpoints" / f"{step:06d}" / "pretrained_model"
    ckpt.mkdir(parents=True, exist_ok=True)
    # Save merged model (LoRA weights baked in)
    merged = policy.merge_and_unload()
    merged.save_pretrained(ckpt)
    if hasattr(merged, "config"):
        merged.config.save_pretrained(ckpt)
    copy_processors(init_path, ckpt)
    print(f"[ckpt] saved step {step:06d} → {ckpt}", flush=True)


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(description="Approach D: action co-training")
    ap.add_argument("--init",         default=DEFAULT_INIT)
    ap.add_argument("--dataset",      default=DEFAULT_DATASET)
    ap.add_argument("--dataset-root", default=DEFAULT_DATASET_ROOT)
    ap.add_argument("--out-dir",      default=DEFAULT_OUT)
    ap.add_argument("--steps",        type=int,   default=3000)
    ap.add_argument("--batch-size",   type=int,   default=32)
    ap.add_argument("--num-workers",  type=int,   default=0,
                    help="DataLoader workers (0 = inline, avoids dataset copy overhead)")
    ap.add_argument("--lora-rank",    type=int,   default=16)
    ap.add_argument("--lora-alpha",   type=int,   default=32)
    ap.add_argument("--lora-dropout", type=float, default=0.05)
    ap.add_argument("--lora-lr",      type=float, default=1e-5)
    ap.add_argument("--action-lr",    type=float, default=2e-5)
    ap.add_argument("--lora-min-lr",  type=float, default=5e-7)
    ap.add_argument("--action-min-lr",type=float, default=1e-6)
    ap.add_argument("--weight-decay", type=float, default=1e-10)
    ap.add_argument("--grad-clip",    type=float, default=10.0)
    ap.add_argument("--warmup-steps", type=int,   default=300)
    ap.add_argument("--frame-min",    type=int,   default=5,
                    help="Min frame_index within episode to use")
    ap.add_argument("--frame-max",    type=int,   default=80,
                    help="Max frame_index within episode to use")
    ap.add_argument("--save-steps",   default="500,1000,1500,2000,3000")
    ap.add_argument("--log-every",    type=int,   default=25)
    ap.add_argument("--use-amp",      action="store_true", default=True)
    ap.add_argument("--no-amp",       action="store_false", dest="use_amp")
    ap.add_argument("--seed",         type=int,   default=42)
    ap.add_argument("--dry-run",      action="store_true")
    ap.add_argument("--wandb",        action="store_true")
    ap.add_argument("--wandb-project",default="lerobot")
    ap.add_argument("--wandb-name",   default="eval2-stage2d-action-cotrain-stage2-20k-rel60")
    args = ap.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    init_path = Path(args.init)
    out_dir   = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    save_steps = {int(x) for x in args.save_steps.split(",") if x.strip()}

    # ── Dataset ──────────────────────────────────────────────────────────────
    print(f"Loading dataset: {args.dataset}")
    ds_meta = LeRobotDatasetMetadata(args.dataset, root=args.dataset_root)
    # Use a temporary policy load just to get delta_timestamps
    _tmp_policy = SmolVLAPolicy.from_pretrained(args.init)
    delta_ts = resolve_delta_timestamps(_tmp_policy.config, ds_meta)
    del _tmp_policy

    ds = LeRobotDataset(
        args.dataset,
        root=args.dataset_root,
        delta_timestamps=delta_ts,
        video_backend="pyav",
    )
    print(f"  Dataset loaded: {len(ds)} frames, {ds.num_episodes} episodes")

    ep_meta = load_episode_meta(Path(args.dataset_root))
    print(f"  Episode metadata: {len(ep_meta)} episodes")

    pool = build_frame_pool(ds, ep_meta, args.frame_min, args.frame_max)
    print(f"  Frame pool groups: {sorted(pool.keys())}")
    for k in sorted(pool.keys()):
        print(f"    {k}: {len(pool[k])} frames")

    sampler = BalancedActionSampler(pool, args.seed)

    # Dry-run: just print composition stats
    if args.dry_run:
        probe = sampler.next_batch(360)
        fam_cnt   = Counter(f for _, f, _ in probe)
        slot_cnt  = Counter(r.meta.target_slot for r, _, _ in probe)
        layout_cnt = Counter(r.meta.layout_code for r, _, _ in probe)
        print("Dry-run 360-sample composition:")
        print("  family:", dict(fam_cnt))
        print("  slot:  ", dict(slot_cnt))
        print("  layout:", dict(layout_cnt))
        # Sample some prompts
        for r, fam, p in probe[:10]:
            print(f"    [{fam:10s}] {p[:80]}")
        return

    # ── Policy ───────────────────────────────────────────────────────────────
    print(f"\nLoading policy from {args.init}")
    policy = SmolVLAPolicy.from_pretrained(args.init)
    policy.config.device = device
    policy.config.train_expert_only = True
    policy.config.freeze_vision_encoder = True

    lora_cfg = LoraConfig(
        r=args.lora_rank,
        lora_alpha=args.lora_alpha,
        target_modules=(
            r"model\.vlm_with_expert\.vlm\.model\.text_model"
            r"\.layers\.[0-9]+\.self_attn\.(q|v)_proj"
        ),
        lora_dropout=args.lora_dropout,
        bias="none",
    )
    policy = get_peft_model(policy, lora_cfg)
    raw = policy.get_base_model().model  # VLAFlowMatching

    print("\nConfiguring trainability...")
    configure_trainability(policy, raw)

    n_train = sum(p.numel() for p in policy.parameters() if p.requires_grad)
    n_total = sum(p.numel() for p in policy.parameters())
    print(f"  Total params: {n_total:,}  |  Trainable: {n_train:,} ({100*n_train/n_total:.2f}%)")

    policy = policy.to(device)
    raw    = policy.get_base_model().model  # re-get after .to(device)

    # ── Optimizer ─────────────────────────────────────────────────────────────
    print("\nBuilding optimizer param groups...")
    param_groups = build_param_groups(policy, raw, args.lora_lr, args.action_lr)
    optimizer = torch.optim.AdamW(
        param_groups,
        weight_decay=args.weight_decay,
        betas=(0.9, 0.95),
        eps=1e-8,
    )

    # BFloat16 has sufficient dynamic range — GradScaler is NOT needed (and errors).
    # We use autocast(bfloat16) for speed without a scaler.
    use_bf16_autocast = args.use_amp and device == "cuda"

    # ── Preprocessor ──────────────────────────────────────────────────────────
    preprocessor, _ = make_pre_post_processors(
        policy.config,
        pretrained_path=args.init,
        preprocessor_overrides={
            "device_processor": {"device": device},
            "normalizer_processor": {
                "stats": ds.meta.stats,
                "features": {
                    **policy.config.input_features,
                    **policy.config.output_features,
                },
                "norm_map": policy.config.normalization_mapping,
            },
            "rename_observations_processor": {"rename_map": {}},
        },
    )

    # ── W&B ───────────────────────────────────────────────────────────────────
    wandb_run = None
    if args.wandb:
        import wandb
        wandb_run = wandb.init(
            project=args.wandb_project,
            name=args.wandb_name,
            config=vars(args),
        )

    # ── Training loop ────────────────────────────────────────────────────────
    policy.train()
    # Keep frozen modules in eval mode throughout training
    raw.vlm_with_expert.get_vlm_model().vision_model.eval()
    raw.vlm_with_expert.get_vlm_model().connector.eval()

    print(f"\nTraining Approach D: {args.steps} steps | batch={args.batch_size} | AMP={args.use_amp}")
    print(f"Output: {out_dir}\n")

    t0      = time.time()
    loss_ema = None
    ema_alpha = 0.1

    for step in range(1, args.steps + 1):
        # LR schedule update
        update_lr(
            optimizer, step,
            warmup=args.warmup_steps, total=args.steps,
            lora_base=args.lora_lr,    action_base=args.action_lr,
            lora_min=args.lora_min_lr, action_min=args.action_min_lr,
        )

        # Build batch
        samples = sampler.next_batch(args.batch_size)
        items   = [fetch_item(ds, rec, prompt) for rec, _, prompt in samples]
        # Scale uint8 images to float [0,1]
        for item in items:
            for key in ds.meta.camera_keys:
                if key in item and isinstance(item[key], torch.Tensor) and item[key].dtype == torch.uint8:
                    item[key] = item[key].float() / 255.0
        batch = collate_batch(items)
        batch = preprocessor(batch)

        # Forward + loss (bfloat16 autocast, no GradScaler needed)
        autocast_ctx = (
            torch.amp.autocast("cuda", dtype=torch.bfloat16)
            if use_bf16_autocast
            else torch.amp.autocast("cuda", enabled=False)
        )
        with autocast_ctx:
            loss, loss_dict = policy.forward(batch)

        # Backward
        all_params = [p for p in policy.parameters() if p.requires_grad]
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(all_params, args.grad_clip)
        optimizer.step()

        loss_val = float(loss.item())
        loss_ema = loss_val if loss_ema is None else (1 - ema_alpha) * loss_ema + ema_alpha * loss_val

        # Logging
        if step % args.log_every == 0 or step == 1:
            elapsed = time.time() - t0
            sps = step / max(elapsed, 1e-6)
            fam_cnt = Counter(fam for _, fam, _ in samples)
            lora_lr   = optimizer.param_groups[0]["lr"]
            action_lr = optimizer.param_groups[1]["lr"]
            print(
                f"step={step:5d} loss={loss_val:.4f}(ema={loss_ema:.4f}) "
                f"grad={float(grad_norm):.2f} "
                f"lr_lora={lora_lr:.1e} lr_act={action_lr:.1e} "
                f"sps={sps:.1f} fam={dict(fam_cnt)}",
                flush=True,
            )
            if wandb_run is not None:
                import wandb
                wandb.log(
                    {
                        "loss": loss_val,
                        "loss_ema": loss_ema,
                        "grad_norm": float(grad_norm),
                        "lr_lora": lora_lr,
                        "lr_action": action_lr,
                        "steps_per_s": sps,
                        **{f"loss_detail/{k}": v for k, v in loss_dict.items()},
                        **{f"family/{k}": v for k, v in fam_cnt.items()},
                    },
                    step=step,
                )

        # Checkpoint
        if step in save_steps or step == args.steps:
            save_ckpt(out_dir, step, policy, init_path)

    if wandb_run is not None:
        wandb_run.finish()
    print(f"\nDone. Output: {out_dir}")


if __name__ == "__main__":
    main()
