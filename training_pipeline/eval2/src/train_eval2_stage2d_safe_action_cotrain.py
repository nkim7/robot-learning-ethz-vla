#!/usr/bin/env python3
"""Approach D2 — Conservative Stage2-20k relation co-training with behavior preservation.

Key differences vs D1:
  - Much lower LRs (5e-6 LoRA, 1e-6 action expert / projections)
  - Frozen teacher (Stage2-20k) for behavior preservation distillation
  - L = L_demo_action + beta * L_distill_to_stage2
  - Lighter data mix (50% relative instead of 60%)
  - Short first run: 1000 steps, save at 200/500/1000
  - No auxiliary slot loss, no slot head, no counterfactual prompts

Architecture:
  Trainable:
    VLM q/v LoRA          (LR 5e-6)
    lm_expert             (LR 1e-6)
    state/action projs    (LR 1e-6)
  Frozen:
    vision encoder
    base VLM except LoRA
    connector

Teacher:
    frozen copy of Stage2-20k
    used to regularize student via velocity distillation on action_out_proj
"""

from __future__ import annotations

import argparse
import copy
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
import torch.nn.functional as F
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

LAYOUT_CODES  = ("bgr", "brg", "gbr", "grb", "rbg", "rgb")
COLORS        = ("blue", "green", "red")
SLOT_NAMES    = ("left", "middle", "right")
COLOR_FULL    = {"b": "blue", "g": "green", "r": "red"}

# D2 prompt mix
FAMILY_WEIGHTS = {
    "relative":  0.50,
    "ordinal":   0.20,
    "exclusion": 0.10,
    "direct":    0.10,
    "position":  0.10,
}
FAMILY_ORDER      = list(FAMILY_WEIGHTS.keys())
FAMILY_CUMULATIVE = []
_acc = 0.0
for _fam in FAMILY_ORDER:
    _acc += FAMILY_WEIGHTS[_fam]
    FAMILY_CUMULATIVE.append(_acc)

# From policy config inspection
MAX_ACTION_DIM = 32
CHUNK_SIZE     = 50

DEFAULT_DATASET      = "robot-learning-group47/eval2_108_position_only"
DEFAULT_DATASET_ROOT = "/home/shadeform/lerobot_cache/robot-learning-group47/eval2_108_position_only"
DEFAULT_INIT         = (
    "/home/shadeform/outputs/eval2_stage2_compositional_015000_balanced_r2"
    "/checkpoints/020000/pretrained_model"
)
DEFAULT_OUT          = "/home/shadeform/outputs/eval2_stage2_20k_relation_cotrain_d2_safe_v1"
COMBO_CSV            = "combo_episode_map_eval2_fixed_bowl_fixed_banana_108.csv"

# ──────────────────────────────────────────────────────────────────────────────
# Episode metadata
# ──────────────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class EpisodeMeta:
    episode_index:   int
    layout:          list
    layout_code:     str
    target_color:    str
    target_slot:     str
    target_slot_idx: int


def load_episode_meta(dataset_root: Path) -> dict[int, EpisodeMeta]:
    csv_path = dataset_root / COMBO_CSV
    ep_meta: dict[int, EpisodeMeta] = {}
    with csv_path.open() as f:
        for row in csv.DictReader(f):
            layout_code  = row["bowl_order_left_to_right"].strip().lower()
            layout       = [COLOR_FULL[ch] for ch in layout_code]
            target_color = row["target_bowl_color"].strip().lower()
            ti           = layout.index(target_color)
            for ep in range(int(row["episode_start"]), int(row["episode_end"]) + 1):
                ep_meta[ep] = EpisodeMeta(
                    episode_index   = ep,
                    layout          = layout,
                    layout_code     = layout_code,
                    target_color    = target_color,
                    target_slot     = SLOT_NAMES[ti],
                    target_slot_idx = ti,
                )
    return ep_meta


# ──────────────────────────────────────────────────────────────────────────────
# Action-consistent prompt generation
# ──────────────────────────────────────────────────────────────────────────────

def valid_relative_prompts(meta: EpisodeMeta) -> list[str]:
    layout = meta.layout
    ti     = meta.target_slot_idx
    out    = []
    if ti > 0:
        anchor = layout[ti - 1]
        out.append(
            f"Put the banana into the bowl on the right of the {anchor} bowl "
            f"from the robot perspective."
        )
    if ti < 2:
        anchor = layout[ti + 1]
        out.append(
            f"Put the banana into the bowl on the left of the {anchor} bowl "
            f"from the robot perspective."
        )
    return out


def ordinal_prompt(meta: EpisodeMeta) -> str:
    return (
        f"Put the banana into the {('1st','2nd','3rd')[meta.target_slot_idx]} bowl "
        f"from the left from the robot perspective."
    )


def exclusion_prompt(meta: EpisodeMeta) -> str:
    excl = [c for c in COLORS if c != meta.target_color]
    return f"Put the banana into the bowl that is not {excl[0]} and not {excl[1]}."


def direct_prompt(meta: EpisodeMeta) -> str:
    return f"Put the banana into the {meta.target_color} bowl."


def position_prompt(meta: EpisodeMeta) -> str:
    return f"Pick up the banana and put it in the {meta.target_slot} bowl."


def sample_prompt(meta: EpisodeMeta, rng: random.Random) -> tuple[str, str]:
    roll   = rng.random()
    family = FAMILY_ORDER[-1]
    for f, thresh in zip(FAMILY_ORDER, FAMILY_CUMULATIVE):
        if roll < thresh:
            family = f
            break
    if family == "relative":
        opts = valid_relative_prompts(meta)
        if not opts:
            return "ordinal", ordinal_prompt(meta)
        return family, rng.choice(opts)
    elif family == "ordinal":
        return family, ordinal_prompt(meta)
    elif family == "exclusion":
        return family, exclusion_prompt(meta)
    elif family == "direct":
        return family, direct_prompt(meta)
    else:
        return family, position_prompt(meta)


# ──────────────────────────────────────────────────────────────────────────────
# Frame pool & sampler
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class FrameRecord:
    global_idx:    int
    episode_index: int
    frame_index:   int
    meta:          EpisodeMeta


def build_frame_pool(
    ds:        LeRobotDataset,
    ep_meta:   dict[int, EpisodeMeta],
    frame_min: int,
    frame_max: int,
) -> dict[str, list[FrameRecord]]:
    ep_vals = list(ds.hf_dataset["episode_index"])
    fr_vals = list(ds.hf_dataset["frame_index"])
    pool: dict[str, list[FrameRecord]] = {}
    for gi, (ep_val, fr_val) in enumerate(zip(ep_vals, fr_vals)):
        ep   = int(ep_val)
        fr   = int(fr_val)
        if fr < frame_min or fr > frame_max:
            continue
        meta = ep_meta.get(ep)
        if meta is None:
            continue
        key = f"{meta.layout_code}_{meta.target_color[0]}"
        pool.setdefault(key, []).append(FrameRecord(gi, ep, fr, meta))
    return pool


class BalancedSampler:
    def __init__(self, pool: dict[str, list[FrameRecord]], seed: int):
        self.pool = pool
        self.keys = sorted(pool.keys())
        self.ptrs = {k: 0 for k in self.keys}
        self.rng  = random.Random(seed)
        for k in self.keys:
            self.rng.shuffle(self.pool[k])

    def next_batch(self, n: int) -> list[tuple[FrameRecord, str, str]]:
        out       = []
        key_cycle = [self.keys[i % len(self.keys)] for i in range(n)]
        self.rng.shuffle(key_cycle)
        for key in key_cycle:
            frames = self.pool[key]
            rec    = frames[self.ptrs[key] % len(frames)]
            self.ptrs[key] += 1
            family, prompt = sample_prompt(rec.meta, self.rng)
            out.append((rec, family, prompt))
        return out


# ──────────────────────────────────────────────────────────────────────────────
# Batch helpers
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


def scale_images(items: list[dict], camera_keys: list[str]) -> None:
    for item in items:
        for key in camera_keys:
            if key in item and isinstance(item[key], torch.Tensor) and item[key].dtype == torch.uint8:
                item[key] = item[key].float() / 255.0


# ──────────────────────────────────────────────────────────────────────────────
# Model setup
# ──────────────────────────────────────────────────────────────────────────────

def configure_trainability(policy, raw, lora_only: bool = False) -> None:
    """Freeze everything, then unfreeze LoRA (and optionally action expert + projections)."""
    for p in raw.parameters():
        p.requires_grad = False
    for name, p in policy.named_parameters():
        if "lora_" in name:
            p.requires_grad = True
    if not lora_only:
        for p in raw.vlm_with_expert.lm_expert.parameters():
            p.requires_grad = True
        for sub in [raw.state_proj, raw.action_in_proj, raw.action_out_proj,
                    raw.action_time_mlp_in, raw.action_time_mlp_out]:
            for p in sub.parameters():
                p.requires_grad = True
    # Keep frozen in eval mode
    raw.vlm_with_expert.get_vlm_model().vision_model.eval()
    raw.vlm_with_expert.get_vlm_model().connector.eval()


def build_param_groups(policy, raw, lora_lr: float, action_lr: float) -> list[dict]:
    lora_params = [p for n, p in policy.named_parameters() if "lora_" in n and p.requires_grad]
    action_params = (
        list(raw.vlm_with_expert.lm_expert.parameters()) +
        list(raw.state_proj.parameters()) +
        list(raw.action_in_proj.parameters()) +
        list(raw.action_out_proj.parameters()) +
        list(raw.action_time_mlp_in.parameters()) +
        list(raw.action_time_mlp_out.parameters())
    )
    lora_ids            = {id(p) for p in lora_params}
    action_params_dedup = [p for p in action_params if p.requires_grad and id(p) not in lora_ids]
    n_lora   = sum(p.numel() for p in lora_params)
    n_action = sum(p.numel() for p in action_params_dedup)
    print(f"  LoRA params:   {n_lora:,} (lr={lora_lr})")
    print(f"  Action params: {n_action:,} (lr={action_lr})")
    return [
        {"params": lora_params,          "lr": lora_lr,   "name": "lora"},
        {"params": action_params_dedup,  "lr": action_lr, "name": "action"},
    ]


def get_cosine_lr(step: int, warmup: int, total: int, base: float, min_lr: float,
                  hold_until: int = 0) -> float:
    """Warmup → optional hold at peak → cosine decay.

    hold_until: step at which decay begins (0 = decay immediately after warmup).
    """
    if step <= warmup:
        return base * max(step, 1) / max(warmup, 1)
    hold_end = max(hold_until, warmup)
    if step <= hold_end:
        return base  # hold at peak LR
    progress = (step - hold_end) / max(total - hold_end, 1)
    return min_lr + 0.5 * (base - min_lr) * (1.0 + math.cos(math.pi * progress))


def update_lr(optimizer, step, warmup, total, lora_base, action_base, lora_min, action_min,
              hold_until: int = 0):
    lrs = [
        get_cosine_lr(step, warmup, total, lora_base,   lora_min, hold_until),
        get_cosine_lr(step, warmup, total, action_base, action_min, hold_until),
    ]
    for pg, lr in zip(optimizer.param_groups, lrs):
        pg["lr"] = lr


# ──────────────────────────────────────────────────────────────────────────────
# Velocity distillation hook
# ──────────────────────────────────────────────────────────────────────────────

class VelocityCapture:
    """Hooks action_out_proj to capture the velocity prediction tensor."""
    def __init__(self):
        self.vel: torch.Tensor | None = None
        self._handle = None

    def register(self, module: torch.nn.Module) -> None:
        def _hook(_mod, _inp, output):
            self.vel = output
        self._handle = module.register_forward_hook(_hook)

    def remove(self) -> None:
        if self._handle is not None:
            self._handle.remove()

    def clear(self) -> None:
        self.vel = None


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
    # deepcopy to avoid merge_and_unload() destroying the training policy's LoRA adapters
    policy_copy = copy.deepcopy(policy)
    merged = policy_copy.merge_and_unload()
    merged.save_pretrained(ckpt)
    if hasattr(merged, "config"):
        merged.config.save_pretrained(ckpt)
    copy_processors(init_path, ckpt)
    del policy_copy, merged
    print(f"[ckpt] saved step {step:06d} → {ckpt}", flush=True)


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(description="Approach D2: conservative action co-training")
    ap.add_argument("--init",          default=DEFAULT_INIT)
    ap.add_argument("--dataset",       default=DEFAULT_DATASET)
    ap.add_argument("--dataset-root",  default=DEFAULT_DATASET_ROOT)
    ap.add_argument("--out-dir",       default=DEFAULT_OUT)
    ap.add_argument("--steps",         type=int,   default=1000)
    ap.add_argument("--batch-size",    type=int,   default=32)
    ap.add_argument("--lora-rank",     type=int,   default=16)
    ap.add_argument("--lora-alpha",    type=int,   default=32)
    ap.add_argument("--lora-dropout",  type=float, default=0.05)
    ap.add_argument("--lora-lr",       type=float, default=5e-6)
    ap.add_argument("--lora-min-lr",   type=float, default=1e-7)
    ap.add_argument("--action-lr",     type=float, default=1e-6)
    ap.add_argument("--action-min-lr", type=float, default=1e-8)
    ap.add_argument("--weight-decay",  type=float, default=1e-10)
    ap.add_argument("--grad-clip",     type=float, default=10.0)
    ap.add_argument("--warmup-steps",  type=int,   default=200)
    ap.add_argument("--hold-steps",    type=int,   default=0,
                    help="Hold peak LR until this step, then cosine decay. 0 = decay right after warmup.")
    ap.add_argument("--lora-only",     action="store_true", default=False,
                    help="Freeze action expert and all projections. Train LoRA only. No teacher needed.")
    ap.add_argument("--beta",          type=float, default=0.5,
                    help="Distillation loss weight. Increase to 1.0 if movement degrades.")
    ap.add_argument("--frame-min",     type=int,   default=5)
    ap.add_argument("--frame-max",     type=int,   default=80)
    ap.add_argument("--save-steps",    default="200,500,1000")
    ap.add_argument("--log-every",     type=int,   default=25)
    ap.add_argument("--use-amp",       action="store_true", default=True)
    ap.add_argument("--no-amp",        action="store_false", dest="use_amp")
    ap.add_argument("--seed",          type=int,   default=42)
    ap.add_argument("--dry-run",       action="store_true")
    ap.add_argument("--wandb",         action="store_true")
    ap.add_argument("--wandb-project", default="lerobot")
    ap.add_argument("--wandb-name",    default="eval2-stage2d-safe-action-cotrain-stage2-20k-rel50")
    args = ap.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device    = "cuda" if torch.cuda.is_available() else "cpu"
    init_path = Path(args.init)
    out_dir   = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    save_steps = {int(x) for x in args.save_steps.split(",") if x.strip()}

    # ── Dataset ──────────────────────────────────────────────────────────────
    print(f"Loading dataset: {args.dataset}")
    ds_meta  = LeRobotDatasetMetadata(args.dataset, root=args.dataset_root)
    _tmp     = SmolVLAPolicy.from_pretrained(args.init)
    delta_ts = resolve_delta_timestamps(_tmp.config, ds_meta)
    del _tmp

    ds = LeRobotDataset(args.dataset, root=args.dataset_root,
                        delta_timestamps=delta_ts, video_backend="pyav")
    print(f"  {len(ds)} frames, {ds.num_episodes} episodes")

    ep_meta = load_episode_meta(Path(args.dataset_root))
    pool    = build_frame_pool(ds, ep_meta, args.frame_min, args.frame_max)
    sampler = BalancedSampler(pool, args.seed)

    if args.dry_run:
        probe = sampler.next_batch(360)
        fam   = Counter(f for _, f, _ in probe)
        slot  = Counter(r.meta.target_slot for r, _, _ in probe)
        lay   = Counter(r.meta.layout_code for r, _, _ in probe)
        print("Dry-run 360-sample composition:")
        print("  family:", dict(fam))
        print("  slot:  ", dict(slot))
        print("  layout:", dict(lay))
        for r, f, p in probe[:10]:
            print(f"    [{f:10s}] {p[:80]}")
        return

    # ── Teacher (frozen Stage2-20k) — loaded whenever beta > 0 ──────────────
    teacher_policy = None
    teacher_raw    = None
    teacher_cap    = VelocityCapture()
    if args.beta > 0.0:
        print(f"\nLoading teacher (frozen): {args.init}")
        teacher_policy = SmolVLAPolicy.from_pretrained(args.init)
        teacher_policy = teacher_policy.to(device)
        for p in teacher_policy.parameters():
            p.requires_grad = False
        teacher_policy.eval()
        teacher_raw = teacher_policy.model
        teacher_cap.register(teacher_raw.action_out_proj)
        print(f"  Teacher loaded: {sum(p.numel() for p in teacher_policy.parameters()):,} params")
    else:
        print("\n[beta=0] Skipping teacher load — no distillation.")

    # ── Student policy ───────────────────────────────────────────────────────
    print(f"\nLoading student policy: {args.init}")
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
    raw    = policy.get_base_model().model

    print("\nConfiguring trainability...")
    configure_trainability(policy, raw, lora_only=args.lora_only)

    n_train = sum(p.numel() for p in policy.parameters() if p.requires_grad)
    n_total = sum(p.numel() for p in policy.parameters())
    print(f"  Total: {n_total:,}  |  Trainable: {n_train:,} ({100*n_train/n_total:.2f}%)")

    policy = policy.to(device)
    raw    = policy.get_base_model().model

    student_cap = VelocityCapture()
    student_cap.register(raw.action_out_proj)

    # ── Optimizer ─────────────────────────────────────────────────────────────
    print("\nBuilding optimizer param groups...")
    param_groups = build_param_groups(policy, raw, args.lora_lr, args.action_lr)
    optimizer = torch.optim.AdamW(
        param_groups, weight_decay=args.weight_decay, betas=(0.9, 0.95), eps=1e-8,
    )

    use_bf16 = args.use_amp and device == "cuda"

    # ── Preprocessor ──────────────────────────────────────────────────────────
    preprocessor, _ = make_pre_post_processors(
        policy.config,
        pretrained_path=args.init,
        preprocessor_overrides={
            "device_processor": {"device": device},
            "normalizer_processor": {
                "stats": ds.meta.stats,
                "features": {**policy.config.input_features, **policy.config.output_features},
                "norm_map": policy.config.normalization_mapping,
            },
            "rename_observations_processor": {"rename_map": {}},
        },
    )

    # ── W&B ───────────────────────────────────────────────────────────────────
    wandb_run = None
    if args.wandb:
        import wandb
        wandb_run = wandb.init(project=args.wandb_project, name=args.wandb_name, config=vars(args))

    # ── Training ─────────────────────────────────────────────────────────────
    policy.train()
    raw.vlm_with_expert.get_vlm_model().vision_model.eval()
    raw.vlm_with_expert.get_vlm_model().connector.eval()

    print(f"\nTraining D2: {args.steps} steps | batch={args.batch_size} | beta={args.beta} | AMP={use_bf16}")
    print(f"Output: {out_dir}\n")

    t0        = time.time()
    loss_ema  = None
    ema_alpha = 0.1
    all_params = [p for p in policy.parameters() if p.requires_grad]

    for step in range(1, args.steps + 1):
        update_lr(optimizer, step,
                  warmup=args.warmup_steps, total=args.steps,
                  lora_base=args.lora_lr,    action_base=args.action_lr,
                  lora_min=args.lora_min_lr, action_min=args.action_min_lr,
                  hold_until=args.hold_steps)

        # ── Build batch ──────────────────────────────────────────────────────
        samples = sampler.next_batch(args.batch_size)
        items   = [fetch_item(ds, rec, prompt) for rec, _, prompt in samples]
        scale_images(items, ds.meta.camera_keys)
        batch = collate_batch(items)
        batch = preprocessor(batch)

        # ── Pre-generate shared noise/time for distillation ──────────────────
        # Same noise/time → teacher and student process identical flow matching inputs
        B = args.batch_size
        noise = torch.randn(B, CHUNK_SIZE, MAX_ACTION_DIM, device=device)
        time_ = torch.rand(B, device=device)

        # ── Student forward (action loss) ────────────────────────────────────
        autocast_ctx = (
            torch.amp.autocast("cuda", dtype=torch.bfloat16)
            if use_bf16 else torch.amp.autocast("cuda", enabled=False)
        )
        student_cap.clear()
        with autocast_ctx:
            loss_action, loss_dict = policy.forward(batch, noise=noise, time=time_)

        # ── Teacher forward + distillation ───────────────────────────────────────
        loss_distill = torch.tensor(0.0, device=device)
        if teacher_policy is not None:
            teacher_cap.clear()
            with torch.no_grad(), autocast_ctx:
                teacher_policy.forward(batch, noise=noise, time=time_)
            v_student = student_cap.vel
            v_teacher = teacher_cap.vel
            if v_student is not None and v_teacher is not None:
                loss_distill = F.mse_loss(v_student.float(), v_teacher.float().detach())

        loss = loss_action + args.beta * loss_distill

        # ── Backward ─────────────────────────────────────────────────────────
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(all_params, args.grad_clip)
        optimizer.step()

        loss_val = float(loss.item())
        loss_ema = loss_val if loss_ema is None else (1 - ema_alpha) * loss_ema + ema_alpha * loss_val

        # ── Logging ──────────────────────────────────────────────────────────
        if step % args.log_every == 0 or step == 1:
            elapsed   = time.time() - t0
            sps       = step / max(elapsed, 1e-6)
            fam_cnt   = Counter(fam for _, fam, _ in samples)
            lora_lr   = optimizer.param_groups[0]["lr"]
            action_lr = optimizer.param_groups[1]["lr"]
            print(
                f"step={step:5d} loss={loss_val:.4f}(ema={loss_ema:.4f}) "
                f"act={float(loss_action):.4f} distill={float(loss_distill):.4f} "
                f"grad={float(grad_norm):.3f} "
                f"lr_lora={lora_lr:.1e} lr_act={action_lr:.1e} "
                f"sps={sps:.1f} fam={dict(fam_cnt)}",
                flush=True,
            )
            if wandb_run is not None:
                import wandb
                wandb.log({
                    "loss":          loss_val,
                    "loss_ema":      loss_ema,
                    "loss_action":   float(loss_action),
                    "loss_distill":  float(loss_distill),
                    "grad_norm":     float(grad_norm),
                    "lr_lora":       lora_lr,
                    "lr_action":     action_lr,
                    "steps_per_s":   sps,
                    **{f"family/{k}": v for k, v in fam_cnt.items()},
                }, step=step)

        # ── Checkpoint ───────────────────────────────────────────────────────
        if step in save_steps or step == args.steps:
            save_ckpt(out_dir, step, policy, init_path)

    student_cap.remove()
    teacher_cap.remove()
    if wandb_run is not None:
        wandb_run.finish()
    print(f"\nDone. Output: {out_dir}")


if __name__ == "__main__":
    main()
