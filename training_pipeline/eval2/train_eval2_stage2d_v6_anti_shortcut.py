#!/usr/bin/env python3
"""Approach D2_v6 — LoRA-only anti-shortcut contrastive relative sampling.

Problem diagnosed from D2_v4:
  The model learned a direction-word shortcut:
    "right of X" -> always rightmost bowl
    "left of X"  -> always leftmost bowl
  It ignored the anchor color entirely.

Fix:
  Oversample anti-shortcut relative cases where the direction word does NOT
  map to the corresponding edge bowl:
    right_of_leftmost_anchor -> middle target  (direction=right, target=middle)
    left_of_rightmost_anchor -> middle target  (direction=left,  target=middle)

  These force the model to use the anchor color, not just the direction word.

Relative sampling split (within the relative family):
  60% anti-shortcut: middle-target episodes  (right_of_left, left_of_right -> middle)
  40% edge cases:    edge-target episodes    (left_of_middle -> left, right_of_middle -> right)

Loss design (same as D2_v5):
  Relative samples:    L = L_demo_action only (no distillation)
  Preservation samples (ordinal/exclusion/direct/position):
                       L = L_demo_action + 0.5 * L_distill_to_stage2

Architecture: LoRA-only (connector, action expert, all projections frozen).
"""

from __future__ import annotations

import argparse
import copy
import csv
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

COLORS     = ("blue", "green", "red")
SLOT_NAMES = ("left", "middle", "right")
COLOR_FULL = {"b": "blue", "g": "green", "r": "red"}

PRESERVATION_FAMILIES = {"ordinal", "exclusion", "direct", "position"}

# Overall family weights (relative = 50% of all samples)
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

# Within relative samples: 60% anti-shortcut (middle target), 40% edge target
ANTI_SHORTCUT_FRACTION = 0.60

MAX_ACTION_DIM = 32
CHUNK_SIZE     = 50

DEFAULT_DATASET      = "robot-learning-group47/eval2_108_position_only"
DEFAULT_DATASET_ROOT = "/home/shadeform/lerobot_cache/robot-learning-group47/eval2_108_position_only"
DEFAULT_INIT = (
    "/home/shadeform/outputs/eval2_stage2_compositional_015000_balanced_r2"
    "/checkpoints/020000/pretrained_model"
)
DEFAULT_OUT  = "/home/shadeform/outputs/eval2_stage2_20k_relation_cotrain_d2_v6_anti_shortcut"
COMBO_CSV    = "combo_episode_map_eval2_fixed_bowl_fixed_banana_108.csv"


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
# Prompt generators
# ──────────────────────────────────────────────────────────────────────────────

def anti_shortcut_relative_prompts(meta: EpisodeMeta) -> list[str]:
    """Anti-shortcut: target=middle (ti==1). Direction word != edge bowl.
    right_of_leftmost -> middle  (right word but NOT rightmost result)
    left_of_rightmost -> middle  (left word but NOT leftmost result)
    """
    assert meta.target_slot_idx == 1
    layout = meta.layout
    return [
        f"Put the banana into the bowl on the right of the {layout[0]} bowl from the robot perspective.",
        f"Put the banana into the bowl on the left of the {layout[2]} bowl from the robot perspective.",
    ]


def edge_relative_prompts(meta: EpisodeMeta) -> list[str]:
    """Edge cases: target=left (ti==0) or target=right (ti==2).
    left_of_middle  -> left   (direction=left, target IS leftmost)
    right_of_middle -> right  (direction=right, target IS rightmost)
    """
    assert meta.target_slot_idx != 1
    layout = meta.layout
    ti     = meta.target_slot_idx
    if ti == 0:
        return [f"Put the banana into the bowl on the left of the {layout[1]} bowl from the robot perspective."]
    else:  # ti == 2
        return [f"Put the banana into the bowl on the right of the {layout[1]} bowl from the robot perspective."]


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


# ──────────────────────────────────────────────────────────────────────────────
# Frame pool & anti-shortcut sampler
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


class AntiShortcutSampler:
    """Balanced sampler with anti-shortcut relative weighting.

    For relative samples:
      60% drawn from middle-target episodes (anti-shortcut)
      40% drawn from edge-target episodes

    For all other families: drawn uniformly from full pool.
    """

    def __init__(self, pool: dict[str, list[FrameRecord]], seed: int,
                 anti_shortcut_frac: float = 0.60):
        self.rng  = random.Random(seed)
        self.anti_shortcut_frac = anti_shortcut_frac

        # Full pool (for non-relative families)
        self.all_keys = sorted(pool.keys())
        self.all_pool = {k: list(v) for k, v in pool.items()}
        self.all_ptrs = {k: 0 for k in self.all_keys}
        for k in self.all_keys:
            self.rng.shuffle(self.all_pool[k])

        # Middle-target pool (ti==1) for anti-shortcut relative
        self.mid_pool = {}
        for k, frames in pool.items():
            mid = [f for f in frames if f.meta.target_slot_idx == 1]
            if mid:
                self.mid_pool[k] = list(mid)
                self.rng.shuffle(self.mid_pool[k])
        self.mid_keys = sorted(self.mid_pool.keys())
        self.mid_ptrs = {k: 0 for k in self.mid_keys}

        # Edge-target pool (ti != 1) for edge relative
        self.edge_pool = {}
        for k, frames in pool.items():
            edge = [f for f in frames if f.meta.target_slot_idx != 1]
            if edge:
                self.edge_pool[k] = list(edge)
                self.rng.shuffle(self.edge_pool[k])
        self.edge_keys = sorted(self.edge_pool.keys())
        self.edge_ptrs = {k: 0 for k in self.edge_keys}

        print(f"  AntiShortcutSampler: "
              f"{sum(len(v) for v in self.mid_pool.values())} middle-target frames, "
              f"{sum(len(v) for v in self.edge_pool.values())} edge-target frames")

    def _next_from(self, pool, keys, ptrs) -> FrameRecord:
        key    = keys[self.rng.randint(0, len(keys) - 1)]
        frames = pool[key]
        rec    = frames[ptrs[key] % len(frames)]
        ptrs[key] += 1
        return rec

    def next_batch(self, n: int) -> list[tuple[FrameRecord, str, str]]:
        out = []
        for _ in range(n):
            # Determine family
            roll   = self.rng.random()
            family = FAMILY_ORDER[-1]
            for f, thresh in zip(FAMILY_ORDER, FAMILY_CUMULATIVE):
                if roll < thresh:
                    family = f
                    break

            if family == "relative":
                # Weighted choice: anti-shortcut vs edge
                if self.rng.random() < self.anti_shortcut_frac and self.mid_keys:
                    rec    = self._next_from(self.mid_pool, self.mid_keys, self.mid_ptrs)
                    opts   = anti_shortcut_relative_prompts(rec.meta)
                    prompt = self.rng.choice(opts)
                    tag    = "relative_anti"
                elif self.edge_keys:
                    rec    = self._next_from(self.edge_pool, self.edge_keys, self.edge_ptrs)
                    opts   = edge_relative_prompts(rec.meta)
                    prompt = self.rng.choice(opts)
                    tag    = "relative_edge"
                else:
                    # Fallback: ordinal
                    rec    = self._next_from(self.all_pool, self.all_keys, self.all_ptrs)
                    prompt = ordinal_prompt(rec.meta)
                    tag    = "ordinal"
                out.append((rec, tag, prompt))

            else:
                rec = self._next_from(self.all_pool, self.all_keys, self.all_ptrs)
                if family == "ordinal":
                    prompt = ordinal_prompt(rec.meta)
                elif family == "exclusion":
                    prompt = exclusion_prompt(rec.meta)
                elif family == "direct":
                    prompt = direct_prompt(rec.meta)
                else:
                    prompt = position_prompt(rec.meta)
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

def configure_trainability_lora_only(policy, raw) -> None:
    """Freeze everything, unfreeze only LoRA. Connector stays frozen."""
    for p in raw.parameters():
        p.requires_grad = False
    for name, p in policy.named_parameters():
        if "lora_" in name:
            p.requires_grad = True
    # Keep vision encoder and connector in eval mode
    raw.vlm_with_expert.get_vlm_model().vision_model.eval()
    raw.vlm_with_expert.get_vlm_model().connector.eval()


def build_param_groups_lora_only(policy, lora_lr: float) -> list[dict]:
    lora_params = [p for n, p in policy.named_parameters() if "lora_" in n and p.requires_grad]
    n_lora = sum(p.numel() for p in lora_params)
    print(f"  LoRA params: {n_lora:,} (lr={lora_lr})")
    return [{"params": lora_params, "lr": lora_lr, "name": "lora"}]


def get_cosine_lr(step: int, warmup: int, total: int, base: float, min_lr: float,
                  hold_until: int = 0) -> float:
    if step <= warmup:
        return base * max(step, 1) / max(warmup, 1)
    hold_end = max(hold_until, warmup)
    if step <= hold_end:
        return base
    progress = (step - hold_end) / max(total - hold_end, 1)
    return min_lr + 0.5 * (base - min_lr) * (1.0 + math.cos(math.pi * progress))


# ──────────────────────────────────────────────────────────────────────────────
# Velocity distillation hook
# ──────────────────────────────────────────────────────────────────────────────

class VelocityCapture:
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
    ap = argparse.ArgumentParser(description="D2_v6: LoRA-only anti-shortcut")
    ap.add_argument("--init",             default=DEFAULT_INIT)
    ap.add_argument("--dataset",          default=DEFAULT_DATASET)
    ap.add_argument("--dataset-root",     default=DEFAULT_DATASET_ROOT)
    ap.add_argument("--out-dir",          default=DEFAULT_OUT)
    ap.add_argument("--steps",            type=int,   default=2000)
    ap.add_argument("--batch-size",       type=int,   default=32)
    ap.add_argument("--lora-rank",        type=int,   default=16)
    ap.add_argument("--lora-alpha",       type=int,   default=32)
    ap.add_argument("--lora-dropout",     type=float, default=0.05)
    ap.add_argument("--lora-lr",          type=float, default=2e-5)
    ap.add_argument("--lora-min-lr",      type=float, default=1e-7)
    ap.add_argument("--beta",             type=float, default=0.5,
                    help="Distillation beta for preservation samples only.")
    ap.add_argument("--anti-shortcut-frac", type=float, default=0.60,
                    help="Fraction of relative samples that are anti-shortcut (middle target).")
    ap.add_argument("--weight-decay",     type=float, default=1e-10)
    ap.add_argument("--grad-clip",        type=float, default=10.0)
    ap.add_argument("--warmup-steps",     type=int,   default=100)
    ap.add_argument("--hold-steps",       type=int,   default=1200)
    ap.add_argument("--frame-min",        type=int,   default=5)
    ap.add_argument("--frame-max",        type=int,   default=80)
    ap.add_argument("--save-steps",       default="200,500,1000,1500,2000")
    ap.add_argument("--log-every",        type=int,   default=25)
    ap.add_argument("--use-amp",          action="store_true", default=True)
    ap.add_argument("--no-amp",           action="store_false", dest="use_amp")
    ap.add_argument("--seed",             type=int,   default=42)
    ap.add_argument("--dry-run",          action="store_true")
    ap.add_argument("--wandb",            action="store_true")
    ap.add_argument("--wandb-project",    default="lerobot")
    ap.add_argument("--wandb-name",       default="eval2-stage2d-v6-antishortcut-lora2e5")
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
    sampler = AntiShortcutSampler(pool, args.seed, args.anti_shortcut_frac)

    if args.dry_run:
        probe = sampler.next_batch(400)
        fam   = Counter(f for _, f, _ in probe)
        slot  = Counter(r.meta.target_slot for r, _, _ in probe)
        lay   = Counter(r.meta.layout_code for r, _, _ in probe)
        print("Dry-run 400-sample composition:")
        print("  family:", dict(fam))
        print("  slot:  ", dict(slot))
        print("  layout:", dict(lay))
        print("  Anti-shortcut fraction in relative:",
              fam.get("relative_anti", 0) /
              max(fam.get("relative_anti", 0) + fam.get("relative_edge", 0), 1))
        for r, f, p in probe[:12]:
            print(f"    [{f:15s}] slot={r.meta.target_slot} {p[:70]}")
        return

    # ── Teacher (frozen Stage2-20k) ─────────────────────────────────────────
    teacher_policy = None
    teacher_cap    = VelocityCapture()
    if args.beta > 0.0:
        print(f"\nLoading teacher (frozen): {args.init}")
        teacher_policy = SmolVLAPolicy.from_pretrained(args.init)
        teacher_policy = teacher_policy.to(device)
        for p in teacher_policy.parameters():
            p.requires_grad = False
        teacher_policy.eval()
        teacher_cap.register(teacher_policy.model.action_out_proj)
        print(f"  Teacher loaded: {sum(p.numel() for p in teacher_policy.parameters()):,} params")

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

    print("\nConfiguring trainability (LoRA only, connector frozen)...")
    configure_trainability_lora_only(policy, raw)

    n_train = sum(p.numel() for p in policy.parameters() if p.requires_grad)
    n_total = sum(p.numel() for p in policy.parameters())
    print(f"  Total: {n_total:,}  |  Trainable: {n_train:,} ({100*n_train/n_total:.2f}%)")

    policy = policy.to(device)
    raw    = policy.get_base_model().model

    student_cap = VelocityCapture()
    student_cap.register(raw.action_out_proj)

    # ── Optimizer ─────────────────────────────────────────────────────────────
    print("\nBuilding optimizer param groups...")
    param_groups = build_param_groups_lora_only(policy, args.lora_lr)
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

    print(f"\nTraining D2_v6: {args.steps} steps | batch={args.batch_size} | "
          f"anti_shortcut_frac={args.anti_shortcut_frac} | beta={args.beta} | AMP={use_bf16}")
    print(f"  LR: warmup {args.warmup_steps} → hold until {args.hold_steps} → cosine to {args.steps}")
    print(f"  Distillation: preservation samples only (relative → no distill)")
    print(f"Output: {out_dir}\n")

    t0        = time.time()
    loss_ema  = None
    ema_alpha = 0.1
    all_params = [p for p in policy.parameters() if p.requires_grad]

    for step in range(1, args.steps + 1):
        # LR schedule
        lr = get_cosine_lr(step, args.warmup_steps, args.steps,
                           args.lora_lr, args.lora_min_lr, args.hold_steps)
        for pg in optimizer.param_groups:
            pg["lr"] = lr

        # ── Build batch ──────────────────────────────────────────────────────
        samples = sampler.next_batch(args.batch_size)
        items   = [fetch_item(ds, rec, prompt) for rec, _, prompt in samples]
        scale_images(items, ds.meta.camera_keys)
        batch = collate_batch(items)
        batch = preprocessor(batch)

        # ── Family classification ─────────────────────────────────────────────
        families = [fam for _, fam, _ in samples]
        # Preservation = non-relative (teacher distillation applied)
        preserve_mask = torch.tensor(
            [fam in PRESERVATION_FAMILIES for fam in families],
            dtype=torch.bool, device=device
        )
        n_anti = sum(1 for f in families if f == "relative_anti")
        n_edge = sum(1 for f in families if f == "relative_edge")
        n_pres = int(preserve_mask.sum().item())

        # ── Shared noise/time ────────────────────────────────────────────────
        B     = args.batch_size
        noise = torch.randn(B, CHUNK_SIZE, MAX_ACTION_DIM, device=device)
        time_ = torch.rand(B, device=device)

        # ── Student forward ──────────────────────────────────────────────────
        autocast_ctx = (
            torch.amp.autocast("cuda", dtype=torch.bfloat16)
            if use_bf16 else torch.amp.autocast("cuda", enabled=False)
        )
        student_cap.clear()
        with autocast_ctx:
            loss_action, _ = policy.forward(batch, noise=noise, time=time_)

        # ── Selective distillation (preservation only) ────────────────────────
        loss_distill = torch.tensor(0.0, device=device)
        if teacher_policy is not None and n_pres > 0:
            teacher_cap.clear()
            with torch.no_grad(), autocast_ctx:
                teacher_policy.forward(batch, noise=noise, time=time_)
            v_student = student_cap.vel
            v_teacher = teacher_cap.vel
            if v_student is not None and v_teacher is not None:
                per_sample = F.mse_loss(
                    v_student.float(),
                    v_teacher.float().detach(),
                    reduction="none",
                ).mean(dim=list(range(1, v_student.dim())))
                loss_distill = per_sample[preserve_mask].mean()

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
            elapsed  = time.time() - t0
            sps      = step / max(elapsed, 1e-6)
            fam_cnt  = Counter(families)
            print(
                f"step={step:5d} loss={loss_val:.4f}(ema={loss_ema:.4f}) "
                f"act={float(loss_action):.4f} distill={float(loss_distill):.4f} "
                f"[anti={n_anti} edge={n_edge} pres={n_pres}] "
                f"grad={float(grad_norm):.3f} lr={lr:.1e} "
                f"sps={sps:.1f} fam={dict(fam_cnt)}",
                flush=True,
            )
            if wandb_run is not None:
                import wandb
                wandb.log({
                    "loss":            loss_val,
                    "loss_ema":        loss_ema,
                    "loss_action":     float(loss_action),
                    "loss_distill":    float(loss_distill),
                    "n_anti_shortcut": n_anti,
                    "n_edge":          n_edge,
                    "n_preserve":      n_pres,
                    "grad_norm":       float(grad_norm),
                    "lr_lora":         lr,
                    "steps_per_s":     sps,
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
