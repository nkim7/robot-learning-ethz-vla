#!/usr/bin/env python3
"""Approach D2_v5 — LoRA + tiny connector, action expert frozen.

Hypothesis:
  D2_v3 (high-LR LoRA-only) created relational signal but shifted VLM representations
  too far, breaking the frozen action expert's expected input distribution.
  D2_v5 adds a tiny connector LR to adapt the interface back to what the frozen action
  expert expects, while LoRA corrects the relational features.

Architecture:
  Trainable:
    VLM q/v LoRA              (LR 2e-5)
    connector / modality proj (LR 5e-7)
  Frozen:
    vision encoder
    base VLM except LoRA
    lm_expert
    state/action projections
    action in/out projections

Loss design:
  Relative samples:
    L = L_demo_action  (no distillation — Stage2-20k may be wrong on relational prompts)
  Preservation samples (ordinal / exclusion / direct / position):
    L = L_demo_action + 0.5 * L_distill_to_stage2

Teacher: frozen Stage2-20k, used only on preservation samples.
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

LAYOUT_CODES = ("bgr", "brg", "gbr", "grb", "rbg", "rgb")
COLORS       = ("blue", "green", "red")
SLOT_NAMES   = ("left", "middle", "right")
COLOR_FULL   = {"b": "blue", "g": "green", "r": "red"}

# Preservation families — distillation applied here
PRESERVATION_FAMILIES = {"ordinal", "exclusion", "direct", "position"}
RELATIVE_FAMILIES     = {"relative"}

# D2_v5 prompt mix
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

MAX_ACTION_DIM = 32
CHUNK_SIZE     = 50

DEFAULT_DATASET      = "robot-learning-group47/eval2_108_position_only"
DEFAULT_DATASET_ROOT = "/home/shadeform/lerobot_cache/robot-learning-group47/eval2_108_position_only"
DEFAULT_INIT = (
    "/home/shadeform/outputs/eval2_stage2_compositional_015000_balanced_r2"
    "/checkpoints/020000/pretrained_model"
)
DEFAULT_OUT  = "/home/shadeform/outputs/eval2_stage2_20k_relation_cotrain_d2_v5_lora_connector_safe"
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

def configure_trainability_v5(policy, raw) -> None:
    """Freeze everything, then unfreeze LoRA + connector.

    Unlike previous versions, connector stays in TRAIN mode (not eval).
    Vision encoder stays in eval mode.
    Action expert, state/action projections all frozen.
    """
    # Freeze all raw model parameters
    for p in raw.parameters():
        p.requires_grad = False

    # Unfreeze LoRA params (named in policy wrapper)
    for name, p in policy.named_parameters():
        if "lora_" in name:
            p.requires_grad = True

    # Unfreeze connector (modality projection)
    connector = raw.vlm_with_expert.get_vlm_model().connector
    for p in connector.parameters():
        p.requires_grad = True

    # Vision encoder stays frozen and in eval mode
    raw.vlm_with_expert.get_vlm_model().vision_model.eval()
    # NOTE: connector stays in TRAIN mode (do NOT call connector.eval())


def build_param_groups_v5(policy, raw, lora_lr: float, connector_lr: float) -> list[dict]:
    """Two param groups: LoRA at lora_lr, connector at connector_lr."""
    lora_params = [p for n, p in policy.named_parameters() if "lora_" in n and p.requires_grad]
    lora_ids    = {id(p) for p in lora_params}

    connector       = raw.vlm_with_expert.get_vlm_model().connector
    connector_params = [p for p in connector.parameters()
                        if p.requires_grad and id(p) not in lora_ids]

    n_lora      = sum(p.numel() for p in lora_params)
    n_connector = sum(p.numel() for p in connector_params)
    print(f"  LoRA params:      {n_lora:,} (lr={lora_lr})")
    print(f"  Connector params: {n_connector:,} (lr={connector_lr})")

    return [
        {"params": lora_params,       "lr": lora_lr,       "name": "lora"},
        {"params": connector_params,  "lr": connector_lr,  "name": "connector"},
    ]


def get_cosine_lr(step: int, warmup: int, total: int, base: float, min_lr: float,
                  hold_until: int = 0) -> float:
    """Warmup → optional hold at peak → cosine decay."""
    if step <= warmup:
        return base * max(step, 1) / max(warmup, 1)
    hold_end = max(hold_until, warmup)
    if step <= hold_end:
        return base
    progress = (step - hold_end) / max(total - hold_end, 1)
    return min_lr + 0.5 * (base - min_lr) * (1.0 + math.cos(math.pi * progress))


def update_lr_v5(optimizer, step, warmup, total, lora_base, connector_base,
                 lora_min, connector_min, hold_until=0):
    lrs = [
        get_cosine_lr(step, warmup, total, lora_base,      lora_min,      hold_until),
        get_cosine_lr(step, warmup, total, connector_base, connector_min, hold_until),
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
    # deepcopy to avoid merge_and_unload() destroying the live LoRA adapters
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
    ap = argparse.ArgumentParser(description="Approach D2_v5: LoRA + tiny connector co-training")
    ap.add_argument("--init",             default=DEFAULT_INIT)
    ap.add_argument("--dataset",          default=DEFAULT_DATASET)
    ap.add_argument("--dataset-root",     default=DEFAULT_DATASET_ROOT)
    ap.add_argument("--out-dir",          default=DEFAULT_OUT)
    ap.add_argument("--steps",            type=int,   default=1000)
    ap.add_argument("--batch-size",       type=int,   default=32)
    ap.add_argument("--lora-rank",        type=int,   default=16)
    ap.add_argument("--lora-alpha",       type=int,   default=32)
    ap.add_argument("--lora-dropout",     type=float, default=0.05)
    ap.add_argument("--lora-lr",          type=float, default=2e-5)
    ap.add_argument("--lora-min-lr",      type=float, default=1e-7)
    ap.add_argument("--connector-lr",     type=float, default=5e-7)
    ap.add_argument("--connector-min-lr", type=float, default=1e-9)
    ap.add_argument("--beta",             type=float, default=0.5,
                    help="Distillation beta for preservation samples (relative always 0).")
    ap.add_argument("--weight-decay",     type=float, default=1e-10)
    ap.add_argument("--grad-clip",        type=float, default=10.0)
    ap.add_argument("--warmup-steps",     type=int,   default=100)
    ap.add_argument("--hold-steps",       type=int,   default=600,
                    help="Hold peak LR until this step, then cosine decay.")
    ap.add_argument("--frame-min",        type=int,   default=5)
    ap.add_argument("--frame-max",        type=int,   default=80)
    ap.add_argument("--save-steps",       default="200,500,1000")
    ap.add_argument("--log-every",        type=int,   default=25)
    ap.add_argument("--use-amp",          action="store_true", default=True)
    ap.add_argument("--no-amp",           action="store_false", dest="use_amp")
    ap.add_argument("--seed",             type=int,   default=42)
    ap.add_argument("--dry-run",          action="store_true")
    ap.add_argument("--wandb",            action="store_true")
    ap.add_argument("--wandb-project",    default="lerobot")
    ap.add_argument("--wandb-name",       default="eval2-stage2d-v5-lora2e5-connector5e7-preserve")
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

    # ── Teacher (frozen Stage2-20k) ─────────────────────────────────────────
    # Used only on preservation samples (ordinal/exclusion/direct/position)
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

    print("\nConfiguring trainability (LoRA + connector)...")
    configure_trainability_v5(policy, raw)

    n_train = sum(p.numel() for p in policy.parameters() if p.requires_grad)
    n_total = sum(p.numel() for p in policy.parameters())
    print(f"  Total: {n_total:,}  |  Trainable: {n_train:,} ({100*n_train/n_total:.2f}%)")

    policy = policy.to(device)
    raw    = policy.get_base_model().model

    student_cap = VelocityCapture()
    student_cap.register(raw.action_out_proj)

    # ── Optimizer ─────────────────────────────────────────────────────────────
    print("\nBuilding optimizer param groups...")
    param_groups = build_param_groups_v5(policy, raw, args.lora_lr, args.connector_lr)
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
    # Vision encoder stays frozen in eval mode
    raw.vlm_with_expert.get_vlm_model().vision_model.eval()
    # Connector stays in TRAIN mode intentionally

    print(f"\nTraining D2_v5: {args.steps} steps | batch={args.batch_size} | beta={args.beta} | AMP={use_bf16}")
    print(f"  Distillation: preservation samples only (relative=no, preserve=yes)")
    print(f"Output: {out_dir}\n")

    t0        = time.time()
    loss_ema  = None
    ema_alpha = 0.1
    all_params = [p for p in policy.parameters() if p.requires_grad]

    for step in range(1, args.steps + 1):
        update_lr_v5(optimizer, step,
                     warmup=args.warmup_steps, total=args.steps,
                     lora_base=args.lora_lr,           connector_base=args.connector_lr,
                     lora_min=args.lora_min_lr,         connector_min=args.connector_min_lr,
                     hold_until=args.hold_steps)

        # ── Build batch ──────────────────────────────────────────────────────
        samples = sampler.next_batch(args.batch_size)
        items   = [fetch_item(ds, rec, prompt) for rec, _, prompt in samples]
        scale_images(items, ds.meta.camera_keys)
        batch = collate_batch(items)
        batch = preprocessor(batch)

        # ── Family masks ─────────────────────────────────────────────────────
        # preserve_mask: True for ordinal/exclusion/direct/position (distillation applied)
        # relative_mask: True for relative (no distillation)
        families = [fam for _, fam, _ in samples]
        preserve_mask = torch.tensor(
            [fam in PRESERVATION_FAMILIES for fam in families],
            dtype=torch.bool, device=device
        )
        n_preserve = int(preserve_mask.sum().item())
        n_relative = len(families) - n_preserve

        # ── Pre-generate shared noise/time for distillation ──────────────────
        B     = args.batch_size
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

        # ── Teacher forward + selective distillation ─────────────────────────
        # Teacher runs only when we have preservation samples AND beta > 0
        loss_distill = torch.tensor(0.0, device=device)
        if teacher_policy is not None and n_preserve > 0:
            teacher_cap.clear()
            with torch.no_grad(), autocast_ctx:
                teacher_policy.forward(batch, noise=noise, time=time_)
            v_student = student_cap.vel   # [B, T, D]
            v_teacher = teacher_cap.vel   # [B, T, D]
            if v_student is not None and v_teacher is not None:
                # Per-sample MSE → apply only to preservation samples
                per_sample = F.mse_loss(
                    v_student.float(),
                    v_teacher.float().detach(),
                    reduction="none",
                ).mean(dim=list(range(1, v_student.dim())))  # [B]
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
            elapsed      = time.time() - t0
            sps          = step / max(elapsed, 1e-6)
            fam_cnt      = Counter(families)
            lora_lr      = optimizer.param_groups[0]["lr"]
            connector_lr = optimizer.param_groups[1]["lr"]
            print(
                f"step={step:5d} loss={loss_val:.4f}(ema={loss_ema:.4f}) "
                f"act={float(loss_action):.4f} distill={float(loss_distill):.4f} "
                f"[rel={n_relative} pres={n_preserve}] "
                f"grad={float(grad_norm):.3f} "
                f"lr_lora={lora_lr:.1e} lr_conn={connector_lr:.1e} "
                f"sps={sps:.1f} fam={dict(fam_cnt)}",
                flush=True,
            )
            if wandb_run is not None:
                import wandb
                wandb.log({
                    "loss":              loss_val,
                    "loss_ema":          loss_ema,
                    "loss_action":       float(loss_action),
                    "loss_distill":      float(loss_distill),
                    "n_relative":        n_relative,
                    "n_preserve":        n_preserve,
                    "grad_norm":         float(grad_norm),
                    "lr_lora":           lora_lr,
                    "lr_connector":      connector_lr,
                    "steps_per_s":       sps,
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
