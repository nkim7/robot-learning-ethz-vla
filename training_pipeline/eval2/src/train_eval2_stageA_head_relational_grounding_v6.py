#!/usr/bin/env python3
"""Eval2 Stage A v6: clockwise-convention hard-case fine-tune from v5@300.

Audit finding (v4@750 – v5@500)
--------------------------------
All 6 checkpoints fail on exactly 2 bgr val cases, both from episode 10 rep 4:

  Case 1: right_of / blue   → expected middle_green  (model→right_red,  prob=0.67–0.82)
  Case 2: right_of / green  → expected right_red     (model→middle_blue, prob=0.60–0.69)

In that episode's bowl arrangement the green bowl is placed at TOP-CENTER, so the
model's image-x-axis reasoning maps "right of blue" → red (the rightmost bowl in
the image).  The correct answer follows the CLOCKWISE bowl-order convention:

  bgr layout:  blue (left) → green (middle) → red (right)  [clockwise]
  right_of_blue  = next clockwise from blue  = green  ✓
  right_of_green = next clockwise from green = red    ✓

The model uses visual-x-axis shortcut; v6 teaches it to use clockwise slot order.

v6 changes
----------
1. New prompt family: `clockwise_convention`
   Four scaffold prompts per (frame, side, anchor) pair that explicitly state:
   - the full clockwise bowl sequence (e.g. "blue → green → red")
   - that 'right_of' means NEXT clockwise, not rightmost in the image
   - the resolved target colour
   Generated for ALL layouts (teaches general convention) and added to both
   train and val sets (val convention cases are informational, not gate criteria).

2. Hard-case loss weighting (normalised so batch mean weight = 1.0):
   - clockwise_conflict:  bgr or gbr layout with right_of raw_relative  → 4.0×
   - clockwise_convention cases (any layout)                             → 3.0×
   - bgr or gbr (any side, any non-convention family)                   → 2.0×
   - all other cases                                                     → 1.0×

3. Family mix (v6):
   raw_relative=0.40 | clockwise_convention=0.30 | scaffolded_relative=0.20
   semi_resolved=0.05 | simple=0.05

4. Loss formula (unchanged from v5):
   1.0 × CE(pair)  +  0.2 × CE(slot)  +  0.2 × CE(color)   [per-sample weighted]

5. Evaluation & gate (unchanged from v5): pair-primary, 95% threshold.

6. Init: v5@300 (only 1 gate criterion failing: bgr=91.7%).
   LR: lora=2e-6, head=2e-5.  Steps: 200.  Eval every 50.

Stage B gate (pair-primary, all must reach 95%):
  raw_relative pair_acc              >= 95%
  raw_anti_shortcut pair_acc         >= 95%
  every layout raw pair_acc          >= 95%
  every anchor raw pair_acc          >= 95%
  left_of / right_of raw pair_acc    >= 95%

Do NOT start Stage B until STAGE_B_GATE_PASS.
"""

from __future__ import annotations

import argparse
import json
import random
import time
from collections import Counter, defaultdict
from pathlib import Path

import torch
import torch.nn.functional as F
from peft import LoraConfig, PeftModel, get_peft_model
from torch import nn
from torch.utils.data import DataLoader, Dataset

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies.factory import make_pre_post_processors
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
from lerobot.utils.constants import ACTION

from train_eval2_stageA_head_relational_grounding import (
    COLORS, COLOR_TO_ID, ID_TO_COLOR, ID_TO_SLOT, SLOT_TO_ID, SLOTS,
    FrameRecord, StageACase,
    BalancedStageASampler, configure_trainability, copy_processors,
    first_frames, layout_id, load_episode_truth, lr_lambda,
    raw_relative_prompt, relative_targets, scaffolded_prompts,
    semi_resolved_prompts, simple_prompts, summarize_cases,
    vlm_last_text_hidden,
)
from train_eval2_stageA_head_relational_grounding_v4 import (
    PAIRS, PAIR_TO_ID, ID_TO_PAIR,
    TargetHeadsV4, collate_v4,
)


def parse_family_weights_v6(s: str) -> dict:
    """Parse 'key=val,...' string; must define all 5 v6 families."""
    REQUIRED = {"raw_relative", "clockwise_convention", "scaffolded_relative",
                "semi_resolved", "simple"}
    out = {}
    for part in s.split(","):
        k, v = part.strip().split("=")
        out[k.strip()] = float(v.strip())
    missing = REQUIRED - out.keys()
    extra   = out.keys() - REQUIRED
    if missing or extra:
        raise ValueError(
            f"family weights must define exactly {sorted(REQUIRED)}; "
            f"missing={sorted(missing)}, extra={sorted(extra)}"
        )
    return out
from train_eval2_stageA_head_relational_grounding_v5 import (
    evaluate_v5, check_stage_b_gate_v5, print_eval_summary_v5,
    STAGE_B_GATE_THRESHOLD, WEAK_LOSS_WEIGHT, NORMAL_LOSS_WEIGHT,
)

# ---------------------------------------------------------------------------
# v6 defaults
# ---------------------------------------------------------------------------
DEFAULT_DATASET = "robot-learning-group47/eval2_fixed_bowl_fixed_banana_108"
DEFAULT_DATASET_ROOT = (
    "/home/shadeform/lerobot_cache/"
    "robot-learning-group47/eval2_fixed_bowl_fixed_banana_108"
)
# v5@300 — only bgr failing (1 criterion), all others pass
DEFAULT_INIT = (
    "/home/shadeform/outputs/"
    "eval2_stageA_head_relational_grounding_v5_pairprimary_from_v4_750/"
    "checkpoints/000300/pretrained_model"
)
DEFAULT_BASE_INIT = (
    "/home/shadeform/outputs/"
    "eval2_stage2_compositional_015000_balanced_r2/checkpoints/020000/pretrained_model"
)
DEFAULT_OUT = (
    "/home/shadeform/outputs/"
    "eval2_stageA_head_relational_grounding_v6_clockwise_from_v5_300"
)

FAMILY_WEIGHTS_V6 = {
    "raw_relative":        0.40,
    "clockwise_convention": 0.30,
    "scaffolded_relative": 0.20,
    "semi_resolved":       0.05,
    "simple":              0.05,
}

# Loss weights for hard cases (normalised to batch mean = 1.0)
CLOCKWISE_CONFLICT_WEIGHT = 4.0   # bgr/gbr right_of raw
CONVENTION_WEIGHT         = 3.0   # clockwise_convention prompts
BGRxGBR_WEIGHT            = 2.0   # other bgr/gbr cases
NORMAL_WEIGHT             = 1.0   # everything else


# ---------------------------------------------------------------------------
# Clockwise convention scaffold prompts
# ---------------------------------------------------------------------------

def generate_clockwise_convention_prompts(
    side: str,
    anchor: str,
    target: str,
    layout: tuple[str, str, str],
) -> list[str]:
    """Generate 4 explicit clockwise-convention scaffold prompts.

    These teach the model that right_of = NEXT clockwise in the bowl sequence,
    NOT the bowl furthest to the right in the image.

    layout = (left_color, middle_color, right_color)
    Clockwise sequence: layout[0] → layout[1] → layout[2] → layout[0]
    """
    order_str = f"{layout[0]} → {layout[1]} → {layout[2]}"
    side_str = "right of" if side == "right_of" else "left of"
    if side == "right_of":
        direction     = "clockwise"
        direction_op  = "next"
    else:
        direction     = "counterclockwise"
        direction_op  = "previous"

    return [
        # Prompt 1: explicit sequence + next-in-order resolution
        (
            f"Read the bowls in clockwise order around the arc: {order_str}. "
            f"The bowl {direction_op} from {anchor} in this order is {target}. "
            f"Put the banana into the {target} bowl."
        ),
        # Prompt 2: clockwise order ≠ horizontal image position
        (
            f"Use the clockwise bowl order, not the horizontal image position. "
            f"Clockwise order: {order_str}. "
            f"The bowl to the {side_str} of {anchor} in clockwise order is {target}. "
            f"Put the banana into the {target} bowl."
        ),
        # Prompt 3: treat right_of / left_of as next / previous clockwise
        (
            f"Treat '{side_str}' as the {direction_op} bowl {direction} from the anchor bowl. "
            f"Starting at {anchor}, the {direction_op} bowl {direction} is {target}. "
            f"Put the banana into the {target} bowl."
        ),
        # Prompt 4: full sequence + slot-based reasoning
        (
            f"The bowls follow the clockwise sequence: {order_str}. "
            f"'{side_str} {anchor}' means the bowl that comes {direction_op} after {anchor} "
            f"in the clockwise sequence. That bowl is {target}. Place the banana there."
        ),
    ]


# ---------------------------------------------------------------------------
# Clockwise conflict and weight detection
# ---------------------------------------------------------------------------

def _get_loss_weight_v6(case: StageACase) -> float:
    """Per-sample loss weight for v6.

    Priority order (highest wins):
      clockwise_conflict (bgr/gbr + right_of + raw)  → 4.0×
      clockwise_convention (any layout)               → 3.0×
      bgr or gbr (other families)                     → 2.0×
      everything else                                  → 1.0×
    """
    is_bgr_gbr = case.layout_code in ("bgr", "gbr")

    # Hardest: raw right_of in bgr/gbr (the visually ambiguous cases)
    if (is_bgr_gbr and case.side == "right_of"
            and case.family == "raw_relative"):
        return CLOCKWISE_CONFLICT_WEIGHT

    # Convention scaffold prompts (teach the clockwise rule)
    if case.family == "clockwise_convention":
        return CONVENTION_WEIGHT

    # Medium: other bgr/gbr cases
    if is_bgr_gbr:
        return BGRxGBR_WEIGHT

    return NORMAL_WEIGHT


# ---------------------------------------------------------------------------
# Case builder v6 (adds clockwise_convention family)
# ---------------------------------------------------------------------------

def build_cases_v6(
    frames: list[FrameRecord],
    train_reps: set[int],
    val_reps: set[int],
) -> list[StageACase]:
    """Build all training and validation cases including clockwise_convention."""
    cases: list[StageACase] = []

    def _split(frame: FrameRecord) -> str | None:
        if frame.replicate_index in train_reps:
            return "train"
        if frame.replicate_index in val_reps:
            return "val"
        return None

    for frame_id, frame in enumerate(frames):
        split = _split(frame)
        if split is None:
            continue
        lid = layout_id(frame.layout)

        for rel in relative_targets(frame.layout):
            key    = f"{rel['side']}_{rel['anchor_color']}"
            side   = rel["side"]
            anchor = rel["anchor_color"]
            tslot  = rel["target_slot"]
            tcolor = rel["target_color"]
            anti   = bool(rel["anti_shortcut"])

            def _case(family: str, prompt: str) -> StageACase:
                return StageACase(
                    frame_id=frame_id, split=split, family=family,
                    prompt=prompt,
                    target_slot=tslot, target_color=tcolor,
                    layout_code=frame.layout_code, layout_id=lid,
                    side=side, anchor_color=anchor,
                    relation_key=key, anti_shortcut=anti,
                )

            # --- raw_relative (original single template) ---
            cases.append(_case("raw_relative", raw_relative_prompt(side, anchor)))

            # --- clockwise_convention (4 prompts per relation, ALL layouts) ---
            for prompt in generate_clockwise_convention_prompts(
                side, anchor, tcolor, frame.layout
            ):
                cases.append(_case("clockwise_convention", prompt))

            # --- scaffolded_relative (5 reasoning templates, skip [0]) ---
            for prompt in scaffolded_prompts(side, anchor, tslot, tcolor)[1:]:
                cases.append(_case("scaffolded_relative", prompt))

        # --- semi_resolved and simple (unchanged from v4/v5) ---
        for slot_idx, slot in enumerate(SLOTS):
            color = frame.layout[slot_idx]
            for prompt in semi_resolved_prompts(slot, color):
                cases.append(StageACase(
                    frame_id=frame_id, split=split, family="semi_resolved",
                    prompt=prompt, target_slot=slot, target_color=color,
                    layout_code=frame.layout_code, layout_id=lid,
                ))
            for prompt, mode in simple_prompts(slot, color):
                cases.append(StageACase(
                    frame_id=frame_id, split=split, family="simple",
                    prompt=prompt, target_slot=slot, target_color=color,
                    layout_code=frame.layout_code, layout_id=lid,
                    relation_key=mode,
                ))

    return cases


# ---------------------------------------------------------------------------
# Dataset v6 (reuses V4 infrastructure, adds clockwise_convention pool)
# ---------------------------------------------------------------------------

class StageADatasetV6(Dataset):
    """V4 dataset extended with clockwise_convention family."""

    def __init__(
        self,
        base: LeRobotDataset,
        frames: list[FrameRecord],
        cases: list[StageACase],
        split: str,
        family_weights: dict[str, float] | None = None,
    ):
        self.base = base
        self.frames = frames
        self.cases = [c for c in cases if c.split == split]
        if not self.cases:
            raise ValueError(f"No cases for split={split!r}")
        self.family_weights = family_weights or FAMILY_WEIGHTS_V6
        # Slot-indexed pools for BalancedStageASampler
        self.pools: dict[str, dict[str, list[int]]] = {
            family: {slot: [] for slot in SLOTS}
            for family in self.family_weights
        }
        for idx, case in enumerate(self.cases):
            if case.family in self.pools:
                self.pools[case.family][case.target_slot].append(idx)
        missing = [
            f"{fam}/{slot}"
            for fam, by_slot in self.pools.items()
            for slot, rows in by_slot.items()
            if not rows
        ]
        if missing:
            raise ValueError(f"Missing sampler pools: {missing}")

    def __len__(self) -> int:
        return len(self.cases)

    def __getitem__(self, idx: int) -> dict:
        case  = self.cases[idx]
        frame = self.frames[case.frame_id]
        item  = dict(self.base[frame.global_index])
        item.pop(ACTION, None)
        item.pop("action_is_pad", None)
        item["task"] = case.prompt
        item["target_slot_id"]  = torch.tensor(SLOT_TO_ID[case.target_slot],  dtype=torch.long)
        item["target_color_id"] = torch.tensor(COLOR_TO_ID[case.target_color], dtype=torch.long)
        pair_key = f"{case.target_slot}_{case.target_color}"
        item["target_pair_id"]  = torch.tensor(PAIR_TO_ID[pair_key], dtype=torch.long)
        item["_case_index"]     = torch.tensor(idx, dtype=torch.long)
        return item


# ---------------------------------------------------------------------------
# Checkpoint save (v6)
# ---------------------------------------------------------------------------

def save_ckpt_v6(
    out_dir: Path, step: int, policy, heads: TargetHeadsV4,
    init_path: Path, train_report: dict,
) -> Path:
    ckpt = out_dir / "checkpoints" / f"{step:06d}" / "pretrained_model"
    ckpt.mkdir(parents=True, exist_ok=True)
    policy.save_pretrained(ckpt)
    policy.config.save_pretrained(ckpt)
    copy_processors(init_path, ckpt)
    torch.save(heads.state_dict(), ckpt / "stageA_v4_target_heads.pt")
    (ckpt / "stageA_v6_metadata.json").write_text(
        json.dumps(train_report, indent=2) + "\n"
    )
    return ckpt


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Stage A v6: clockwise-convention hard-case fine-tune from v5@300"
    )
    p.add_argument("--init",          default=DEFAULT_INIT)
    p.add_argument("--base-init",     default=DEFAULT_BASE_INIT)
    p.add_argument("--dataset",       default=DEFAULT_DATASET)
    p.add_argument("--dataset-root",  type=Path, default=Path(DEFAULT_DATASET_ROOT))
    p.add_argument("--out-dir",       type=Path, default=Path(DEFAULT_OUT))
    p.add_argument("--steps",         type=int,   default=200)
    p.add_argument("--batch-size",    type=int,   default=64)
    p.add_argument("--eval-batch-size", type=int, default=64)
    p.add_argument("--lora-lr",       type=float, default=2e-6)
    p.add_argument("--head-lr",       type=float, default=2e-5)
    p.add_argument("--weight-decay",  type=float, default=1e-10)
    p.add_argument("--warmup-steps",  type=int,   default=20)
    p.add_argument("--decay-start-step", type=int, default=130)
    p.add_argument("--save-every",    type=int,   default=50)
    p.add_argument("--eval-every",    type=int,   default=50)
    p.add_argument("--log-every",     type=int,   default=10)
    p.add_argument("--lora-rank",     type=int,   default=16)
    p.add_argument("--lora-alpha",    type=int,   default=32)
    p.add_argument("--lora-dropout",  type=float, default=0.05)
    p.add_argument("--head-dropout",  type=float, default=0.05)
    p.add_argument("--grad-clip-norm", type=float, default=10.0)
    p.add_argument("--num-workers",   type=int,   default=8)
    p.add_argument(
        "--family-weights", default="",
        help="Override family weights, e.g. raw_relative=0.40,...",
    )
    p.add_argument("--clockwise-conflict-weight", type=float, default=CLOCKWISE_CONFLICT_WEIGHT)
    p.add_argument("--convention-weight",         type=float, default=CONVENTION_WEIGHT)
    p.add_argument("--bgr-gbr-weight",            type=float, default=BGRxGBR_WEIGHT)
    p.add_argument("--train-reps", default="0,1,2,3")
    p.add_argument("--val-reps",   default="4,5")
    p.add_argument("--seed",       type=int, default=52)
    p.add_argument("--use-amp", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--dry-inspect", action="store_true")
    p.add_argument("--wandb", action="store_true")
    p.add_argument("--wandb-project", default="lerobot")
    p.add_argument("--wandb-name", default="eval2-stageA-v6-clockwise-from-v5-300")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device     = "cuda" if torch.cuda.is_available() else "cpu"
    init_path  = Path(args.init)
    base_init  = Path(args.base_init)
    train_reps = {int(x) for x in args.train_reps.split(",") if x}
    val_reps   = {int(x) for x in args.val_reps.split(",") if x}

    if args.family_weights:
        family_weights = parse_family_weights_v6(args.family_weights)
    else:
        family_weights = dict(FAMILY_WEIGHTS_V6)

    # ------------------------------------------------------------------
    # Data
    # ------------------------------------------------------------------
    truth   = load_episode_truth(args.dataset, args.dataset_root)
    base_ds = LeRobotDataset(args.dataset, root=args.dataset_root, video_backend="pyav")
    frames  = first_frames(base_ds, truth)
    all_cases = build_cases_v6(frames, train_reps, val_reps)

    # Report case counts
    train_cc = sum(1 for c in all_cases if c.split == "train" and c.family == "clockwise_convention")
    val_cc   = sum(1 for c in all_cases if c.split == "val"   and c.family == "clockwise_convention")
    print(f"Cases built:  train={sum(1 for c in all_cases if c.split=='train')}  "
          f"val={sum(1 for c in all_cases if c.split=='val')}", flush=True)
    print(f"  clockwise_convention: train={train_cc}  val={val_cc}", flush=True)

    # Weight distribution diagnostics
    train_cases = [c for c in all_cases if c.split == "train"]
    w_conflict   = sum(1 for c in train_cases
                       if c.layout_code in ("bgr","gbr") and c.side == "right_of"
                       and c.family == "raw_relative")
    w_convention = sum(1 for c in train_cases if c.family == "clockwise_convention")
    w_bgr_gbr    = sum(1 for c in train_cases
                       if c.layout_code in ("bgr","gbr") and c.family != "clockwise_convention"
                       and not (c.side == "right_of" and c.family == "raw_relative"))
    print(f"  loss weight groups (train cases): conflict={w_conflict}×{args.clockwise_conflict_weight}  "
          f"convention={w_convention}×{args.convention_weight}  "
          f"bgr_gbr={w_bgr_gbr}×{args.bgr_gbr_weight}  "
          f"normal={len(train_cases)-w_conflict-w_convention-w_bgr_gbr}×1.0",
          flush=True)

    # ------------------------------------------------------------------
    # Policy — load v5@300 adapter
    # ------------------------------------------------------------------
    init_is_adapter = (init_path / "adapter_config.json").exists()
    if init_is_adapter:
        print(f"Loading Stage A adapter (v5@300): {init_path}", flush=True)
        base_policy = SmolVLAPolicy.from_pretrained(str(base_init))
        base_policy.config.device = device
        policy = PeftModel.from_pretrained(base_policy, str(init_path), is_trainable=True)
    else:
        policy = SmolVLAPolicy.from_pretrained(str(init_path))
    policy.config.device = device
    policy.config.train_expert_only = True
    policy.config.freeze_vision_encoder = True

    if not init_is_adapter:
        lora_cfg = LoraConfig(
            r=args.lora_rank, lora_alpha=args.lora_alpha,
            target_modules=(
                r"model\.vlm_with_expert\.vlm\.model\.text_model\.layers"
                r"\.[0-9]+\.self_attn\.(q|v)_proj"
            ),
            lora_dropout=args.lora_dropout, bias="none",
        )
        policy = get_peft_model(policy, lora_cfg)

    configure_trainability(policy)
    policy = policy.to(device)

    raw_policy  = policy.get_base_model() if hasattr(policy, "get_base_model") else policy
    hidden_size = raw_policy.model.vlm_with_expert.config.text_config.hidden_size

    # Heads: warm-start all 3 from v5@300 stageA_v4_target_heads.pt
    heads = TargetHeadsV4(hidden_size, dropout=args.head_dropout).to(device)
    heads_path = init_path / "stageA_v4_target_heads.pt"
    if heads_path.exists():
        state = torch.load(heads_path, map_location=device)
        miss, unexp = heads.load_state_dict(state, strict=False)
        print(f"Heads warm-started from v5@300: missing={miss}  unexpected={unexp}", flush=True)
    else:
        print(f"WARNING: no heads file at {heads_path}; all randomly init.", flush=True)

    lora_params = [p for n, p in policy.named_parameters() if p.requires_grad and "lora_" in n]
    head_params = list(heads.parameters())

    report = {
        "stage":   "A-v6-clockwise-convention",
        "version": "v6",
        "purpose": (
            "Clockwise-convention fine-tune. "
            "Targets bgr/gbr right_of failures where visual x-axis conflicts "
            "with clockwise bowl-order convention. "
            "New clockwise_convention prompts teach the model to use "
            "slot-adjacency (next clockwise) not image-x-axis."
        ),
        "init":    str(init_path),
        "audit_finding": {
            "persistent_failures": [
                {"episode": 10, "rep": 4, "side": "right_of", "anchor": "blue",
                 "expected": "middle_green", "model_predicts": "right_red"},
                {"episode": 10, "rep": 4, "side": "right_of", "anchor": "green",
                 "expected": "right_red",   "model_predicts": "middle_blue"},
            ],
            "root_cause": (
                "Episode 10 bgr frame: green bowl at top-center. "
                "Model uses visual-x-axis → red. "
                "Convention requires clockwise slot-adjacency → green."
            ),
        },
        "new_family": {
            "name": "clockwise_convention",
            "prompts_per_relation": 4,
            "layouts": "all",
            "purpose": "Explicit clockwise-sequence scaffold",
        },
        "loss_weights": {
            "clockwise_conflict (bgr/gbr right_of raw)": args.clockwise_conflict_weight,
            "clockwise_convention (any layout)":          args.convention_weight,
            "bgr_gbr (other)":                           args.bgr_gbr_weight,
            "normal":                                     1.0,
            "normalization": "weights / batch_mean → mean weight = 1.0",
        },
        "loss_formula": "1.0×CE(pair) + 0.2×CE(slot) + 0.2×CE(color)  [per-sample weighted]",
        "family_weights": family_weights,
        "evaluation": "pair-primary: both_ok = pair_pred == target_pair",
        "gate_threshold": STAGE_B_GATE_THRESHOLD,
        "hparams": {
            "steps": args.steps, "batch_size": args.batch_size,
            "lora_lr": args.lora_lr, "head_lr": args.head_lr,
            "warmup_steps": args.warmup_steps,
            "decay_start_step": args.decay_start_step,
            "seed": args.seed,
        },
        "trainable": {
            "lora": sum(p.numel() for p in lora_params),
            "heads": sum(p.numel() for p in head_params),
            "total": sum(p.numel() for p in lora_params + head_params),
        },
    }

    print("STAGE_A_V6_CONSTRUCTION_REPORT", flush=True)
    print(json.dumps(report, indent=2), flush=True)
    print(f"CLOCKWISE_CONFLICT_WEIGHT={args.clockwise_conflict_weight}  "
          f"CONVENTION_WEIGHT={args.convention_weight}  "
          f"BGRxGBR_WEIGHT={args.bgr_gbr_weight}", flush=True)
    print(f"EVAL_MODE: pair-primary  (both_ok = pair_correct)", flush=True)

    if args.dry_inspect:
        # Show sample convention prompts
        bgr_frames = [f for f in frames if f.layout_code == "bgr"]
        if bgr_frames:
            f0 = bgr_frames[0]
            print(f"\nSAMPLE CLOCKWISE_CONVENTION PROMPTS (bgr frame ep={f0.episode_index}):", flush=True)
            for prompt in generate_clockwise_convention_prompts(
                "right_of", f0.layout[0], f0.layout[1], f0.layout
            ):
                print(f"  • {prompt}", flush=True)
        print("DRY_INSPECT_DONE", flush=True)
        return

    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "stageA_v6_construction_report.json").write_text(
        json.dumps(report, indent=2) + "\n"
    )

    # ------------------------------------------------------------------
    # Datasets and sampler
    # ------------------------------------------------------------------
    train_ds = StageADatasetV6(base_ds, frames, all_cases, "train", family_weights)
    val_ds   = StageADatasetV6(base_ds, frames, all_cases, "val",   family_weights)

    # Use StageADatasetV6 as the base for evaluate_v5 (it expects .cases and .base)
    sampler = BalancedStageASampler(
        train_ds, args.batch_size, args.steps,
        family_weights=family_weights, seed=args.seed,
    )
    loader = DataLoader(
        train_ds, batch_sampler=sampler,
        num_workers=args.num_workers, collate_fn=collate_v4,
        pin_memory=(device == "cuda"),
    )

    preprocessor, _ = make_pre_post_processors(
        policy.config,
        pretrained_path=str(init_path),
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

    # ------------------------------------------------------------------
    # Optimiser and scheduler
    # ------------------------------------------------------------------
    opt = torch.optim.AdamW(
        [
            {"params": lora_params, "lr": args.lora_lr},
            {"params": head_params, "lr": args.head_lr},
        ],
        weight_decay=args.weight_decay, betas=(0.9, 0.95), eps=1e-8,
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        opt,
        lr_lambda=lambda step: lr_lambda(
            step, args.warmup_steps, args.decay_start_step, args.steps
        ),
    )
    scaler = torch.amp.GradScaler("cuda", enabled=False)

    # ------------------------------------------------------------------
    # WandB
    # ------------------------------------------------------------------
    wandb_run = None
    if args.wandb:
        import wandb as _wandb
        wandb_run = _wandb.init(
            project=args.wandb_project, name=args.wandb_name, config=vars(args),
        )
        report["wandb_url"] = getattr(wandb_run, "url", None)
        print(f"WANDB_URL={report['wandb_url']}", flush=True)

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------
    policy.train()
    heads.train()
    start = time.time()

    for step, batch in enumerate(loader, start=1):
        case_indices = batch.pop("_case_index").tolist()
        target_slot  = batch.pop("target_slot_id").to(device)
        target_color = batch.pop("target_color_id").to(device)
        target_pair  = batch.pop("target_pair_id").to(device)

        for key in list(base_ds.meta.camera_keys):
            if key in batch and batch[key].dtype == torch.uint8:
                batch[key] = batch[key].float() / 255.0
        batch = preprocessor(batch)

        with torch.autocast(device_type="cuda", dtype=torch.bfloat16,
                            enabled=args.use_amp and device == "cuda"):
            hidden = vlm_last_text_hidden(policy, batch)
            slot_logits, color_logits, pair_logits = heads(hidden)

        slot_losses  = F.cross_entropy(slot_logits.float(),  target_slot,  reduction="none")
        color_losses = F.cross_entropy(color_logits.float(), target_color, reduction="none")
        pair_losses  = F.cross_entropy(pair_logits.float(),  target_pair,  reduction="none")

        # Per-sample weights
        raw_weights = torch.tensor(
            [_get_loss_weight_v6(train_ds.cases[int(i)]) for i in case_indices],
            dtype=torch.float32, device=device,
        )
        norm_weights = raw_weights / raw_weights.mean().clamp_min(1e-6)

        combined = 1.0 * pair_losses + 0.2 * slot_losses + 0.2 * color_losses
        loss      = (combined * norm_weights).mean()

        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        grad_norm = torch.nn.utils.clip_grad_norm_(
            lora_params + head_params, args.grad_clip_norm
        )
        scaler.step(opt)
        scaler.update()
        scheduler.step()

        with torch.no_grad():
            pair_acc  = (pair_logits.argmax(-1) == target_pair).float().mean()
            slot_acc  = (slot_logits.argmax(-1) == target_slot).float().mean()
            color_acc = (color_logits.argmax(-1) == target_color).float().mean()
            conflict_count = int(raw_weights.ge(args.clockwise_conflict_weight - 0.1).sum())
            convention_count = int(
                torch.tensor([
                    train_ds.cases[int(i)].family == "clockwise_convention"
                    for i in case_indices
                ]).sum()
            )

        if step == 1 or step % args.log_every == 0:
            lrs = scheduler.get_last_lr()
            print(
                f"step={step} loss={loss.item():.4f} "
                f"pair_acc={pair_acc.item()*100:.1f}% "
                f"slot_acc={slot_acc.item()*100:.1f}% "
                f"color_acc={color_acc.item()*100:.1f}% "
                f"lr=({lrs[0]:.1e},{lrs[1]:.1e}) "
                f"conflict={conflict_count} convention={convention_count} "
                f"grad={grad_norm.item():.2f}",
                flush=True,
            )
            if wandb_run:
                wandb_run.log({
                    "train/loss": loss.item(),
                    "train/pair_acc": pair_acc.item(),
                    "train/conflict_count": conflict_count,
                    "train/convention_count": convention_count,
                    "train/grad_norm": grad_norm.item(),
                    "train/lora_lr": lrs[0],
                }, step=step)

        # Checkpoint + eval
        if step % args.save_every == 0 or step == args.steps:
            ckpt = save_ckpt_v6(args.out_dir, step, policy, heads, init_path, {
                "step": step, "loss": loss.item(), "pair_acc": pair_acc.item()
            })
            print(f"Checkpoint saved: {ckpt}", flush=True)

        if step % args.eval_every == 0 or step == args.steps:
            # evaluate_v5 works on any dataset with .cases and .base attributes
            val_metrics = evaluate_v5(policy, heads, preprocessor, val_ds, args, device)
            gate = check_stage_b_gate_v5(val_metrics)

            print_eval_summary_v5(step, val_metrics)

            if gate["gate_pass"]:
                print(f"\nSTAGE_B_GATE_PASS  step={step}  ALL criteria >= 95%", flush=True)
            else:
                failing_str = "  ".join(
                    f"{k}={v*100:.1f}%"
                    for k, v in gate["failing_criteria"].items()
                )
                print(
                    f"\nSTAGE_B_GATE_FAIL  step={step}  "
                    f"{gate['n_failing']} criteria below 95%: "
                    f"{failing_str}  Do NOT start Stage B yet.",
                    flush=True,
                )

            print(f"EVAL_METRICS_JSON step={step}", flush=True)
            print(json.dumps(val_metrics, indent=2), flush=True)

            if wandb_run:
                raw = val_metrics["by_family"].get("raw_relative", {})
                w_log = {
                    "val/raw_relative_pair": raw.get("pair_acc", 0),
                    "val/gate_n_failing":    gate["n_failing"],
                }
                for code, d in val_metrics.get("raw_by_layout", {}).items():
                    w_log[f"val/raw_layout_{code}_pair"] = d.get("pair_acc", 0)
                wandb_run.log(w_log, step=step)

    if wandb_run:
        wandb_run.finish()
    print(f"\nStage A v6 training complete.", flush=True)


if __name__ == "__main__":
    main()
