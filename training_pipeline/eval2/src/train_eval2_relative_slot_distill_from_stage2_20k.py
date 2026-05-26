#!/usr/bin/env python3
"""Relative-prompt → slot-teacher distillation from eval2_stage2_20k.

Central idea
============
The model already knows left/middle/right slot prompts. The goal is NOT to
teach a new manipulation skill, but to teach this equivalence:

    student("Put the banana into the bowl left of green.")
    ≈
    frozen_teacher("Put the banana into the middle bowl.")

The teacher is the same checkpoint used as student init, but frozen.
The teacher always receives the RESOLVED SLOT prompt; the student receives
the RELATIVE prompt for the same observation.

Loss function
=============
    L_total =
        1.0 * L_A  (relative → slot teacher distillation)   ← dominant
      + 0.5 * L_B  (direct slot preservation)
      + 0.3 * L_D  (scaffold consistency)
      + 0.3 * L_C  (real demo action preservation)
      + 0.2 * L_slot_aux   (auxiliary slot classification head)
      + 0.2 * L_layout_aux (auxiliary layout classification head)

Sample types
============
  A: student(relative prompt)  ≈ teacher(resolved slot prompt)   [main]
  B: student(slot prompt)      ≈ teacher(same slot prompt)       [preservation]
  C: student(original prompt)  ≈ recorded demo action            [anti-forgetting]
  D: student(plain relative)   ≈ student(scaffold)               [consistency]

Batch mix (configurable):
  ~55% Type A  ~20% Type B  ~15% Type C  ~10% Type D

Training schedule
=================
  Stage R1 (semantic warmup): LoRA only, 500-1000 steps, 1e-5 LR
    → teaches relative prompt + image → target slot (aux heads)
  Stage R2 (slot action distillation): LoRA + small action expert unfreeze
    → teacher velocity distillation on action_out_proj, 1000-3000 steps

Anti-shortcut validation
========================
  Same phrase must map to DIFFERENT slots depending on visual layout.
  The model must read the image, not memorize phrase shortcuts.

Checkpoint paths (override via CLI for local dev vs remote training):
  --init:         /home/shadeform/lerobot_checkpoints/eval2_stage2_20k
  --dataset-root: /home/shadeform/lerobot_cache/robot-learning-group47/eval2_108_position_only
  --out-dir:      /home/shadeform/outputs/eval2_relative_slot_distill_r2
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
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from peft import LoraConfig, get_peft_model

# ──────────────────────────────────────────────────────────────────────────────
# Path setup — adjust for remote training environment
# ──────────────────────────────────────────────────────────────────────────────
sys.path.insert(0, "/home/shadeform")
sys.path.insert(0, "/home/shadeform/lerobot/src")

from lerobot.datasets.factory import resolve_delta_timestamps
from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
from lerobot.policies.factory import make_pre_post_processors
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

# ──────────────────────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────────────────────

LAYOUT_CODES  = ("bgr", "brg", "gbr", "grb", "rbg", "rgb")
LAYOUT_TO_ID  = {code: i for i, code in enumerate(LAYOUT_CODES)}
SLOT_NAMES    = ("left", "middle", "right")
SLOT_TO_ID    = {s: i for i, s in enumerate(SLOT_NAMES)}
COLOR_FULL    = {"b": "blue", "g": "green", "r": "red"}
COLORS        = ("blue", "green", "red")

MAX_ACTION_DIM = 32
CHUNK_SIZE     = 50

# Sample type tags
TYPE_A = "A_relative_slot_distill"   # student(relative) ≈ teacher(slot)
TYPE_B = "B_slot_preservation"       # student(slot)     ≈ teacher(slot)
TYPE_C = "C_demo_action"             # student(original) ≈ recorded action
TYPE_D = "D_scaffold_consistency"    # student(plain)    ≈ student(scaffold)

# Loss weights
WEIGHT_A        = 1.0
WEIGHT_B        = 0.5
WEIGHT_C        = 0.3
WEIGHT_D        = 0.3
WEIGHT_SLOT_AUX = 0.2
WEIGHT_LAYOUT   = 0.2

# Batch mix fractions (approximate)
FRAC_A = 0.55
FRAC_B = 0.20
FRAC_C = 0.15
FRAC_D = 0.10

DEFAULT_DATASET      = "robot-learning-group47/eval2_108_position_only"
DEFAULT_DATASET_ROOT = "/home/shadeform/lerobot_cache/robot-learning-group47/eval2_108_position_only"
DEFAULT_INIT         = "/home/shadeform/lerobot_checkpoints/eval2_stage2_20k"
DEFAULT_OUT          = "/home/shadeform/outputs/eval2_relative_slot_distill_r2"
COMBO_CSV            = "combo_episode_map_eval2_fixed_bowl_fixed_banana_108.csv"


# ──────────────────────────────────────────────────────────────────────────────
# Episode metadata
# ──────────────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class EpisodeMeta:
    episode_index:   int
    layout:          list      # e.g. ["blue", "red", "green"]
    layout_code:     str       # e.g. "brg"
    layout_id:       int       # 0–5
    target_color:    str
    target_slot:     str
    target_slot_idx: int       # 0=left, 1=middle, 2=right


def load_episode_meta(dataset_root: Path) -> dict[int, EpisodeMeta]:
    csv_path = dataset_root / COMBO_CSV
    ep_meta: dict[int, EpisodeMeta] = {}
    with csv_path.open() as f:
        for row in csv.DictReader(f):
            layout_code  = row["bowl_order_left_to_right"].strip().lower()
            layout       = [COLOR_FULL[ch] for ch in layout_code]
            target_color = row["target_bowl_color"].strip().lower()
            ti           = layout.index(target_color)
            lid          = LAYOUT_TO_ID.get(layout_code, 0)
            for ep in range(int(row["episode_start"]), int(row["episode_end"]) + 1):
                ep_meta[ep] = EpisodeMeta(
                    episode_index   = ep,
                    layout          = layout,
                    layout_code     = layout_code,
                    layout_id       = lid,
                    target_color    = target_color,
                    target_slot     = SLOT_NAMES[ti],
                    target_slot_idx = ti,
                )
    return ep_meta


# ──────────────────────────────────────────────────────────────────────────────
# Prompt generators
# ──────────────────────────────────────────────────────────────────────────────

def slot_prompt(slot_name: str) -> str:
    """Resolved slot prompt used by the TEACHER."""
    return f"Put the banana into the {slot_name} bowl."


def relative_prompts_for_slot(layout: list[str], target_slot_idx: int) -> list[str]:
    """All valid relative prompts whose resolution equals target_slot_idx.

    Returns prompts of the form:
      left_of_ref, right_of_ref, between_X_and_Y, not_X_and_not_Y
    """
    ti = target_slot_idx
    prompts = []

    # left_of_ref: target = ref_idx - 1
    # → ref_idx = ti + 1
    if ti + 1 <= 2:
        ref_color = layout[ti + 1]
        prompts.append(
            f"Put the banana into the bowl on the left of the {ref_color} bowl "
            f"from the robot perspective."
        )

    # right_of_ref: target = ref_idx + 1
    # → ref_idx = ti - 1
    if ti - 1 >= 0:
        ref_color = layout[ti - 1]
        prompts.append(
            f"Put the banana into the bowl on the right of the {ref_color} bowl "
            f"from the robot perspective."
        )

    # between: only valid for middle slot (ti == 1)
    if ti == 1:
        prompts.append(
            f"Put the banana into the bowl that is between the {layout[0]} "
            f"and {layout[2]} bowls."
        )
        prompts.append(
            f"Put the banana into the bowl that is between the {layout[2]} "
            f"and {layout[0]} bowls."
        )

    # not X and not Y: the target is the color not excluded
    excl = [c for i, c in enumerate(layout) if i != ti]
    prompts.append(
        f"Put the banana into the bowl that is not {excl[0]} and not {excl[1]}."
    )
    prompts.append(
        f"Put the banana into the bowl that is not {excl[1]} and not {excl[0]}."
    )

    return prompts


def scaffold_prompt(layout: list[str], target_slot_idx: int, relative_prompt: str) -> str:
    """Scaffold prompt: embeds layout + resolution before the command.

    Only used during training (Type D). Never used at inference.
    """
    slot_name = SLOT_NAMES[target_slot_idx]
    layout_str = ", ".join(layout)
    return (
        f"The bowls from left to right are {layout_str}. "
        f"The target bowl is the {slot_name} bowl. "
        f"Put the banana into the {slot_name} bowl."
    )


def slot_prompt_for_episode(meta: EpisodeMeta) -> str:
    """Resolved slot prompt for the ORIGINAL target of this episode."""
    return slot_prompt(meta.target_slot)


def direct_color_prompt(meta: EpisodeMeta) -> str:
    return f"Put the banana into the {meta.target_color} bowl."


def ordinal_prompt(meta: EpisodeMeta) -> str:
    ordinals = ("1st", "2nd", "3rd")
    return (
        f"Put the banana into the {ordinals[meta.target_slot_idx]} bowl "
        f"from the left from the robot perspective."
    )


def exclusion_prompt(meta: EpisodeMeta) -> str:
    excl = [c for c in COLORS if c != meta.target_color]
    return f"Put the banana into the bowl that is not {excl[0]} and not {excl[1]}."


# Anti-shortcut: for the same relative phrase, resolve via the provided layout
def resolve_relative_prompt_to_slot(prompt: str, layout: list[str]) -> Optional[int]:
    """Return the slot index (0/1/2) that a relative prompt resolves to.

    Returns None if the prompt cannot be resolved.
    """
    p = prompt.lower()
    # left_of / right_of
    for ref_idx, ref_color in enumerate(layout):
        if f"left of the {ref_color}" in p or f"left of {ref_color}" in p:
            t = ref_idx - 1
            return t if 0 <= t <= 2 else None
        if f"right of the {ref_color}" in p or f"right of {ref_color}" in p:
            t = ref_idx + 1
            return t if 0 <= t <= 2 else None
    # between
    if "between" in p:
        return 1  # only valid when outer bowls are mentioned → always middle
    # not X and not Y
    if "not " in p:
        excluded = []
        for c in COLORS:
            if f"not {c}" in p:
                excluded.append(c)
        remaining = [c for c in layout if c not in excluded]
        if len(remaining) == 1:
            return layout.index(remaining[0])
    return None


# ──────────────────────────────────────────────────────────────────────────────
# Frame records and frame pool
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class FrameRecord:
    global_idx:    int
    episode_index: int
    frame_index:   int
    meta:          EpisodeMeta


def build_frame_pool(
    ds:           LeRobotDataset,
    ep_meta:      dict[int, EpisodeMeta],
    phase2_start: int,
    phase2_end:   int,
) -> dict[str, list[FrameRecord]]:
    """Build frame pool keyed by layout_code.

    Only includes frames in the phase-2 window [phase2_start, phase2_end]
    within each episode — the window where the banana is grasped and the
    robot approaches the bowls without yet committing to a direction.
    """
    ep_vals = list(ds.hf_dataset["episode_index"])
    fr_vals = list(ds.hf_dataset["frame_index"])
    pool: dict[str, list[FrameRecord]] = {code: [] for code in LAYOUT_CODES}
    for gi, (ep_val, fr_val) in enumerate(zip(ep_vals, fr_vals)):
        ep = int(ep_val)
        fr = int(fr_val)
        if not (phase2_start <= fr <= phase2_end):
            continue
        meta = ep_meta.get(ep)
        if meta is None:
            continue
        pool[meta.layout_code].append(FrameRecord(gi, ep, fr, meta))
    sizes = {c: len(v) for c, v in pool.items()}
    print(f"  Phase-2 frame pool (frames {phase2_start}–{phase2_end}): {sizes}")
    empty = [c for c, v in sizes.items() if v == 0]
    if empty:
        print(f"  WARNING: empty pool for layouts {empty}. Widening frame window?")
    return pool


def anti_shortcut_prompts_for_slot(layout: list[str], target_slot_idx: int) -> list[str]:
    """Direction-conflict relative prompts: direction word contradicts naïve edge heuristic.

    These are the hardest cases — a shortcut learner relying only on the
    direction word ("left" → leftmost, "right" → rightmost) would be wrong.

      left  (ti=0): right_of_middle → left   (direction=right, result=leftmost)
      middle(ti=1): right_of_leftmost → mid  (direction=right, result=middle)
                    left_of_rightmost → mid  (direction=left,  result=middle)
      right (ti=2): left_of_middle → right   (direction=left,  result=rightmost)

    Also included: between / not-X-not-Y (always unambiguous direction-free).
    """
    ti = target_slot_idx
    out = []
    if ti == 0:
        # right of middle → left
        out.append(
            f"Put the banana into the bowl on the right of the {layout[1]} bowl "
            f"from the robot perspective."
        )
    elif ti == 1:
        # right of leftmost → middle
        out.append(
            f"Put the banana into the bowl on the right of the {layout[0]} bowl "
            f"from the robot perspective."
        )
        # left of rightmost → middle
        out.append(
            f"Put the banana into the bowl on the left of the {layout[2]} bowl "
            f"from the robot perspective."
        )
        # between (unambiguous, always middle when outer bowls are named)
        out.append(
            f"Put the banana into the bowl that is between the {layout[0]} "
            f"and {layout[2]} bowls."
        )
    elif ti == 2:
        # left of middle → right
        out.append(
            f"Put the banana into the bowl on the left of the {layout[1]} bowl "
            f"from the robot perspective."
        )
    # not-X-not-Y is always direction-free → include for all slots
    excl = [c for i, c in enumerate(layout) if i != ti]
    out.append(
        f"Put the banana into the bowl that is not {excl[0]} and not {excl[1]}."
    )
    return [p for p in out if p]


class RelativeDistillSampler:
    """Samples training examples across the four sample types.

    Within a batch (approximate):
      ~55% Type A  (relative → slot teacher)
      ~20% Type B  (slot preservation)
      ~15% Type C  (demo action)
      ~10% Type D  (scaffold consistency)

    Type A — balanced across left / middle / right (round-robin):
      Each call to next_batch draws equal numbers of left-, middle-, and
      right-target Type A samples so no slot is over-represented.

      Within each slot, 40% of samples use direction-conflict (anti-shortcut)
      prompts; 60% use the full relative-prompt set.  This prevents the model
      from learning "right → rightmost" or "left → leftmost" shortcuts.
    """

    # Fraction of Type A samples drawn from direction-conflict (anti-shortcut) prompts.
    ANTI_SHORTCUT_FRAC = 0.40

    def __init__(
        self,
        pool:   dict[str, list[FrameRecord]],
        seed:   int,
        frac_a: float = FRAC_A,
        frac_b: float = FRAC_B,
        frac_c: float = FRAC_C,
    ):
        self.rng    = random.Random(seed)
        self.frac_a = frac_a
        self.frac_b = frac_b
        self.frac_c = frac_c
        self.frac_d = max(0.0, 1.0 - frac_a - frac_b - frac_c)

        # ── Global pool (for B / C / D) ───────────────────────────────────
        self.pool  = {k: list(v) for k, v in pool.items()}
        self.ptrs  = {k: 0 for k in pool}
        self.codes = sorted(pool.keys())
        for k in self.pool:
            self.rng.shuffle(self.pool[k])

        # ── Per-slot pools for Type A (balanced left/middle/right) ────────
        # slot_pool[slot_idx][layout_code] = [FrameRecord, ...]
        self.slot_pool: dict[int, dict[str, list[FrameRecord]]] = {}
        for si in range(3):
            self.slot_pool[si] = {}
            for code, frames in pool.items():
                sf = [f for f in frames if f.meta.target_slot_idx == si]
                if sf:
                    lst = list(sf)
                    self.rng.shuffle(lst)
                    self.slot_pool[si][code] = lst
        self.slot_ptrs: dict[int, dict[str, int]] = {
            si: {code: 0 for code in d}
            for si, d in self.slot_pool.items()
        }
        # Round-robin slot counter for Type A
        self._slot_rr = 0

        # Sanity check
        for si in range(3):
            n = sum(len(v) for v in self.slot_pool[si].values())
            sname = SLOT_NAMES[si]
            if n == 0:
                print(f"  WARNING: no phase-2 frames for slot '{sname}' — "
                      f"Type A will skip this slot. Widen --phase2-end?")
            else:
                print(f"  Type A pool [{sname:6s}]: "
                      f"{sum(len(v) for v in self.slot_pool[si].values())} frames "
                      f"across {len(self.slot_pool[si])} layouts")

    # ── Internal helpers ──────────────────────────────────────────────────

    def _next_global(self) -> FrameRecord:
        """Draw next frame from the global (all-slots) pool, cycling layouts."""
        code   = self.codes[self.rng.randint(0, len(self.codes) - 1)]
        frames = self.pool[code]
        idx    = self.ptrs[code] % len(frames)
        rec    = frames[idx]
        self.ptrs[code] += 1
        if self.ptrs[code] % len(frames) == 0:
            self.rng.shuffle(frames)
            self.ptrs[code] = 0
        return rec

    def _next_for_slot(self, si: int) -> Optional[FrameRecord]:
        """Draw next frame from the per-slot pool, cycling layouts."""
        d = self.slot_pool.get(si, {})
        if not d:
            return None
        codes = list(d.keys())
        code  = codes[self.rng.randint(0, len(codes) - 1)]
        frames = d[code]
        idx    = self.slot_ptrs[si][code] % len(frames)
        rec    = frames[idx]
        self.slot_ptrs[si][code] += 1
        if self.slot_ptrs[si][code] % len(frames) == 0:
            self.rng.shuffle(frames)
            self.slot_ptrs[si][code] = 0
        return rec

    def _gen_type_a_prompt(self, rec: FrameRecord) -> tuple[str, str]:
        """Return (student_relative_prompt, teacher_slot_prompt) for a Type A sample.

        40% of the time: draw from direction-conflict (anti-shortcut) prompts.
        60% of the time: draw from the full relative-prompt set.
        Falls back to full set if direction-conflict set is empty.
        """
        si     = rec.meta.target_slot_idx
        layout = rec.meta.layout

        if self.rng.random() < self.ANTI_SHORTCUT_FRAC:
            opts = anti_shortcut_prompts_for_slot(layout, si)
        else:
            opts = []

        if not opts:
            opts = relative_prompts_for_slot(layout, si)

        # Last-resort fallback (should never happen for a valid episode)
        if not opts:
            opts = [slot_prompt(SLOT_NAMES[si])]

        return self.rng.choice(opts), slot_prompt(SLOT_NAMES[si])

    # ── Public interface ──────────────────────────────────────────────────

    def next_batch(self, n: int) -> list[tuple[FrameRecord, str, dict]]:
        """Return list of (record, type_tag, sample_info).

        Type A sample_info: student_prompt, teacher_prompt, target_slot_idx,
                            layout_id, _slot_name (for logging)
        Type B sample_info: student_prompt, teacher_prompt, target_slot_idx, layout_id
        Type C sample_info: student_prompt, target_slot_idx, layout_id
        Type D sample_info: student_prompt, scaffold_prompt, teacher_prompt,
                            target_slot_idx, layout_id
        """
        out = []
        for _ in range(n):
            roll = self.rng.random()

            # ── Type A: relative → slot teacher (balanced round-robin) ────
            if roll < self.frac_a:
                # Round-robin: left(0) → middle(1) → right(2) → left(0) → …
                si  = self._slot_rr % 3
                self._slot_rr += 1
                rec = self._next_for_slot(si)
                if rec is None:          # slot has no frames → fall through to B
                    rec = self._next_global()
                    si  = rec.meta.target_slot_idx
                rel_p, slot_p = self._gen_type_a_prompt(rec)
                out.append((rec, TYPE_A, {
                    "student_prompt":  rel_p,
                    "teacher_prompt":  slot_p,
                    "target_slot_idx": si,
                    "layout_id":       rec.meta.layout_id,
                    "_slot_name":      SLOT_NAMES[si],
                }))

            # ── Type B: slot preservation ─────────────────────────────────
            elif roll < self.frac_a + self.frac_b:
                rec       = self._next_global()
                slot_name = SLOT_NAMES[rec.meta.target_slot_idx]
                sp        = slot_prompt(slot_name)
                out.append((rec, TYPE_B, {
                    "student_prompt":  sp,
                    "teacher_prompt":  sp,
                    "target_slot_idx": rec.meta.target_slot_idx,
                    "layout_id":       rec.meta.layout_id,
                }))

            # ── Type C: demo action preservation ──────────────────────────
            elif roll < self.frac_a + self.frac_b + self.frac_c:
                rec      = self._next_global()
                demo_opts = [
                    slot_prompt_for_episode(rec.meta),
                    direct_color_prompt(rec.meta),
                    ordinal_prompt(rec.meta),
                    exclusion_prompt(rec.meta),
                ]
                out.append((rec, TYPE_C, {
                    "student_prompt":  self.rng.choice(demo_opts),
                    "target_slot_idx": rec.meta.target_slot_idx,
                    "layout_id":       rec.meta.layout_id,
                }))

            # ── Type D: scaffold consistency ──────────────────────────────
            else:
                rec  = self._next_global()
                opts = relative_prompts_for_slot(rec.meta.layout, rec.meta.target_slot_idx)
                if not opts:
                    # Degrade to Type B if no relative prompts exist
                    sp = slot_prompt(rec.meta.target_slot)
                    out.append((rec, TYPE_B, {
                        "student_prompt":  sp,
                        "teacher_prompt":  sp,
                        "target_slot_idx": rec.meta.target_slot_idx,
                        "layout_id":       rec.meta.layout_id,
                    }))
                    continue
                plain_p    = self.rng.choice(opts)
                scaffold_p = scaffold_prompt(rec.meta.layout, rec.meta.target_slot_idx, plain_p)
                out.append((rec, TYPE_D, {
                    "student_prompt":  plain_p,
                    "scaffold_prompt": scaffold_p,
                    "teacher_prompt":  slot_prompt(rec.meta.target_slot),
                    "target_slot_idx": rec.meta.target_slot_idx,
                    "layout_id":       rec.meta.layout_id,
                }))

        return out


# ──────────────────────────────────────────────────────────────────────────────
# Batch helpers
# ──────────────────────────────────────────────────────────────────────────────

def fetch_item(ds: LeRobotDataset, rec: FrameRecord, prompt: str) -> dict:
    item = dict(ds[rec.global_idx])
    item["task"] = prompt
    item.pop("task_index", None)
    return item


def collate_items(items: list[dict]) -> dict:
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
# Change 2 — Teacher slot audit
# ──────────────────────────────────────────────────────────────────────────────

def audit_teacher_slot_separation(
    teacher_policy,
    teacher_cap:    "VelocityCapture",
    ds:             LeRobotDataset,
    ep_meta:        dict,
    preprocessor,
    device:         torch.device,
    probe_frame:    int   = 15,
    threshold:      float = 0.05,
    n_frames:       int   = 6,
    audit_out_dir:  Optional[Path] = None,
) -> tuple[dict, set]:
    """Run the teacher on left / middle / right slot prompts for N reference frames.

    Measures pairwise velocity MSE between slots.  A slot is declared
    "verified" if its velocity is clearly separated from both other slots
    (min pairwise distance > threshold).

    Returns
    -------
    report : dict
        Per-pair and per-slot mean distances + verified flag.
    verified_slots : set[str]
        E.g. {"left", "middle", "right"} if all three pass, {"middle"} if only middle.
    """
    print("\n" + "="*60)
    print("  TEACHER SLOT AUDIT")
    print(f"  Probing at frame_index={probe_frame}, n_frames={n_frames}, threshold={threshold:.4f}")
    print("="*60)

    # ── Collect reference frames (one per layout per demo slot, up to n_frames) ──
    ref_frames: list[dict] = []
    ep_vals = list(ds.hf_dataset["episode_index"])
    fr_vals = list(ds.hf_dataset["frame_index"])
    seen_layout_slots: set = set()

    for gi, (ep_val, fr_val) in enumerate(zip(ep_vals, fr_vals)):
        if len(ref_frames) >= n_frames:
            break
        ep = int(ep_val)
        fr = int(fr_val)
        if fr != probe_frame:
            continue
        meta = ep_meta.get(ep)
        if meta is None:
            continue
        key = (meta.layout_code, meta.target_slot_idx)
        if key in seen_layout_slots:
            continue
        seen_layout_slots.add(key)
        ref_frames.append({"gi": gi, "meta": meta})

    if not ref_frames:
        print("  [AUDIT] No reference frames found at probe_frame — skipping audit.")
        print("  [AUDIT] All slots will be treated as VERIFIED (unsafe, check dataset).")
        return {}, set(SLOT_NAMES)

    # ── For each reference frame, run teacher with all 3 slot prompts ─────────
    pair_mse: dict[str, list[float]] = {
        "left_vs_middle":  [],
        "middle_vs_right": [],
        "left_vs_right":   [],
    }
    per_slot_min_dist: dict[str, list[float]] = {s: [] for s in SLOT_NAMES}

    teacher_policy.eval()
    for ref in ref_frames:
        gi   = ref["gi"]
        meta = ref["meta"]
        vels = {}
        for sn in SLOT_NAMES:
            item = dict(ds[gi])
            item["task"] = slot_prompt(sn)
            item.pop("task_index", None)
            scale_images([item], ds.meta.camera_keys)
            batch = collate_items([item])
            batch = preprocessor(batch)
            noise = torch.randn(1, CHUNK_SIZE, MAX_ACTION_DIM, device=device)
            time_ = torch.rand(1, device=device)
            teacher_cap.clear()
            with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16,
                                                     enabled=(device.type == "cuda")):
                teacher_policy.forward(batch, noise=noise, time=time_)
            if teacher_cap.vel is not None:
                vels[sn] = teacher_cap.vel.float().detach()

        if len(vels) < 3:
            continue

        lm = float(F.mse_loss(vels["left"],   vels["middle"]))
        mr = float(F.mse_loss(vels["middle"],  vels["right"]))
        lr = float(F.mse_loss(vels["left"],    vels["right"]))
        pair_mse["left_vs_middle"].append(lm)
        pair_mse["middle_vs_right"].append(mr)
        pair_mse["left_vs_right"].append(lr)

        # Per-slot minimum distance to the other two slots
        per_slot_min_dist["left"].append(min(lm, lr))
        per_slot_min_dist["middle"].append(min(lm, mr))
        per_slot_min_dist["right"].append(min(mr, lr))

        print(f"  frame gi={gi:6d}  layout={meta.layout_code}  "
              f"L↔M={lm:.4f}  M↔R={mr:.4f}  L↔R={lr:.4f}")

    # ── Aggregate ──────────────────────────────────────────────────────────────
    def mean(lst): return sum(lst) / len(lst) if lst else 0.0

    report = {
        "n_frames":          len(pair_mse["left_vs_middle"]),
        "threshold":         threshold,
        "left_vs_middle":    mean(pair_mse["left_vs_middle"]),
        "middle_vs_right":   mean(pair_mse["middle_vs_right"]),
        "left_vs_right":     mean(pair_mse["left_vs_right"]),
        "per_slot_min_dist": {s: mean(v) for s, v in per_slot_min_dist.items()},
    }

    verified_slots: set = set()
    for sn in SLOT_NAMES:
        min_d = report["per_slot_min_dist"][sn]
        if min_d >= threshold:
            verified_slots.add(sn)
        flag = "✓ VERIFIED" if sn in verified_slots else "✗ UNRELIABLE"
        print(f"  [{flag}] slot='{sn}'  min_separation={min_d:.5f}  (threshold={threshold:.4f})")

    print(f"\n  Verified slots : {sorted(verified_slots)}")
    if len(verified_slots) < 3:
        unverified = sorted(set(SLOT_NAMES) - verified_slots)
        print(f"  Unverified slots: {unverified}")
        print(f"  → Type A samples targeting unverified slots will use DEMO ACTION loss instead.")
    else:
        print(f"  → All slots verified. Full teacher distillation active.")

    if audit_out_dir is not None:
        audit_out_dir.mkdir(parents=True, exist_ok=True)
        report_path = audit_out_dir / "teacher_slot_audit.json"
        with report_path.open("w") as f:
            json.dump(report, f, indent=2)
        print(f"  Audit report saved: {report_path}")

    print("="*60 + "\n")
    return report, verified_slots


# ──────────────────────────────────────────────────────────────────────────────
# Change 4 — Dry-run frame visualiser
# ──────────────────────────────────────────────────────────────────────────────

def save_dry_run_samples(
    samples:     list,
    ds:          LeRobotDataset,
    out_dir:     Path,
    n:           int = 12,
) -> None:
    """Save N representative Type A samples as annotated PNG images.

    Each image shows the first camera frame with a text overlay containing:
      - episode id, frame index, layout
      - original target slot / color
      - generated relative prompt (student)
      - resolved slot
      - teacher slot prompt

    Uses PIL if available, otherwise writes a JSON manifest only.
    """
    save_dir = out_dir / "dry_run_samples"
    save_dir.mkdir(parents=True, exist_ok=True)

    # Filter to Type A samples only (most informative)
    type_a_samples = [(rec, info) for rec, tag, info in samples if tag == TYPE_A]
    chosen         = type_a_samples[:n]
    manifest       = []

    try:
        from PIL import Image, ImageDraw, ImageFont
        pil_ok = True
    except ImportError:
        pil_ok = False
        print("  [dry-run] PIL not available — saving JSON manifest only.")

    camera_key = ds.meta.camera_keys[0] if ds.meta.camera_keys else None

    for i, (rec, info) in enumerate(chosen):
        meta = rec.meta
        # Raw frame dict (uint8 images)
        raw = dict(ds[rec.global_idx])

        entry = {
            "sample_index":    i,
            "global_idx":      rec.global_idx,
            "episode_index":   rec.episode_index,
            "frame_index":     rec.frame_index,
            "layout":          meta.layout,
            "layout_code":     meta.layout_code,
            "original_target_color": meta.target_color,
            "original_target_slot":  meta.target_slot,
            "student_prompt":  info["student_prompt"],
            "resolved_slot":   SLOT_NAMES[info["target_slot_idx"]],
            "teacher_prompt":  info["teacher_prompt"],
            "_slot_name":      info.get("_slot_name", SLOT_NAMES[info["target_slot_idx"]]),
        }
        manifest.append(entry)

        if not pil_ok or camera_key is None:
            continue

        # Build image
        img_tensor = raw.get(camera_key)
        if img_tensor is None:
            continue
        if isinstance(img_tensor, torch.Tensor):
            arr = img_tensor.numpy()
        else:
            arr = img_tensor
        # arr may be [C,H,W] or [H,W,C]
        if arr.ndim == 3 and arr.shape[0] in (1, 3, 4):
            arr = arr.transpose(1, 2, 0)
        if arr.dtype != "uint8":
            arr = (arr * 255).clip(0, 255).astype("uint8")

        try:
            img = Image.fromarray(arr).convert("RGB")
        except Exception:
            continue

        # Expand canvas below the image for text
        text_h = 200
        canvas = Image.new("RGB", (img.width, img.height + text_h), (30, 30, 30))
        canvas.paste(img, (0, 0))
        draw = ImageDraw.Draw(canvas)
        try:
            font = ImageFont.load_default()
        except Exception:
            font = None

        lines = [
            f"ep={rec.episode_index}  fr={rec.frame_index}  layout={' / '.join(meta.layout)}",
            f"orig_target: {meta.target_color} ({meta.target_slot})",
            f"student: {info['student_prompt'][:80]}",
            f"  → resolved slot: {SLOT_NAMES[info['target_slot_idx']]}",
            f"teacher: {info['teacher_prompt']}",
        ]
        y = img.height + 6
        for line in lines:
            draw.text((6, y), line, fill=(220, 220, 220), font=font)
            y += 36

        out_path = save_dir / f"sample_{i:03d}_ep{rec.episode_index}_fr{rec.frame_index}.png"
        canvas.save(out_path)

    # Always save JSON manifest
    with (save_dir / "manifest.json").open("w") as f:
        json.dump(manifest, f, indent=2)

    print(f"  [dry-run] Saved {len(chosen)} sample cards → {save_dir}")
    if pil_ok:
        print(f"    (PNG images + manifest.json)")
    else:
        print(f"    (manifest.json only — install Pillow for images)")


# ──────────────────────────────────────────────────────────────────────────────
# Hooks: velocity capture + hidden state capture (for auxiliary heads)
# ──────────────────────────────────────────────────────────────────────────────

class VelocityCapture:
    """Captures the OUTPUT of action_out_proj (velocity prediction)."""
    def __init__(self):
        self.vel: Optional[torch.Tensor] = None
        self._handle = None

    def register(self, module: nn.Module) -> None:
        def _hook(_mod, _inp, output):
            self.vel = output
        self._handle = module.register_forward_hook(_hook)

    def remove(self) -> None:
        if self._handle is not None:
            self._handle.remove()

    def clear(self) -> None:
        self.vel = None


class HiddenStateCapture:
    """Captures the INPUT to action_out_proj (pre-projection hidden state).

    This is the representation used by auxiliary heads to predict slot / layout.
    """
    def __init__(self):
        self.hidden: Optional[torch.Tensor] = None
        self._handle = None

    def register(self, module: nn.Module) -> None:
        def _hook(_mod, inp, _out):
            # inp is a tuple; the first element is the hidden state
            if isinstance(inp, tuple) and len(inp) > 0:
                self.hidden = inp[0]
            else:
                self.hidden = inp
        self._handle = module.register_forward_hook(_hook)

    def remove(self) -> None:
        if self._handle is not None:
            self._handle.remove()

    def clear(self) -> None:
        self.hidden = None


# ──────────────────────────────────────────────────────────────────────────────
# Auxiliary heads
# ──────────────────────────────────────────────────────────────────────────────

class AuxiliaryHeads(nn.Module):
    """Lightweight heads on top of the action_out_proj input representation.

    target_slot_head : 3-way classification (left / middle / right)
    layout_head      : 6-way classification (bgr / brg / gbr / grb / rbg / rgb)
    """
    def __init__(self, hidden_dim: int, dropout: float = 0.1):
        super().__init__()
        self.slot_head   = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 3),
        )
        self.layout_head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 6),
        )

    def forward(self, hidden: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        hidden: [B, T, D] or [B, D]
        Returns (slot_logits [B,3], layout_logits [B,6])
        """
        if hidden.dim() == 3:
            h = hidden.mean(dim=1)      # pool over time
        else:
            h = hidden
        return self.slot_head(h), self.layout_head(h)


def detect_hidden_dim(raw) -> int:
    """Detect the input dimension of action_out_proj via its weight matrix."""
    proj = raw.action_out_proj
    # Handle both nn.Linear and custom modules
    if hasattr(proj, "weight"):
        return proj.weight.shape[1]
    # Try to find a child Linear
    for m in proj.modules():
        if isinstance(m, nn.Linear):
            return m.weight.shape[1]
    raise RuntimeError("Cannot detect hidden_dim from action_out_proj")


# ──────────────────────────────────────────────────────────────────────────────
# Model setup helpers
# ──────────────────────────────────────────────────────────────────────────────

def configure_trainability_r1(policy, raw) -> None:
    """Stage R1: Semantic warmup — LoRA only.

    Trainable: VLM q/v LoRA + auxiliary heads (managed separately)
    Frozen:    vision encoder, connector, action expert, projections
    """
    for p in raw.parameters():
        p.requires_grad = False
    for name, p in policy.named_parameters():
        if "lora_" in name:
            p.requires_grad = True
    raw.vlm_with_expert.get_vlm_model().vision_model.eval()
    raw.vlm_with_expert.get_vlm_model().connector.eval()


def configure_trainability_r2(policy, raw) -> None:
    """Stage R2: Slot action distillation — LoRA + small action path unfreeze.

    Trainable: LoRA, lm_expert, action projections (lower LR)
    Frozen:    vision encoder, connector, base VLM
    """
    for p in raw.parameters():
        p.requires_grad = False
    for name, p in policy.named_parameters():
        if "lora_" in name:
            p.requires_grad = True
    # Unfreeze action expert and projections
    for sub in [
        raw.vlm_with_expert.lm_expert,
        raw.state_proj,
        raw.action_in_proj,
        raw.action_out_proj,
        raw.action_time_mlp_in,
        raw.action_time_mlp_out,
    ]:
        for p in sub.parameters():
            p.requires_grad = True
    raw.vlm_with_expert.get_vlm_model().vision_model.eval()
    raw.vlm_with_expert.get_vlm_model().connector.eval()


def build_param_groups_r1(policy, aux_heads: AuxiliaryHeads, lora_lr: float, aux_lr: float) -> list[dict]:
    lora_params = [p for n, p in policy.named_parameters() if "lora_" in n and p.requires_grad]
    aux_params  = list(aux_heads.parameters())
    n_lora      = sum(p.numel() for p in lora_params)
    n_aux       = sum(p.numel() for p in aux_params)
    print(f"  R1 — LoRA: {n_lora:,} (lr={lora_lr})  AuxHeads: {n_aux:,} (lr={aux_lr})")
    return [
        {"params": lora_params, "lr": lora_lr, "name": "lora"},
        {"params": aux_params,  "lr": aux_lr,  "name": "aux"},
    ]


def build_param_groups_r2(
    policy, raw, aux_heads: AuxiliaryHeads,
    lora_lr: float, action_lr: float, aux_lr: float,
) -> list[dict]:
    lora_params   = [p for n, p in policy.named_parameters() if "lora_" in n and p.requires_grad]
    lora_ids      = {id(p) for p in lora_params}
    action_params = (
        list(raw.vlm_with_expert.lm_expert.parameters()) +
        list(raw.state_proj.parameters()) +
        list(raw.action_in_proj.parameters()) +
        list(raw.action_out_proj.parameters()) +
        list(raw.action_time_mlp_in.parameters()) +
        list(raw.action_time_mlp_out.parameters())
    )
    action_params = [p for p in action_params if p.requires_grad and id(p) not in lora_ids]
    aux_params    = list(aux_heads.parameters())
    n_lora        = sum(p.numel() for p in lora_params)
    n_action      = sum(p.numel() for p in action_params)
    n_aux         = sum(p.numel() for p in aux_params)
    print(f"  R2 — LoRA: {n_lora:,} (lr={lora_lr})  Action: {n_action:,} (lr={action_lr})  Aux: {n_aux:,} (lr={aux_lr})")
    return [
        {"params": lora_params,   "lr": lora_lr,   "name": "lora"},
        {"params": action_params, "lr": action_lr, "name": "action"},
        {"params": aux_params,    "lr": aux_lr,    "name": "aux"},
    ]


# ──────────────────────────────────────────────────────────────────────────────
# LR schedule
# ──────────────────────────────────────────────────────────────────────────────

def get_cosine_lr(step: int, warmup: int, total: int, base: float, min_lr: float) -> float:
    if step <= warmup:
        return base * max(step, 1) / max(warmup, 1)
    progress = (step - warmup) / max(total - warmup, 1)
    return min_lr + 0.5 * (base - min_lr) * (1.0 + math.cos(math.pi * progress))


def update_lr_r1(optimizer, step: int, warmup: int, total: int, lora_lr: float, aux_lr: float) -> None:
    lrs = [
        get_cosine_lr(step, warmup, total, lora_lr, lora_lr * 0.1),
        get_cosine_lr(step, warmup, total, aux_lr,  aux_lr  * 0.1),
    ]
    for pg, lr in zip(optimizer.param_groups, lrs):
        pg["lr"] = lr


def update_lr_r2(
    optimizer, step: int, warmup: int, total: int,
    lora_lr: float, action_lr: float, aux_lr: float,
) -> None:
    lrs = [
        get_cosine_lr(step, warmup, total, lora_lr,   lora_lr   * 0.05),
        get_cosine_lr(step, warmup, total, action_lr, action_lr * 0.05),
        get_cosine_lr(step, warmup, total, aux_lr,    aux_lr    * 0.1),
    ]
    for pg, lr in zip(optimizer.param_groups, lrs):
        pg["lr"] = lr


# ──────────────────────────────────────────────────────────────────────────────
# Anti-shortcut validation set
# ──────────────────────────────────────────────────────────────────────────────

ANTI_SHORTCUT_CASES = [
    # (layout_code, relative_prompt_fragment, expected_slot_idx)
    # Same phrase "left of green" → different slots for different layouts
    ("brg", "left of the green", 1),    # blue,red,green → middle
    ("rbg", "left of the green", 1),    # red,blue,green → middle
    ("bgl", "left of the green", 0) if False else None,  # placeholder

    # Same phrase "right of red" → different slots
    ("rgl", "right of the red", None),  # placeholder

    # These are the real anti-shortcut cases:
    # "right of blue" in bgr (blue=0) → middle (idx 1)
    ("bgr", "right of the blue", 1),
    # "right of blue" in gbr (blue=1) → right (idx 2)
    ("gbr", "right of the blue", 2),
    # "right of blue" in grb (blue=2) → invalid (out of range), skip
    # "right of red" in rbg (red=0) → middle (idx 1)
    ("rbg", "right of the red", 1),
    # "right of red" in grb (red=1) → right (idx 2)
    ("grb", "right of the red", 2),
    # "left of green" in bgr (green=2) → middle (idx 1)
    ("bgr", "left of the green", 1),
    # "left of green" in brg (green=1) → left (idx 0)
    ("brg", "left of the green", 0),
    # "left of red" in brg (red=1) → left (idx 0)
    ("brg", "left of the red", 0),
    # "left of red" in bgr (red=1) → left (idx 0) — wait
    # bgr = blue(0),green(1),red(2); "left of red" → idx 1 = green = middle
    ("bgr", "left of the red", 1),
    # "right of green" in bgr (green=1) → right (idx 2)
    ("bgr", "right of the green", 2),
    # "right of green" in rbg (green=1) → right (idx 2)
    ("rbg", "right of the green", 2),
    # "right of green" in grb (green=0) → middle (idx 1)
    ("grb", "right of the green", 1),
]

# Clean up None entries
ANTI_SHORTCUT_CASES = [c for c in ANTI_SHORTCUT_CASES if c is not None]

def build_anti_shortcut_validation(prompt_fragment_to_full: dict | None = None) -> list[dict]:
    """Build anti-shortcut validation cases.

    Returns list of dicts:
      layout_code, layout, prompt, expected_slot_idx, expected_slot_name
    """
    # Full relative prompts for each fragment
    cases = []
    for layout_code, frag, expected_slot_idx in ANTI_SHORTCUT_CASES:
        layout = [COLOR_FULL[ch] for ch in layout_code]
        # Build a full prompt from the fragment
        # Find which color is being referenced
        for ref_idx, ref_color in enumerate(layout):
            if frag == f"right of the {ref_color}":
                if ref_idx + 1 <= 2 and expected_slot_idx == ref_idx + 1:
                    prompt = (
                        f"Put the banana into the bowl on the right of the {ref_color} bowl "
                        f"from the robot perspective."
                    )
                    break
            elif frag == f"left of the {ref_color}":
                if ref_idx - 1 >= 0 and expected_slot_idx == ref_idx - 1:
                    prompt = (
                        f"Put the banana into the bowl on the left of the {ref_color} bowl "
                        f"from the robot perspective."
                    )
                    break
        else:
            continue  # skip if we couldn't build a consistent prompt

        resolved = resolve_relative_prompt_to_slot(prompt, layout)
        if resolved != expected_slot_idx:
            print(f"  [WARNING] Anti-shortcut case resolution mismatch: "
                  f"{layout_code} {prompt!r} → {resolved} != {expected_slot_idx}")
            continue

        cases.append({
            "layout_code":       layout_code,
            "layout":            layout,
            "prompt":            prompt,
            "expected_slot_idx": expected_slot_idx,
            "expected_slot":     SLOT_NAMES[expected_slot_idx],
        })

    print(f"  Built {len(cases)} anti-shortcut validation cases")
    return cases


# ──────────────────────────────────────────────────────────────────────────────
# Checkpoint utilities
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


def save_ckpt(
    out_dir: Path,
    step: int,
    policy,
    aux_heads: AuxiliaryHeads,
    init_path: Path,
    stage: str,
) -> None:
    label = f"{stage}_{step:06d}"
    ckpt  = out_dir / "checkpoints" / label / "pretrained_model"
    ckpt.mkdir(parents=True, exist_ok=True)

    # Deep-copy policy before merging to avoid destroying LoRA adapters in memory
    policy_copy = copy.deepcopy(policy)
    merged      = policy_copy.merge_and_unload()
    merged.save_pretrained(ckpt)
    if hasattr(merged, "config"):
        merged.config.save_pretrained(ckpt)
    copy_processors(init_path, ckpt)
    del policy_copy, merged

    # Save auxiliary heads separately
    torch.save(aux_heads.state_dict(), ckpt / "aux_heads.pt")

    print(f"[ckpt] {label} saved → {ckpt}", flush=True)


# ──────────────────────────────────────────────────────────────────────────────
# Training loop
# ──────────────────────────────────────────────────────────────────────────────

def run_training_stage(
    stage:           str,
    policy,
    raw,
    teacher_policy,
    teacher_cap:     VelocityCapture,
    student_cap:     VelocityCapture,
    student_hidden:  HiddenStateCapture,
    aux_heads:       AuxiliaryHeads,
    optimizer,
    ds:              LeRobotDataset,
    sampler:         RelativeDistillSampler,
    preprocessor,
    args,
    out_dir:         Path,
    init_path:       Path,
    save_steps:      set,
    # Change 3: teacher distillation only for verified slots
    verified_slots:  set = frozenset(SLOT_NAMES),
    start_step:      int = 1,
) -> None:
    """Run one training stage (R1 or R2)."""
    device    = next(policy.parameters()).device
    use_bf16  = args.use_amp and device.type == "cuda"
    autocast  = (
        torch.amp.autocast("cuda", dtype=torch.bfloat16)
        if use_bf16 else torch.amp.autocast("cuda", enabled=False)
    )

    total_steps   = args.r1_steps if stage == "R1" else args.r2_steps
    warmup_steps  = args.r1_warmup if stage == "R1" else args.r2_warmup
    lora_lr       = args.r1_lora_lr if stage == "R1" else args.r2_lora_lr
    action_lr     = args.r2_action_lr  # only used in R2
    aux_lr        = args.aux_lr

    loss_ema    = None
    ema_alpha   = 0.1
    t0          = time.time()
    all_params  = [p for p in policy.parameters() if p.requires_grad] + \
                  [p for p in aux_heads.parameters() if p.requires_grad]
    tag_counts: Counter = Counter()

    print(f"\n{'='*60}")
    print(f"  Stage {stage}: {total_steps} steps | lora_lr={lora_lr} | AMP={use_bf16}")
    print(f"{'='*60}\n", flush=True)

    for step in range(start_step, start_step + total_steps):
        # LR update
        rel_step = step - start_step + 1
        if stage == "R1":
            update_lr_r1(optimizer, rel_step, warmup_steps, total_steps, lora_lr, aux_lr)
        else:
            update_lr_r2(optimizer, rel_step, warmup_steps, total_steps, lora_lr, action_lr, aux_lr)

        # ── Sample batch ────────────────────────────────────────────────────
        samples = sampler.next_batch(args.batch_size)

        # Separate by type
        type_a = [(rec, info) for rec, tag, info in samples if tag == TYPE_A]
        type_b = [(rec, info) for rec, tag, info in samples if tag == TYPE_B]
        type_c = [(rec, info) for rec, tag, info in samples if tag == TYPE_C]
        type_d = [(rec, info) for rec, tag, info in samples if tag == TYPE_D]

        # Accumulate type tags for logging
        for _, ttype, info in samples:
            tag_counts[ttype] += 1

        optimizer.zero_grad(set_to_none=True)

        # ── Per-sub-batch forward helper ─────────────────────────────────────
        def _run_sub_batch(
            batch_items:     list,
            student_prompts: list[str],
            teacher_prompts: Optional[list[str]],
            compute_demo_loss: bool,
        ) -> Optional[dict]:
            """Run student (and optionally teacher) forward, return loss tensors + logits."""
            if not batch_items:
                return None

            B = len(batch_items)
            noise = torch.randn(B, CHUNK_SIZE, MAX_ACTION_DIM, device=device)
            time_ = torch.rand(B, device=device)

            # ── Student forward ───────────────────────────────────────────────
            s_items = [fetch_item(ds, rec, p)
                       for rec, p in zip(batch_items, student_prompts)]
            scale_images(s_items, ds.meta.camera_keys)
            s_batch = collate_items(s_items)
            s_batch = preprocessor(s_batch)

            student_cap.clear()
            student_hidden.clear()
            with autocast:
                loss_demo, _ = policy.forward(s_batch, noise=noise, time=time_)

            v_student = student_cap.vel     # captured by hook
            h_student = student_hidden.hidden

            # ── Aux head predictions (from student hidden state) ──────────────
            slot_logits = layout_logits = None
            if h_student is not None:
                with autocast:
                    slot_logits, layout_logits = aux_heads(h_student.float())

            # ── Teacher forward (frozen, no_grad) ────────────────────────────
            loss_distill = torch.tensor(0.0, device=device)
            if teacher_prompts is not None and v_student is not None:
                t_items = [fetch_item(ds, rec, p)
                           for rec, p in zip(batch_items, teacher_prompts)]
                scale_images(t_items, ds.meta.camera_keys)
                t_batch = collate_items(t_items)
                t_batch = preprocessor(t_batch)

                teacher_cap.clear()
                with torch.no_grad(), autocast:
                    teacher_policy.forward(t_batch, noise=noise, time=time_)
                v_teacher = teacher_cap.vel

                if v_teacher is not None:
                    loss_distill = F.mse_loss(
                        v_student.float(),
                        v_teacher.float().detach(),
                    )

            return {
                "distill":       loss_distill,
                "demo":          loss_demo if compute_demo_loss
                                 else torch.tensor(0.0, device=device),
                "slot_logits":   slot_logits,
                "layout_logits": layout_logits,
                "v_student":     v_student,
            }

        # Accumulate all losses into a single tensor, then do one backward.
        total_loss     = torch.tensor(0.0, device=device)
        loss_A_val     = 0.0
        loss_B_val     = 0.0
        loss_C_val     = 0.0
        loss_D_val     = 0.0
        loss_slot_val  = 0.0
        loss_layout_val= 0.0

        # ── Type A: relative → slot teacher distillation ─────────────────────
        # Change 3: split into verified (teacher distill) vs unverified (demo loss)
        if type_a:
            type_a_verified   = [(r, info) for r, info in type_a
                                 if SLOT_NAMES[info["target_slot_idx"]] in verified_slots]
            type_a_unverified = [(r, info) for r, info in type_a
                                 if SLOT_NAMES[info["target_slot_idx"]] not in verified_slots]

            # Verified: student(relative) ≈ teacher(slot) distillation
            if type_a_verified:
                recs      = [r for r, _ in type_a_verified]
                s_prompts = [info["student_prompt"]  for _, info in type_a_verified]
                t_prompts = [info["teacher_prompt"]  for _, info in type_a_verified]
                slot_ids  = torch.tensor([info["target_slot_idx"] for _, info in type_a_verified], device=device)
                lay_ids   = torch.tensor([info["layout_id"]       for _, info in type_a_verified], device=device)

                res = _run_sub_batch(recs, s_prompts, t_prompts, compute_demo_loss=False)
                if res is not None:
                    loss_A = WEIGHT_A * res["distill"]
                    total_loss = total_loss + loss_A
                    loss_A_val = float(loss_A)

                    if res["slot_logits"] is not None:
                        l_slot   = F.cross_entropy(res["slot_logits"],   slot_ids)
                        l_layout = F.cross_entropy(res["layout_logits"], lay_ids)
                        loss_slot   = WEIGHT_SLOT_AUX * l_slot
                        loss_layout = WEIGHT_LAYOUT   * l_layout
                        total_loss  = total_loss + loss_slot + loss_layout
                        loss_slot_val   += float(loss_slot)
                        loss_layout_val += float(loss_layout)

            # Unverified: fall back to demo action loss (same as Type C)
            if type_a_unverified:
                recs      = [r for r, _ in type_a_unverified]
                s_prompts = [info["student_prompt"]  for _, info in type_a_unverified]
                slot_ids  = torch.tensor([info["target_slot_idx"] for _, info in type_a_unverified], device=device)
                lay_ids   = torch.tensor([info["layout_id"]       for _, info in type_a_unverified], device=device)

                res = _run_sub_batch(recs, s_prompts, None, compute_demo_loss=True)
                if res is not None:
                    loss_A_demo = WEIGHT_C * res["demo"]   # same weight as Type C
                    total_loss  = total_loss + loss_A_demo
                    loss_A_val += float(loss_A_demo)

                    if res["slot_logits"] is not None:
                        l_slot   = F.cross_entropy(res["slot_logits"],   slot_ids)
                        l_layout = F.cross_entropy(res["layout_logits"], lay_ids)
                        loss_slot   = WEIGHT_SLOT_AUX * l_slot
                        loss_layout = WEIGHT_LAYOUT   * l_layout
                        total_loss  = total_loss + loss_slot + loss_layout
                        loss_slot_val   += float(loss_slot)
                        loss_layout_val += float(loss_layout)

        # ── Type B: slot preservation ─────────────────────────────────────────
        if type_b:
            recs      = [r for r, _ in type_b]
            s_prompts = [info["student_prompt"]  for _, info in type_b]
            t_prompts = [info["teacher_prompt"]  for _, info in type_b]
            slot_ids  = torch.tensor([info["target_slot_idx"] for _, info in type_b], device=device)
            lay_ids   = torch.tensor([info["layout_id"]       for _, info in type_b], device=device)

            res = _run_sub_batch(recs, s_prompts, t_prompts, compute_demo_loss=False)
            if res is not None:
                loss_B = WEIGHT_B * res["distill"]
                total_loss = total_loss + loss_B
                loss_B_val = float(loss_B)

                if res["slot_logits"] is not None:
                    l_slot   = F.cross_entropy(res["slot_logits"],   slot_ids)
                    l_layout = F.cross_entropy(res["layout_logits"], lay_ids)
                    loss_slot   = WEIGHT_SLOT_AUX * l_slot
                    loss_layout = WEIGHT_LAYOUT   * l_layout
                    total_loss  = total_loss + loss_slot + loss_layout
                    loss_slot_val   += float(loss_slot)
                    loss_layout_val += float(loss_layout)

        # ── Type C: demo action preservation ─────────────────────────────────
        if type_c:
            recs      = [r for r, _ in type_c]
            s_prompts = [info["student_prompt"]  for _, info in type_c]
            slot_ids  = torch.tensor([info["target_slot_idx"] for _, info in type_c], device=device)
            lay_ids   = torch.tensor([info["layout_id"]       for _, info in type_c], device=device)

            res = _run_sub_batch(recs, s_prompts, None, compute_demo_loss=True)
            if res is not None:
                loss_C = WEIGHT_C * res["demo"]
                total_loss = total_loss + loss_C
                loss_C_val = float(loss_C)

                if res["slot_logits"] is not None:
                    l_slot   = F.cross_entropy(res["slot_logits"],   slot_ids)
                    l_layout = F.cross_entropy(res["layout_logits"], lay_ids)
                    loss_slot   = WEIGHT_SLOT_AUX * l_slot
                    loss_layout = WEIGHT_LAYOUT   * l_layout
                    total_loss  = total_loss + loss_slot + loss_layout
                    loss_slot_val   += float(loss_slot)
                    loss_layout_val += float(loss_layout)

        # ── Type D: scaffold consistency ──────────────────────────────────────
        if type_d:
            recs        = [r for r, _ in type_d]
            plain_ps    = [info["student_prompt"]  for _, info in type_d]
            scaffold_ps = [info["scaffold_prompt"] for _, info in type_d]
            t_prompts   = [info["teacher_prompt"]  for _, info in type_d]

            # Student on plain prompt → distilled toward teacher slot prompt
            res_plain = _run_sub_batch(recs, plain_ps, t_prompts, compute_demo_loss=False)
            # Student on scaffold prompt → used as consistency target (stop-grad)
            res_scaf  = _run_sub_batch(recs, scaffold_ps, None, compute_demo_loss=False)

            if res_plain is not None:
                l_plain_teacher = res_plain["distill"]

                # Scaffold internal consistency: plain vel ≈ scaffold vel (detach scaffold)
                l_plain_scaffold = torch.tensor(0.0, device=device)
                if (res_scaf is not None
                        and res_plain["v_student"] is not None
                        and res_scaf["v_student"] is not None):
                    l_plain_scaffold = F.mse_loss(
                        res_plain["v_student"].float(),
                        res_scaf["v_student"].float().detach(),
                    )

                loss_D = WEIGHT_D * (l_plain_teacher + 0.5 * l_plain_scaffold)
                total_loss = total_loss + loss_D
                loss_D_val = float(loss_D)

        # ── Single backward + gradient step ───────────────────────────────────
        if total_loss.requires_grad:
            total_loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(all_params, args.grad_clip)
        optimizer.step()

        # EMA loss tracking
        step_loss = loss_A_val + loss_B_val + loss_C_val + loss_D_val + \
                    loss_slot_val + loss_layout_val
        loss_ema  = (step_loss if loss_ema is None
                     else (1 - ema_alpha) * loss_ema + ema_alpha * step_loss)

        # ── Logging ──────────────────────────────────────────────────────────
        if rel_step % args.log_every == 0 or rel_step == 1:
            elapsed = time.time() - t0
            sps     = rel_step / max(elapsed, 1e-6)
            lrs     = {pg["name"]: pg["lr"] for pg in optimizer.param_groups}
            print(
                f"[{stage}] step={rel_step:5d}/{total_steps} "
                f"loss_ema={loss_ema:.4f} "
                f"A={loss_A_val:.4f} B={loss_B_val:.4f} "
                f"C={loss_C_val:.4f} D={loss_D_val:.4f} "
                f"slot={loss_slot_val:.4f} lay={loss_layout_val:.4f} "
                f"grad={float(grad_norm):.3f} "
                f"lr_lora={lrs.get('lora',0):.1e} "
                f"sps={sps:.1f}",
                flush=True,
            )

            if args.wandb and wandb is not None:
                wandb.log({
                    f"{stage}/loss_ema":     loss_ema,
                    f"{stage}/loss_A":       loss_A_val,
                    f"{stage}/loss_B":       loss_B_val,
                    f"{stage}/loss_C":       loss_C_val,
                    f"{stage}/loss_D":       loss_D_val,
                    f"{stage}/loss_slot":    loss_slot_val,
                    f"{stage}/loss_layout":  loss_layout_val,
                    f"{stage}/grad_norm":    float(grad_norm),
                    f"{stage}/lr_lora":      lrs.get("lora", 0),
                    "step":                  step,
                }, step=step)

        # ── Checkpoint ───────────────────────────────────────────────────────
        if rel_step in save_steps or rel_step == total_steps:
            save_ckpt(out_dir, rel_step, policy, aux_heads, init_path, stage)

    print(f"\n  Stage {stage} complete.", flush=True)


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

wandb = None  # will be imported if --wandb is set


def main() -> None:
    global wandb

    ap = argparse.ArgumentParser(
        description="Relative-slot teacher distillation from eval2_stage2_20k"
    )
    # Paths
    ap.add_argument("--init",          default=DEFAULT_INIT,
                    help="Checkpoint to use as both teacher and student init")
    ap.add_argument("--dataset",       default=DEFAULT_DATASET)
    ap.add_argument("--dataset-root",  default=DEFAULT_DATASET_ROOT)
    ap.add_argument("--out-dir",       default=DEFAULT_OUT)

    # Frame selection (phase-2 window)
    ap.add_argument("--phase2-start",  type=int, default=10,
                    help="First frame index within episode to include (phase-2 start)")
    ap.add_argument("--phase2-end",    type=int, default=30,
                    help="Last frame index within episode to include (phase-2 end)")

    # LoRA
    ap.add_argument("--lora-rank",     type=int,   default=16)
    ap.add_argument("--lora-alpha",    type=int,   default=32)
    ap.add_argument("--lora-dropout",  type=float, default=0.05)

    # Stage R1
    ap.add_argument("--r1-steps",      type=int,   default=1000,
                    help="Stage R1 (semantic warmup) steps")
    ap.add_argument("--r1-warmup",     type=int,   default=100)
    ap.add_argument("--r1-lora-lr",    type=float, default=1e-5)
    ap.add_argument("--r1-save-steps", default="100,200,300,500,1000")
    ap.add_argument("--skip-r1",       action="store_true",
                    help="Skip Stage R1; go straight to R2")

    # Stage R2
    ap.add_argument("--r2-steps",      type=int,   default=2000,
                    help="Stage R2 (slot action distillation) steps")
    ap.add_argument("--r2-warmup",     type=int,   default=200)
    ap.add_argument("--r2-lora-lr",    type=float, default=1e-5)
    ap.add_argument("--r2-action-lr",  type=float, default=3e-6)
    ap.add_argument("--r2-save-steps", default="100,200,300,500,1000,1500,2000,3000")
    ap.add_argument("--skip-r2",       action="store_true",
                    help="Skip Stage R2")

    # Auxiliary heads
    ap.add_argument("--aux-lr",        type=float, default=1e-4)
    ap.add_argument("--aux-dropout",   type=float, default=0.1)

    # Training
    ap.add_argument("--batch-size",    type=int,   default=16)
    ap.add_argument("--grad-clip",     type=float, default=10.0)
    ap.add_argument("--weight-decay",  type=float, default=1e-10)
    ap.add_argument("--log-every",     type=int,   default=25)
    ap.add_argument("--use-amp",       action="store_true", default=True)
    ap.add_argument("--no-amp",        action="store_false", dest="use_amp")
    ap.add_argument("--seed",          type=int,   default=42)
    ap.add_argument("--dry-run",       action="store_true")

    # W&B
    ap.add_argument("--wandb",         action="store_true")
    ap.add_argument("--wandb-project", default="lerobot")
    ap.add_argument("--wandb-name",    default="eval2-relative-slot-distill-r2")

    args = ap.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    init_path = Path(args.init)
    out_dir   = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    r1_save = {int(x) for x in args.r1_save_steps.split(",") if x.strip()}
    r2_save = {int(x) for x in args.r2_save_steps.split(",") if x.strip()}

    if args.wandb:
        import wandb as _wandb
        wandb = _wandb
        wandb.init(project=args.wandb_project, name=args.wandb_name, config=vars(args))

    # ── Dataset ───────────────────────────────────────────────────────────────
    print(f"Loading dataset: {args.dataset}")
    ds_meta  = LeRobotDatasetMetadata(args.dataset, root=args.dataset_root)
    _tmp     = SmolVLAPolicy.from_pretrained(str(init_path))
    delta_ts = resolve_delta_timestamps(_tmp.config, ds_meta)
    del _tmp

    ds = LeRobotDataset(
        args.dataset, root=args.dataset_root,
        delta_timestamps=delta_ts, video_backend="pyav",
    )
    print(f"  {len(ds)} frames, {ds.num_episodes} episodes")

    ep_meta = load_episode_meta(Path(args.dataset_root))
    pool    = build_frame_pool(ds, ep_meta, args.phase2_start, args.phase2_end)
    sampler = RelativeDistillSampler(pool, seed=args.seed)

    anti_shortcut_cases = build_anti_shortcut_validation()

    if args.dry_run:
        print("\n[dry-run] Sampling 200 examples and printing type distribution:")
        samples = sampler.next_batch(200)
        type_cnt = Counter(ttype for _, ttype, _ in samples)
        print("  Type distribution:", dict(type_cnt))
        for rec, ttype, info in samples[:8]:
            print(f"    [{ttype:30s}] s={info.get('student_prompt','')[:60]!r}")
            if "teacher_prompt" in info:
                print(f"    {'':30s}  t={info['teacher_prompt'][:60]!r}")
        print(f"\n[dry-run] Anti-shortcut cases: {len(anti_shortcut_cases)}")
        for c in anti_shortcut_cases[:5]:
            print(f"  layout={c['layout_code']} → slot={c['expected_slot']} | {c['prompt'][:70]!r}")
        return

    # ── Teacher (frozen) ─────────────────────────────────────────────────────
    print(f"\nLoading frozen teacher: {init_path}")
    teacher_policy = SmolVLAPolicy.from_pretrained(str(init_path))
    teacher_policy = teacher_policy.to(device)
    for p in teacher_policy.parameters():
        p.requires_grad = False
    teacher_policy.eval()
    teacher_raw = teacher_policy.model
    teacher_cap = VelocityCapture()
    teacher_cap.register(teacher_raw.action_out_proj)
    n_teacher = sum(p.numel() for p in teacher_policy.parameters())
    print(f"  Teacher: {n_teacher:,} params (all frozen)")

    # ── Student policy ────────────────────────────────────────────────────────
    print(f"\nLoading student policy: {init_path}")
    policy = SmolVLAPolicy.from_pretrained(str(init_path))
    policy.config.device         = str(device)
    policy.config.train_expert_only     = True
    policy.config.freeze_vision_encoder = True

    lora_cfg = LoraConfig(
        r             = args.lora_rank,
        lora_alpha    = args.lora_alpha,
        target_modules= (
            r"model\.vlm_with_expert\.vlm\.model\.text_model"
            r"\.layers\.[0-9]+\.self_attn\.(q|v)_proj"
        ),
        lora_dropout  = args.lora_dropout,
        bias          = "none",
    )
    policy = get_peft_model(policy, lora_cfg)
    raw    = policy.get_base_model().model

    student_cap    = VelocityCapture()
    student_hidden = HiddenStateCapture()
    student_cap.register(raw.action_out_proj)
    student_hidden.register(raw.action_out_proj)

    # Auxiliary heads
    try:
        hidden_dim = detect_hidden_dim(raw)
    except RuntimeError:
        hidden_dim = 2048
        print(f"  WARNING: Could not detect hidden_dim, defaulting to {hidden_dim}")
    print(f"  Detected action_out_proj hidden_dim: {hidden_dim}")

    aux_heads = AuxiliaryHeads(hidden_dim, dropout=args.aux_dropout).to(device)
    print(f"  AuxiliaryHeads: {sum(p.numel() for p in aux_heads.parameters()):,} params")

    # ── Preprocessor ──────────────────────────────────────────────────────────
    preprocessor, _ = make_pre_post_processors(
        policy.config,
        pretrained_path=str(init_path),
        preprocessor_overrides={
            "device_processor": {"device": str(device)},
            "normalizer_processor": {
                "stats":    ds.meta.stats,
                "features": {**policy.config.input_features, **policy.config.output_features},
                "norm_map": policy.config.normalization_mapping,
            },
            "rename_observations_processor": {"rename_map": {}},
        },
    )

    # ─────────────────────────────────────────────────────────────────────────
    # Stage R1: Semantic Warmup
    # ─────────────────────────────────────────────────────────────────────────
    if not args.skip_r1:
        print("\n" + "="*60)
        print("  STAGE R1: Semantic Warmup")
        print("  Trainable: LoRA + auxiliary heads")
        print("  Goal: relative prompt + image → target slot via aux head")
        print("="*60)

        configure_trainability_r1(policy, raw)
        n_train = sum(p.numel() for p in policy.parameters() if p.requires_grad)
        n_total = sum(p.numel() for p in policy.parameters())
        print(f"  Student trainable: {n_train:,} / {n_total:,} ({100*n_train/n_total:.2f}%)")

        policy = policy.to(device)
        raw    = policy.get_base_model().model

        param_groups_r1 = build_param_groups_r1(policy, aux_heads, args.r1_lora_lr, args.aux_lr)
        optimizer_r1    = torch.optim.AdamW(
            param_groups_r1, weight_decay=args.weight_decay, betas=(0.9, 0.95), eps=1e-8,
        )

        policy.train()
        aux_heads.train()
        raw.vlm_with_expert.get_vlm_model().vision_model.eval()
        raw.vlm_with_expert.get_vlm_model().connector.eval()

        run_training_stage(
            stage          = "R1",
            policy         = policy,
            raw            = raw,
            teacher_policy = teacher_policy,
            teacher_cap    = teacher_cap,
            student_cap    = student_cap,
            student_hidden = student_hidden,
            aux_heads      = aux_heads,
            optimizer      = optimizer_r1,
            ds             = ds,
            sampler        = sampler,
            preprocessor   = preprocessor,
            args           = args,
            out_dir        = out_dir,
            init_path      = init_path,
            save_steps     = r1_save,
            start_step     = 1,
        )

    # ─────────────────────────────────────────────────────────────────────────
    # Stage R2: Slot Action Distillation
    # ─────────────────────────────────────────────────────────────────────────
    if not args.skip_r2:
        print("\n" + "="*60)
        print("  STAGE R2: Slot Action Distillation")
        print("  Trainable: LoRA + action expert (lower LR) + aux heads")
        print("  Goal: student(relative) velocity ≈ teacher(slot) velocity")
        print("="*60)

        configure_trainability_r2(policy, raw)
        n_train = sum(p.numel() for p in policy.parameters() if p.requires_grad)
        n_total = sum(p.numel() for p in policy.parameters())
        print(f"  Student trainable: {n_train:,} / {n_total:,} ({100*n_train/n_total:.2f}%)")

        policy = policy.to(device)
        raw    = policy.get_base_model().model

        # Re-register hooks after potential model move
        student_cap.remove()
        student_hidden.remove()
        student_cap    = VelocityCapture()
        student_hidden = HiddenStateCapture()
        student_cap.register(raw.action_out_proj)
        student_hidden.register(raw.action_out_proj)

        param_groups_r2 = build_param_groups_r2(
            policy, raw, aux_heads,
            args.r2_lora_lr, args.r2_action_lr, args.aux_lr,
        )
        optimizer_r2 = torch.optim.AdamW(
            param_groups_r2, weight_decay=args.weight_decay, betas=(0.9, 0.95), eps=1e-8,
        )

        policy.train()
        aux_heads.train()
        raw.vlm_with_expert.get_vlm_model().vision_model.eval()
        raw.vlm_with_expert.get_vlm_model().connector.eval()

        run_training_stage(
            stage          = "R2",
            policy         = policy,
            raw            = raw,
            teacher_policy = teacher_policy,
            teacher_cap    = teacher_cap,
            student_cap    = student_cap,
            student_hidden = student_hidden,
            aux_heads      = aux_heads,
            optimizer      = optimizer_r2,
            ds             = ds,
            sampler        = sampler,
            preprocessor   = preprocessor,
            args           = args,
            out_dir        = out_dir,
            init_path      = init_path,
            save_steps     = r2_save,
            start_step     = 1,
        )

    # ── Cleanup ───────────────────────────────────────────────────────────────
    student_cap.remove()
    student_hidden.remove()
    teacher_cap.remove()
    if args.wandb and wandb is not None:
        wandb.finish()

    print(f"\nDone. Output: {out_dir}")


if __name__ == "__main__":
    main()
