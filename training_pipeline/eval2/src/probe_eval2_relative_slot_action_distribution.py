#!/usr/bin/env python3
"""Offline evaluation: relative-slot action distribution probe.

For each test checkpoint, given a set of reference observations, this script:

1. Runs ALL prompt variants (slot, relative, scaffold) and measures:
   - target_slot_aux accuracy  (from aux heads if checkpoint has them)
   - layout_aux accuracy       (from aux heads)
   - student(relative) ↔ teacher(slot) velocity distance
   - direct slot behavior preservation error

2. Anti-shortcut validation:
   - same phrase → different slots depending on layout
   - detects if model is using phrase shortcuts instead of visual layout

3. Action distribution analysis:
   - for the same observation, plots/reports how different prompts separate

Usage:
  python probe_eval2_relative_slot_action_distribution.py \\
      --checkpoint /home/shadeform/outputs/eval2_relative_slot_distill_r2/checkpoints/R2_002000/pretrained_model \\
      --teacher    /home/shadeform/lerobot_checkpoints/eval2_stage2_20k \\
      --dataset-root /home/shadeform/lerobot_cache/robot-learning-group47/eval2_108_position_only \\
      --out-dir ./probe_results/r2_2000

Pass multiple checkpoints to compare them:
  --checkpoints R2_000300,R2_000500,R2_001000,R2_002000,R2_003000

Validation gates:
  plain relative target_slot accuracy >= 97%
  anti-shortcut  target_slot accuracy >= 95%
  layout accuracy                     >= 95%
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, "/home/shadeform")
sys.path.insert(0, "/home/shadeform/lerobot/src")

from lerobot.datasets.factory import resolve_delta_timestamps
from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
from lerobot.policies.factory import make_pre_post_processors
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

# ──────────────────────────────────────────────────────────────────────────────
# Constants (same as training script)
# ──────────────────────────────────────────────────────────────────────────────
LAYOUT_CODES = ("bgr", "brg", "gbr", "grb", "rbg", "rgb")
LAYOUT_TO_ID = {code: i for i, code in enumerate(LAYOUT_CODES)}
SLOT_NAMES   = ("left", "middle", "right")
SLOT_TO_ID   = {s: i for i, s in enumerate(SLOT_NAMES)}
COLOR_FULL   = {"b": "blue", "g": "green", "r": "red"}
COLORS       = ("blue", "green", "red")
MAX_ACTION_DIM = 32
CHUNK_SIZE     = 50

COMBO_CSV = "combo_episode_map_eval2_fixed_bowl_fixed_banana_108.csv"

DEFAULT_DATASET      = "robot-learning-group47/eval2_108_position_only"
DEFAULT_DATASET_ROOT = "/home/shadeform/lerobot_cache/robot-learning-group47/eval2_108_position_only"
DEFAULT_TEACHER      = "/home/shadeform/lerobot_checkpoints/eval2_stage2_20k"

# ──────────────────────────────────────────────────────────────────────────────
# Prompt generators (duplicated from training script for standalone use)
# ──────────────────────────────────────────────────────────────────────────────

def slot_prompt(slot_name: str) -> str:
    return f"Put the banana into the {slot_name} bowl."


def relative_prompts_for_slot(layout: list[str], target_slot_idx: int) -> list[str]:
    ti = target_slot_idx
    prompts = []
    if ti + 1 <= 2:
        ref = layout[ti + 1]
        prompts.append(
            f"Put the banana into the bowl on the left of the {ref} bowl "
            f"from the robot perspective."
        )
    if ti - 1 >= 0:
        ref = layout[ti - 1]
        prompts.append(
            f"Put the banana into the bowl on the right of the {ref} bowl "
            f"from the robot perspective."
        )
    if ti == 1:
        prompts.append(
            f"Put the banana into the bowl that is between the {layout[0]} "
            f"and {layout[2]} bowls."
        )
    excl = [c for i, c in enumerate(layout) if i != ti]
    prompts.append(
        f"Put the banana into the bowl that is not {excl[0]} and not {excl[1]}."
    )
    return prompts


def scaffold_prompt(layout: list[str], target_slot_idx: int) -> str:
    slot_name  = SLOT_NAMES[target_slot_idx]
    layout_str = ", ".join(layout)
    return (
        f"The bowls from left to right are {layout_str}. "
        f"The target bowl is the {slot_name} bowl. "
        f"Put the banana into the {slot_name} bowl."
    )


# ──────────────────────────────────────────────────────────────────────────────
# Episode metadata
# ──────────────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class EpisodeMeta:
    episode_index:   int
    layout:          list
    layout_code:     str
    layout_id:       int
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
# Hooks
# ──────────────────────────────────────────────────────────────────────────────

class VelocityCapture:
    def __init__(self):
        self.vel = None
        self._handle = None

    def register(self, module):
        def _hook(_m, _inp, out):
            self.vel = out
        self._handle = module.register_forward_hook(_hook)

    def remove(self):
        if self._handle:
            self._handle.remove()

    def clear(self):
        self.vel = None


class HiddenCapture:
    def __init__(self):
        self.hidden = None
        self._handle = None

    def register(self, module):
        def _hook(_m, inp, _out):
            self.hidden = inp[0] if isinstance(inp, tuple) else inp
        self._handle = module.register_forward_hook(_hook)

    def remove(self):
        if self._handle:
            self._handle.remove()

    def clear(self):
        self.hidden = None


# ──────────────────────────────────────────────────────────────────────────────
# Auxiliary heads (must match training script definition)
# ──────────────────────────────────────────────────────────────────────────────

class AuxiliaryHeads(nn.Module):
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

    def forward(self, hidden):
        h = hidden.mean(dim=1) if hidden.dim() == 3 else hidden
        return self.slot_head(h), self.layout_head(h)


# ──────────────────────────────────────────────────────────────────────────────
# Anti-shortcut validation cases
# ──────────────────────────────────────────────────────────────────────────────

ANTI_SHORTCUT_CASES = [
    # (layout_code, prompt, expected_slot_idx)
    # "right of blue" with different layout positions
    ("bgr", "Put the banana into the bowl on the right of the blue bowl from the robot perspective.", 1),
    ("gbr", "Put the banana into the bowl on the right of the blue bowl from the robot perspective.", 2),
    # "left of red" with different layouts
    ("brg", "Put the banana into the bowl on the left of the red bowl from the robot perspective.",  0),
    ("bgr", "Put the banana into the bowl on the left of the red bowl from the robot perspective.",  1),
    # "right of red" with different layouts
    ("rbg", "Put the banana into the bowl on the right of the red bowl from the robot perspective.", 1),
    ("grb", "Put the banana into the bowl on the right of the red bowl from the robot perspective.", 2),
    # "left of green" with different layouts
    ("bgr", "Put the banana into the bowl on the left of the green bowl from the robot perspective.", 0),
    # bgr = blue(0),green(1),red(2); left of green → blue = slot 0
    ("rbg", "Put the banana into the bowl on the left of the green bowl from the robot perspective.", 1),
    # rbg = red(0),blue(1),green(2); left of green → blue = slot 1
    # "right of green" with different layouts
    ("grb", "Put the banana into the bowl on the right of the green bowl from the robot perspective.", 1),
    ("bgr", "Put the banana into the bowl on the right of the green bowl from the robot perspective.", 2),
    # Between: always middle
    ("bgr", "Put the banana into the bowl that is between the blue and red bowls.", 1),
    ("gbr", "Put the banana into the bowl that is between the green and red bowls.", 1),
    ("rgb", "Put the banana into the bowl that is between the red and blue bowls.",  1),
    # Negation: not X and not Y
    ("bgr", "Put the banana into the bowl that is not blue and not red.", 1),   # green = middle
    ("rbg", "Put the banana into the bowl that is not red and not green.", 1),  # blue = middle
    ("grb", "Put the banana into the bowl that is not green and not blue.", 1), # red = middle
]


def verify_anti_shortcut_cases() -> list[dict]:
    """Verify each case is consistent with resolve logic and return valid cases."""
    from eval2_layout_utils import COLOR_NAMES as _CN
    valid = []
    for layout_code, prompt, expected_slot in ANTI_SHORTCUT_CASES:
        layout = [COLOR_FULL[ch] for ch in layout_code]
        # Verify by simple string matching
        p = prompt.lower()
        resolved = None
        for ri, rc in enumerate(layout):
            if f"left of the {rc} bowl" in p:
                resolved = ri - 1 if ri - 1 >= 0 else None
                break
            if f"right of the {rc} bowl" in p:
                resolved = ri + 1 if ri + 1 <= 2 else None
                break
        if "between" in p:
            resolved = 1
        if "not " in p:
            excluded = []
            for c in COLORS:
                if f"not {c}" in p:
                    excluded.append(c)
            remaining = [c for c in layout if c not in excluded]
            if len(remaining) == 1:
                resolved = layout.index(remaining[0])

        if resolved is not None and resolved == expected_slot:
            valid.append({
                "layout_code":       layout_code,
                "layout":            layout,
                "layout_id":         LAYOUT_TO_ID.get(layout_code, 0),
                "prompt":            prompt,
                "expected_slot_idx": expected_slot,
                "expected_slot":     SLOT_NAMES[expected_slot],
            })
        else:
            print(f"  [SKIP anti-shortcut] resolved={resolved} != expected={expected_slot} "
                  f"for {layout_code}: {prompt[:60]}")
    print(f"  Valid anti-shortcut cases: {len(valid)}")
    return valid


# ──────────────────────────────────────────────────────────────────────────────
# Batch helpers
# ──────────────────────────────────────────────────────────────────────────────

def fetch_item(ds, global_idx: int, prompt: str) -> dict:
    item = dict(ds[global_idx])
    item["task"] = prompt
    item.pop("task_index", None)
    return item


def collate_single(item: dict) -> dict:
    from torch.utils.data import default_collate
    task = item.pop("task")
    batch = default_collate([item])
    batch["task"] = [task]
    return batch


def collate_items(items: list[dict]) -> dict:
    from torch.utils.data import default_collate
    tasks = [it.pop("task") for it in items]
    batch = default_collate(items)
    batch["task"] = tasks
    return batch


def scale_item(item: dict, camera_keys: list[str]) -> None:
    for key in camera_keys:
        if key in item and isinstance(item[key], torch.Tensor) and item[key].dtype == torch.uint8:
            item[key] = item[key].float() / 255.0


# ──────────────────────────────────────────────────────────────────────────────
# Inference helper
# ──────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def get_velocity(
    policy,
    preprocessor,
    cap: VelocityCapture,
    items: list[dict],
    device: torch.device,
) -> Optional[torch.Tensor]:
    """Run a single forward pass and return the captured velocity tensor."""
    if not items:
        return None
    batch = collate_items(items)
    batch = preprocessor(batch)
    B = len(items)
    noise = torch.randn(B, CHUNK_SIZE, MAX_ACTION_DIM, device=device)
    time_ = torch.rand(B, device=device)
    cap.clear()
    with torch.amp.autocast("cuda", dtype=torch.bfloat16):
        policy.forward(batch, noise=noise, time=time_)
    return cap.vel


# ──────────────────────────────────────────────────────────────────────────────
# Probe function for a single checkpoint
# ──────────────────────────────────────────────────────────────────────────────

def probe_checkpoint(
    ckpt_path: Path,
    teacher_path: Path,
    ds: LeRobotDataset,
    ep_meta: dict[int, EpisodeMeta],
    anti_shortcut_cases: list[dict],
    args,
) -> dict:
    """Run full evaluation for one checkpoint. Returns results dict."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load student
    print(f"\nLoading student: {ckpt_path}")
    student = SmolVLAPolicy.from_pretrained(str(ckpt_path))
    student = student.to(device)
    student.eval()
    raw_s = student.model

    s_vel_cap    = VelocityCapture()
    s_hidden_cap = HiddenCapture()
    s_vel_cap.register(raw_s.action_out_proj)
    s_hidden_cap.register(raw_s.action_out_proj)

    # Load auxiliary heads (if available)
    aux_heads = None
    aux_pt = ckpt_path / "aux_heads.pt"
    if aux_pt.exists():
        # Detect hidden dim
        proj = raw_s.action_out_proj
        hidden_dim = 2048
        for m in proj.modules():
            if isinstance(m, nn.Linear):
                hidden_dim = m.weight.shape[1]
                break
        aux_heads = AuxiliaryHeads(hidden_dim).to(device)
        aux_heads.load_state_dict(torch.load(aux_pt, map_location=device))
        aux_heads.eval()
        print(f"  Loaded aux_heads from {aux_pt} (hidden_dim={hidden_dim})")
    else:
        print(f"  No aux_heads.pt found at {aux_pt}")

    # Load teacher
    print(f"Loading teacher: {teacher_path}")
    teacher = SmolVLAPolicy.from_pretrained(str(teacher_path))
    teacher = teacher.to(device)
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad = False
    t_vel_cap = VelocityCapture()
    t_vel_cap.register(teacher.model.action_out_proj)

    # Preprocessor
    preprocessor, _ = make_pre_post_processors(
        student.config,
        pretrained_path=str(ckpt_path),
        preprocessor_overrides={
            "device_processor": {"device": str(device)},
            "normalizer_processor": {
                "stats":    ds.meta.stats,
                "features": {**student.config.input_features, **student.config.output_features},
                "norm_map": student.config.normalization_mapping,
            },
            "rename_observations_processor": {"rename_map": {}},
        },
    )

    # Select test frames: one per layout per target slot (18 total = 6 layouts × 3 slots)
    test_frames = _select_test_frames(ds, ep_meta, args.phase2_probe_frame)
    print(f"  Test frames selected: {len(test_frames)}")

    results = {
        "ckpt":                       str(ckpt_path),
        "n_test_frames":              len(test_frames),
        "slot_preservation":          [],   # per-frame MSE student(slot) ↔ teacher(slot)
        "relative_to_teacher":        [],   # per-frame MSE student(relative) ↔ teacher(slot)
        "scaffold_to_teacher":        [],   # per-frame MSE student(scaffold) ↔ teacher(slot)
        "slot_aux_acc":               None,
        "layout_acc":                 None,
        "anti_shortcut_slot_acc":     None,
        "action_separation":          {},   # layout_code → {left_mid_dist, mid_right_dist}
    }

    # ── A. Slot preservation + relative distillation ──────────────────────────
    slot_mse_list  = []
    rel_mse_list   = []
    scaf_mse_list  = []
    aux_slot_correct  = 0
    aux_layout_correct= 0
    aux_total         = 0

    for frame_info in test_frames:
        gi     = frame_info["global_idx"]
        layout = frame_info["layout"]
        tsi    = frame_info["target_slot_idx"]

        sp = slot_prompt(SLOT_NAMES[tsi])
        rel_opts = relative_prompts_for_slot(layout, tsi)
        sc_p     = scaffold_prompt(layout, tsi)

        if not rel_opts:
            continue

        # Teacher forward with slot prompt
        t_items  = [fetch_item(ds, gi, sp)]
        scale_item(t_items[0], ds.meta.camera_keys)
        t_vel = get_velocity(teacher, preprocessor, t_vel_cap, t_items, device)
        if t_vel is None:
            continue

        # Student: slot prompt (preservation check)
        s_slot_items = [fetch_item(ds, gi, sp)]
        scale_item(s_slot_items[0], ds.meta.camera_keys)
        s_slot_vel = get_velocity(student, preprocessor, s_vel_cap, s_slot_items, device)
        if s_slot_vel is not None:
            slot_mse_list.append(
                float(F.mse_loss(s_slot_vel.float(), t_vel.float()))
            )

        # Student: relative prompt
        for rel_p in rel_opts[:2]:  # limit to 2 per frame
            s_rel_items = [fetch_item(ds, gi, rel_p)]
            scale_item(s_rel_items[0], ds.meta.camera_keys)
            s_rel_vel = get_velocity(student, preprocessor, s_vel_cap, s_rel_items, device)
            if s_rel_vel is not None:
                rel_mse_list.append(
                    float(F.mse_loss(s_rel_vel.float(), t_vel.float()))
                )

            # Aux head prediction
            if aux_heads is not None and s_hidden_cap.hidden is not None:
                with torch.no_grad():
                    slot_logits, layout_logits = aux_heads(s_hidden_cap.hidden.float())
                pred_slot   = int(slot_logits.argmax(dim=-1).item())
                pred_layout = int(layout_logits.argmax(dim=-1).item())
                if pred_slot == tsi:
                    aux_slot_correct += 1
                if pred_layout == frame_info["layout_id"]:
                    aux_layout_correct += 1
                aux_total += 1

        # Student: scaffold prompt
        s_sc_items = [fetch_item(ds, gi, sc_p)]
        scale_item(s_sc_items[0], ds.meta.camera_keys)
        s_sc_vel = get_velocity(student, preprocessor, s_vel_cap, s_sc_items, device)
        if s_sc_vel is not None:
            scaf_mse_list.append(
                float(F.mse_loss(s_sc_vel.float(), t_vel.float()))
            )

    results["slot_preservation"]   = float(sum(slot_mse_list) / max(len(slot_mse_list), 1))
    results["relative_to_teacher"] = float(sum(rel_mse_list)  / max(len(rel_mse_list),  1))
    results["scaffold_to_teacher"] = float(sum(scaf_mse_list) / max(len(scaf_mse_list), 1))

    if aux_total > 0:
        results["slot_aux_acc"]   = aux_slot_correct   / aux_total
        results["layout_acc"]     = aux_layout_correct / aux_total
    else:
        results["slot_aux_acc"]   = None
        results["layout_acc"]     = None

    # ── B. Anti-shortcut validation ───────────────────────────────────────────
    as_correct = 0
    as_total   = 0

    # Group anti-shortcut cases by layout_code
    layout_frames: dict[str, list[int]] = defaultdict(list)
    ep_vals = list(ds.hf_dataset["episode_index"])
    fr_vals = list(ds.hf_dataset["frame_index"])
    for gi, (ep_val, fr_val) in enumerate(zip(ep_vals, fr_vals)):
        ep = int(ep_val)
        fr = int(fr_val)
        if fr != args.phase2_probe_frame:
            continue
        meta = ep_meta.get(ep)
        if meta is None:
            continue
        layout_frames[meta.layout_code].append(gi)

    for case in anti_shortcut_cases:
        code   = case["layout_code"]
        prompt = case["prompt"]
        exp_si = case["expected_slot_idx"]

        frames_for_layout = layout_frames.get(code, [])
        if not frames_for_layout:
            continue

        # Use first available frame for this layout
        gi = frames_for_layout[0]

        # Run student with plain relative prompt
        s_items = [fetch_item(ds, gi, prompt)]
        scale_item(s_items[0], ds.meta.camera_keys)
        _ = get_velocity(student, preprocessor, s_vel_cap, s_items, device)

        # Compare to teacher running left / middle / right slot prompts
        # Predicted slot = argmin of velocity MSE to teacher(left/mid/right)
        teacher_vels = {}
        for sn in SLOT_NAMES:
            t_items = [fetch_item(ds, gi, slot_prompt(sn))]
            scale_item(t_items[0], ds.meta.camera_keys)
            tv = get_velocity(teacher, preprocessor, t_vel_cap, t_items, device)
            if tv is not None:
                teacher_vels[sn] = tv

        if s_vel_cap.vel is not None and teacher_vels:
            sv = s_vel_cap.vel.float()
            dists = {sn: float(F.mse_loss(sv, tv.float())) for sn, tv in teacher_vels.items()}
            pred_slot = min(dists, key=dists.get)
            pred_si   = SLOT_TO_ID[pred_slot]
            if pred_si == exp_si:
                as_correct += 1
            as_total += 1

    results["anti_shortcut_slot_acc"] = (as_correct / as_total) if as_total > 0 else None
    results["anti_shortcut_n"] = as_total

    # ── C. Action separation per layout ──────────────────────────────────────
    # For one frame per layout, run student with left/middle/right slot prompts
    # and measure how well the velocity separates between slots.
    sep_results = {}
    for code in LAYOUT_CODES:
        if not layout_frames.get(code):
            continue
        gi  = layout_frames[code][0]
        vels = {}
        for sn in SLOT_NAMES:
            s_items = [fetch_item(ds, gi, slot_prompt(sn))]
            scale_item(s_items[0], ds.meta.camera_keys)
            sv = get_velocity(student, preprocessor, s_vel_cap, s_items, device)
            if sv is not None:
                vels[sn] = sv.float()
        if len(vels) == 3:
            sep_results[code] = {
                "left_mid_dist":   float(F.mse_loss(vels["left"],   vels["middle"])),
                "mid_right_dist":  float(F.mse_loss(vels["middle"], vels["right"])),
                "left_right_dist": float(F.mse_loss(vels["left"],   vels["right"])),
            }
    results["action_separation"] = sep_results

    # ── Cleanup ───────────────────────────────────────────────────────────────
    s_vel_cap.remove()
    s_hidden_cap.remove()
    t_vel_cap.remove()

    return results


def _select_test_frames(
    ds: LeRobotDataset,
    ep_meta: dict[int, EpisodeMeta],
    probe_frame: int,
) -> list[dict]:
    """Select one test frame per (layout_code, target_slot_idx) combination."""
    seen: set[tuple[str, int]] = set()
    frames = []
    ep_vals = list(ds.hf_dataset["episode_index"])
    fr_vals = list(ds.hf_dataset["frame_index"])
    for gi, (ep_val, fr_val) in enumerate(zip(ep_vals, fr_vals)):
        ep = int(ep_val)
        fr = int(fr_val)
        if fr != probe_frame:
            continue
        meta = ep_meta.get(ep)
        if meta is None:
            continue
        key = (meta.layout_code, meta.target_slot_idx)
        if key in seen:
            continue
        seen.add(key)
        frames.append({
            "global_idx":      gi,
            "episode":         ep,
            "frame":           fr,
            "layout":          meta.layout,
            "layout_code":     meta.layout_code,
            "layout_id":       meta.layout_id,
            "target_slot":     meta.target_slot,
            "target_slot_idx": meta.target_slot_idx,
            "target_color":    meta.target_color,
        })
        if len(frames) >= 18:
            break
    return frames


# ──────────────────────────────────────────────────────────────────────────────
# Report
# ──────────────────────────────────────────────────────────────────────────────

GATE_SLOT_AUX          = 0.97
GATE_ANTI_SHORTCUT     = 0.95
GATE_LAYOUT            = 0.95

def print_report(results: dict, ckpt_label: str) -> None:
    print(f"\n{'─'*60}")
    print(f"  CHECKPOINT: {ckpt_label}")
    print(f"{'─'*60}")
    print(f"  Slot preservation MSE      : {results['slot_preservation']:.5f}  "
          f"(student(slot) ↔ teacher(slot); lower = better)")
    print(f"  Relative→teacher MSE       : {results['relative_to_teacher']:.5f}  "
          f"(student(relative) ↔ teacher(slot); lower = better)")
    print(f"  Scaffold→teacher MSE       : {results['scaffold_to_teacher']:.5f}")

    if results["slot_aux_acc"] is not None:
        sa  = results["slot_aux_acc"]
        la  = results["layout_acc"]
        sa_pass = "✓ PASS" if sa  >= GATE_SLOT_AUX  else "✗ FAIL"
        la_pass = "✓ PASS" if la  >= GATE_LAYOUT    else "✗ FAIL"
        print(f"  Slot aux accuracy          : {sa:.1%}  (gate={GATE_SLOT_AUX:.0%}) {sa_pass}")
        print(f"  Layout accuracy            : {la:.1%}  (gate={GATE_LAYOUT:.0%}) {la_pass}")
    else:
        print(f"  Slot aux accuracy          : N/A (no aux_heads.pt)")
        print(f"  Layout accuracy            : N/A")

    if results["anti_shortcut_slot_acc"] is not None:
        asa     = results["anti_shortcut_slot_acc"]
        as_pass = "✓ PASS" if asa >= GATE_ANTI_SHORTCUT else "✗ FAIL"
        print(f"  Anti-shortcut accuracy     : {asa:.1%}  "
              f"(n={results['anti_shortcut_n']}, gate={GATE_ANTI_SHORTCUT:.0%}) {as_pass}")
    else:
        print(f"  Anti-shortcut accuracy     : N/A")

    print(f"\n  Action separation by layout (slot vel distances):")
    for code, seps in sorted(results["action_separation"].items()):
        print(f"    {code}: L↔M={seps['left_mid_dist']:.4f}  M↔R={seps['mid_right_dist']:.4f}  "
              f"L↔R={seps['left_right_dist']:.4f}")

    # Overall gate
    gates_checked = [
        (results["slot_aux_acc"], GATE_SLOT_AUX, "slot_aux"),
        (results["anti_shortcut_slot_acc"], GATE_ANTI_SHORTCUT, "anti_shortcut"),
        (results["layout_acc"], GATE_LAYOUT, "layout"),
    ]
    all_pass = all(
        v is None or v >= gate
        for v, gate, _ in gates_checked
    )
    print(f"\n  Overall gate: {'✓ PASSED — safe to test on robot' if all_pass else '✗ FAILED — do NOT test on robot yet'}")
    print()


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(
        description="Offline probe: relative-slot action distribution evaluation"
    )
    ap.add_argument("--checkpoint",     default=None,
                    help="Single checkpoint path to evaluate")
    ap.add_argument("--checkpoints",    default=None,
                    help="Comma-separated list of checkpoint paths (compare multiple)")
    ap.add_argument("--teacher",        default=DEFAULT_TEACHER)
    ap.add_argument("--dataset",        default=DEFAULT_DATASET)
    ap.add_argument("--dataset-root",   default=DEFAULT_DATASET_ROOT)
    ap.add_argument("--out-dir",        default="./probe_results")
    ap.add_argument("--phase2-probe-frame", type=int, default=15,
                    help="Frame index within episode to use as reference observation")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Collect checkpoints to evaluate
    ckpt_paths = []
    if args.checkpoint:
        ckpt_paths.append(Path(args.checkpoint))
    if args.checkpoints:
        for p in args.checkpoints.split(","):
            ckpt_paths.append(Path(p.strip()))
    if not ckpt_paths:
        ap.error("Provide --checkpoint or --checkpoints")

    print(f"Probing {len(ckpt_paths)} checkpoint(s)...")
    teacher_path = Path(args.teacher)

    # Dataset
    print(f"\nLoading dataset: {args.dataset}")
    _tmp     = SmolVLAPolicy.from_pretrained(str(ckpt_paths[0]))
    from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata
    ds_meta  = LeRobotDatasetMetadata(args.dataset, root=args.dataset_root)
    delta_ts = resolve_delta_timestamps(_tmp.config, ds_meta)
    del _tmp
    ds = LeRobotDataset(args.dataset, root=args.dataset_root,
                        delta_timestamps=delta_ts, video_backend="pyav")
    print(f"  {len(ds)} frames, {ds.num_episodes} episodes")

    ep_meta = load_episode_meta(Path(args.dataset_root))
    anti_shortcut_cases = verify_anti_shortcut_cases()

    all_results = []
    for ckpt_path in ckpt_paths:
        label = ckpt_path.parent.name if ckpt_path.name == "pretrained_model" else ckpt_path.name
        try:
            res = probe_checkpoint(
                ckpt_path, teacher_path, ds, ep_meta, anti_shortcut_cases, args
            )
            all_results.append((label, res))
            print_report(res, label)
        except Exception as e:
            print(f"\n  [ERROR] {ckpt_path}: {e}")
            import traceback; traceback.print_exc()

    # Save JSON report
    report_path = out_dir / "probe_report.json"
    with report_path.open("w") as f:
        json.dump(
            [{"label": lbl, **res} for lbl, res in all_results],
            f, indent=2, default=str,
        )
    print(f"\nReport saved: {report_path}")

    # Summary table
    if len(all_results) > 1:
        print("\nSummary comparison:")
        print(f"  {'Checkpoint':<30} {'SlotPres':>10} {'RelMSE':>10} {'SlotAcc':>10} {'AntShrt':>10} {'Layout':>10}")
        print(f"  {'-'*80}")
        for label, res in all_results:
            sa  = f"{res['slot_aux_acc']:.1%}"  if res["slot_aux_acc"] is not None else "N/A"
            asa = f"{res['anti_shortcut_slot_acc']:.1%}" if res["anti_shortcut_slot_acc"] is not None else "N/A"
            la  = f"{res['layout_acc']:.1%}"   if res["layout_acc"] is not None else "N/A"
            print(
                f"  {label:<30} "
                f"{res['slot_preservation']:>10.5f} "
                f"{res['relative_to_teacher']:>10.5f} "
                f"{sa:>10} "
                f"{asa:>10} "
                f"{la:>10}"
            )


if __name__ == "__main__":
    main()
