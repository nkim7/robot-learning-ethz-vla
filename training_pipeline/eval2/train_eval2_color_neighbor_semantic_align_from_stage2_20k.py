#!/usr/bin/env python3
"""Stage S1: Semantic-only color-neighbor relative prompt alignment from Stage2-20k.

Goal
====
Fix ONLY the color-neighbor relative prompt family:

    Put the banana into the bowl on the left/right of the [color] bowl from the robot perspective.

Do NOT retrain basic placing behavior. Do NOT disturb already-working prompt families
(direct slot, direct color, ordinal, exclusion).

Core training objective
=======================

    student(color-neighbor prompt representation)
    ≈
    frozen_reference(canonical slot prompt representation)

where "representation" = vlm_last_nonpad_text_hidden:
    the final hidden state at the last non-padding language token
    from the VLM text model, after image+prompt prefix processing.

Stage S1 losses
===============
    L_total_S1 =
        w_distill  * L_repr_distill    (student[color-neighbor] ≈ frozen_ref[canonical slot])
      + w_preserve * L_preserve        (student[slot/ordinal/excl] ≈ frozen_ref[same])
      + w_scaffold * L_scaffold        (student[plain] ≈ student[scaffold].detach())
      + w_slot_ce * L_slot_ce          (slot classification head, 3-way CE)
      + w_layout_ce * L_layout_ce      (layout classification head, 6-way CE)

Hard restrictions (enforced here)
==================================
- NO action loss
- NO action tensors (noise, time, action_out_proj, etc.)
- NO VelocityCapture
- NO velocity MSE
- NO phase2-start / phase2-end frame selection
- NO branch-point frame selection
- NO connector/bridge training in S1
- Frozen: vision encoder, connector, action expert, all action projections
- Trainable: VLM q/v LoRA + SemanticHeads

Distillation tensor
===================
DISTILLATION_TENSOR_NAME = "vlm_last_nonpad_text_hidden"
DISTILLATION_MODULE_PATH = "model.vlm_with_expert.vlm.model.text_model (last non-padding token)"
This is logged explicitly every run. No silent switching.

action_consumed_hidden (diagnostic only, NOT a training loss in S1):
    Defined in probe_eval2_color_neighbor_semantics.py.
    Not computed during S1 training.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import shutil
import time
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Iterable, Iterator, Optional

import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from huggingface_hub import hf_hub_download
from peft import LoraConfig, PeftModel, get_peft_model
from torch.utils.data import DataLoader, Dataset, Sampler

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies.factory import make_pre_post_processors
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
from lerobot.utils.constants import ACTION, OBS_LANGUAGE_ATTENTION_MASK, OBS_LANGUAGE_TOKENS


def _json_default(obj):
    """JSON serializer for types not handled by default — converts Path to str."""
    if isinstance(obj, Path):
        return str(obj)
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

DISTILLATION_TENSOR_NAME = "vlm_last_nonpad_text_hidden"
DISTILLATION_MODULE_PATH = (
    "model.vlm_with_expert.vlm.model.text_model  "
    "(hidden state at last non-padding language token position)"
)
DISTILLATION_IS_BEFORE_CONNECTOR = True
DISTILLATION_GRADIENTS_FLOW_INTO_IT = True  # for student; False for frozen reference

COLORS = ("blue", "red", "green")
SLOTS = ("left", "middle", "right")
LAYOUT_CODES = ("bgr", "brg", "gbr", "grb", "rbg", "rgb")

COLOR_TO_ID = {c: i for i, c in enumerate(COLORS)}
SLOT_TO_ID = {s: i for i, s in enumerate(SLOTS)}
LAYOUT_TO_ID = {code: i for i, code in enumerate(LAYOUT_CODES)}
COLOR_ABBR = {"b": "blue", "g": "green", "r": "red"}

ID_TO_SLOT = {i: s for s, i in SLOT_TO_ID.items()}
ID_TO_COLOR = {i: c for c, i in COLOR_TO_ID.items()}
ID_TO_LAYOUT = {i: code for code, i in LAYOUT_TO_ID.items()}

# Sample types
TYPE_A = "A_color_neighbor_distill"   # student(color-neighbor) ≈ frozen_ref(canonical slot)
TYPE_B = "B_slot_preserve"            # student(slot) ≈ frozen_ref(slot)
TYPE_C = "C_scaffold"                 # student(plain) ≈ student(scaffold).detach()
TYPE_D = "D_ord_excl_preserve"        # student(ordinal/exclusion) ≈ frozen_ref(same)

# Loss weights
WEIGHT_DISTILL = 1.0             # Type A: main representation distillation
WEIGHT_PRESERVE_B = 0.5          # Type B: direct slot preservation
WEIGHT_SCAFFOLD = 0.30           # Type C: scaffold consistency
WEIGHT_PRESERVE_D = 0.30         # Type D: ordinal/exclusion preservation
WEIGHT_SLOT_CE = 2.0             # auxiliary slot head CE — raised from 0.5; slot was collapsed
WEIGHT_LAYOUT_CE = 0.10          # layout head CE — reduced; layout_acc already 93%
WEIGHT_PHRASE_CONFLICT_CE = 2.0  # extra CE on phrase-conflict (hard) Type A cases

# Sample type fractions in each batch (approximate)
FRAC_A = 0.50
FRAC_B = 0.25
FRAC_C = 0.15
FRAC_D = 0.10

# Phrase-conflict fraction within Type A draws
# (same phrase resolves to different slots across layouts — the true anti-shortcut)
PHRASE_CONFLICT_FRAC = 0.60

DEFAULT_DATASET = "robot-learning-group47/eval2_fixed_bowl_fixed_banana_108"
DEFAULT_DATASET_ROOT = "/home/shadeform/lerobot_cache/robot-learning-group47/eval2_fixed_bowl_fixed_banana_108"
DEFAULT_INIT = "/home/shadeform/outputs/eval2_stage2_compositional_015000_balanced_r2/checkpoints/020000/pretrained_model"
DEFAULT_OUT = "/home/shadeform/outputs/eval2_color_neighbor_semantic_align_s1_v1"
COMBO_MAP_NAME = "combo_episode_map_eval2_fixed_bowl_fixed_banana_108.csv"


# ─────────────────────────────────────────────────────────────────────────────
# Data classes
# ─────────────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class EpisodeMeta:
    episode_index: int
    layout_code: str            # e.g. "bgr"
    layout: tuple               # ("blue", "green", "red")
    target_color: str
    target_slot: str            # "left" / "middle" / "right"
    target_slot_id: int
    layout_id: int              # 0-5
    replicate_index: int


@dataclass(frozen=True)
class FrameRecord:
    global_idx: int             # global index into the LeRobotDataset
    episode_index: int
    meta: EpisodeMeta


@dataclass(frozen=True)
class S1Case:
    frame_id: int               # index into frames list
    split: str                  # "train" or "val"
    sample_type: str            # TYPE_A / TYPE_B / TYPE_C / TYPE_D
    student_prompt: str
    ref_prompt: str             # canonical slot prompt (for distillation); same=student for B/D
    scaffold_prompt: str        # for TYPE_C only; empty string otherwise
    target_slot_id: int
    layout_id: int
    layout_code: str
    # direction_conflict: right_of(leftmost)→middle OR left_of(rightmost)→middle
    # These always target middle; NOT the primary anti-shortcut metric.
    direction_conflict: bool
    # phrase_key: e.g. "right_of_red" — identifies which color-neighbor phrase this is
    phrase_key: str
    # phrase_conflict: True if this phrase_key resolves to ≥2 different target slots
    # within the same split.  This is the true anti-shortcut metric.
    phrase_conflict: bool
    side: str                   # "left_of" / "right_of" for TYPE_A; "" otherwise
    anchor_color: str           # for TYPE_A; "" otherwise


# ─────────────────────────────────────────────────────────────────────────────
# Episode metadata loading
# ─────────────────────────────────────────────────────────────────────────────

def _combo_map_path(dataset: str, dataset_root: Path) -> Path:
    local = dataset_root / COMBO_MAP_NAME
    if local.exists():
        return local
    downloaded = hf_hub_download(dataset, COMBO_MAP_NAME, repo_type="dataset")
    return Path(downloaded)


def _normalize_layout_code(value: str) -> str:
    value = str(value).strip().lower().replace("-", "")
    if len(value) != 3 or set(value) != {"b", "g", "r"}:
        raise ValueError(f"Bad layout code: {value!r}")
    return value


def _layout_from_code(code: str) -> tuple:
    return tuple(COLOR_ABBR[ch] for ch in code)


def load_episode_truth(dataset: str, dataset_root: Path) -> dict[int, EpisodeMeta]:
    df = pd.read_csv(_combo_map_path(dataset, dataset_root))
    truth: dict[int, EpisodeMeta] = {}
    for _, row in df.iterrows():
        code = _normalize_layout_code(str(row["bowl_order_left_to_right"]))
        layout = _layout_from_code(code)
        tgt_color = str(row["target_bowl_color"]).strip().lower()
        tgt_slot_id = list(layout).index(tgt_color)
        tgt_slot = SLOTS[tgt_slot_id]
        layout_id = LAYOUT_TO_ID.get(code, 0)
        start = int(row["episode_start"])
        end = int(row["episode_end"])
        for ep in range(start, end + 1):
            truth[ep] = EpisodeMeta(
                episode_index=ep,
                layout_code=code,
                layout=layout,
                target_color=tgt_color,
                target_slot=tgt_slot,
                target_slot_id=tgt_slot_id,
                layout_id=layout_id,
                replicate_index=ep - start,
            )
    return truth


def first_frames(ds: LeRobotDataset, truth: dict[int, EpisodeMeta]) -> list[FrameRecord]:
    """Return the first frame record for each episode in the dataset."""
    records = []
    for ep in ds.meta.episodes:
        episode_index = int(ep["episode_index"])
        if episode_index not in truth:
            continue
        meta = truth[episode_index]
        records.append(FrameRecord(
            global_idx=int(ep["dataset_from_index"]),
            episode_index=episode_index,
            meta=meta,
        ))
    if not records:
        raise RuntimeError("No frames matched the combo map — check dataset / CSV alignment")
    return records


# ─────────────────────────────────────────────────────────────────────────────
# Prompt generation
# ─────────────────────────────────────────────────────────────────────────────

def canonical_slot_prompt(slot_name: str) -> str:
    """Direct slot prompt used by the frozen reference as the distillation teacher."""
    return f"Put the banana into the {slot_name} bowl."


def color_neighbor_prompt(side: str, anchor_color: str) -> str:
    """Plain color-neighbor relative prompt (used at inference)."""
    word = "left" if side == "left_of" else "right"
    return (
        f"Put the banana into the bowl on the {word} of the {anchor_color} bowl "
        f"from the robot perspective."
    )


def scaffold_color_neighbor_prompt(
    layout: tuple,
    side: str,
    anchor_color: str,
    target_slot: str,
) -> str:
    """Scaffold prompt that explicitly states layout and resolution.

    Used ONLY during training (scaffold consistency loss).
    NEVER used at inference.
    """
    word = "left" if side == "left_of" else "right"
    anchor_idx = list(layout).index(anchor_color)
    layout_str = ", ".join(layout)
    return (
        f"The bowls from left to right from the robot perspective are {layout_str}. "
        f"The {anchor_color} bowl is in the {SLOTS[anchor_idx]} position. "
        f"The bowl on the {word} of the {anchor_color} bowl is the {target_slot} bowl. "
        f"Put the banana into the {target_slot} bowl."
    )


def valid_color_neighbor_cases(layout: tuple) -> list[dict]:
    """All valid color-neighbor cases for a given layout.

    Returns list of dicts with:
      side, anchor_color, target_slot, target_slot_id, phrase_key, direction_conflict.

    Valid cases (from left-to-right layout A|B|C):
      - right_of(A) → B (middle)   [anchor at pos 0, target at pos 1]
      - left_of(B)  → A (left)     [anchor at pos 1, target at pos 0]
      - right_of(B) → C (right)    [anchor at pos 1, target at pos 2]
      - left_of(C)  → B (middle)   [anchor at pos 2, target at pos 1]

    direction_conflict: direction word contradicts naive edge heuristic.
      - right_of(leftmost) → middle (direction=right but result is not rightmost)
      - left_of(rightmost) → middle (direction=left but result is not leftmost)
    NOTE: direction_conflict is always-middle by construction — it is NOT the primary
    anti-shortcut metric. phrase_conflict (set later by mark_phrase_conflicts) is the
    real test: same phrase resolves to different slots across layouts.

    phrase_key: canonical identifier for the prompt, e.g. "right_of_red".
    phrase_conflict: always False here; set to True by mark_phrase_conflicts() after
    all cases for all layouts are built.
    """
    out = []
    for anchor_idx, anchor_color in enumerate(layout):
        # Can go left
        if anchor_idx > 0:
            target_idx = anchor_idx - 1
            direction_conflict = (anchor_idx == 2)  # left_of_rightmost → middle
            out.append({
                "side": "left_of",
                "anchor_color": anchor_color,
                "target_slot": SLOTS[target_idx],
                "target_slot_id": target_idx,
                "phrase_key": f"left_of_{anchor_color}",
                "direction_conflict": direction_conflict,
            })
        # Can go right
        if anchor_idx < 2:
            target_idx = anchor_idx + 1
            direction_conflict = (anchor_idx == 0)  # right_of_leftmost → middle
            out.append({
                "side": "right_of",
                "anchor_color": anchor_color,
                "target_slot": SLOTS[target_idx],
                "target_slot_id": target_idx,
                "phrase_key": f"right_of_{anchor_color}",
                "direction_conflict": direction_conflict,
            })
    return out


def ordinal_prompts_for_episode(meta: EpisodeMeta) -> list[str]:
    """Ordinal prompts that resolve to the episode's target slot."""
    ordinals_left = ("1st", "2nd", "3rd")
    ordinals_right = ("3rd", "2nd", "1st")
    si = meta.target_slot_id
    prompts = [
        f"Put the banana into the {ordinals_left[si]} bowl from the left from the robot perspective.",
        f"Put the banana into the {ordinals_right[si]} bowl from the right from the robot perspective.",
    ]
    return prompts


def exclusion_prompts_for_episode(meta: EpisodeMeta) -> list[str]:
    """Exclusion (negation) prompts that resolve to the episode's target slot."""
    excluded = [c for c in COLORS if c != meta.target_color]
    return [
        f"Put the banana into the bowl that is not {excluded[0]} and not {excluded[1]}.",
        f"Put the banana into the bowl that is not {excluded[1]} and not {excluded[0]}.",
    ]


# ─────────────────────────────────────────────────────────────────────────────
# Case building
# ─────────────────────────────────────────────────────────────────────────────

def build_s1_cases(
    frames: list[FrameRecord],
    train_reps: set[int],
    val_reps: set[int],
) -> list[S1Case]:
    """Build all S1 training and validation cases from the frame records."""
    cases: list[S1Case] = []

    for frame_id, frame in enumerate(frames):
        rep = frame.meta.replicate_index
        if rep in train_reps:
            split = "train"
        elif rep in val_reps:
            split = "val"
        else:
            continue

        meta = frame.meta
        layout = meta.layout

        # ── Type A: color-neighbor relative prompts (main target) ─────────────
        for cn in valid_color_neighbor_cases(layout):
            plain_p = color_neighbor_prompt(cn["side"], cn["anchor_color"])
            slot_p = canonical_slot_prompt(cn["target_slot"])
            scaffold_p = scaffold_color_neighbor_prompt(
                layout, cn["side"], cn["anchor_color"], cn["target_slot"]
            )
            # Main distillation case
            cases.append(S1Case(
                frame_id=frame_id,
                split=split,
                sample_type=TYPE_A,
                student_prompt=plain_p,
                ref_prompt=slot_p,
                scaffold_prompt="",
                target_slot_id=cn["target_slot_id"],
                layout_id=meta.layout_id,
                layout_code=meta.layout_code,
                direction_conflict=cn["direction_conflict"],
                phrase_key=cn["phrase_key"],
                phrase_conflict=False,   # filled by mark_phrase_conflicts()
                side=cn["side"],
                anchor_color=cn["anchor_color"],
            ))
            # Scaffold consistency case (for Type C)
            cases.append(S1Case(
                frame_id=frame_id,
                split=split,
                sample_type=TYPE_C,
                student_prompt=plain_p,
                ref_prompt=slot_p,
                scaffold_prompt=scaffold_p,
                target_slot_id=cn["target_slot_id"],
                layout_id=meta.layout_id,
                layout_code=meta.layout_code,
                direction_conflict=cn["direction_conflict"],
                phrase_key=cn["phrase_key"],
                phrase_conflict=False,   # filled by mark_phrase_conflicts()
                side=cn["side"],
                anchor_color=cn["anchor_color"],
            ))

        # ── Type B: direct slot preservation ─────────────────────────────────
        for si, slot_name in enumerate(SLOTS):
            sp = canonical_slot_prompt(slot_name)
            cases.append(S1Case(
                frame_id=frame_id,
                split=split,
                sample_type=TYPE_B,
                student_prompt=sp,
                ref_prompt=sp,  # same prompt; student ≈ ref
                scaffold_prompt="",
                target_slot_id=si,
                layout_id=meta.layout_id,
                layout_code=meta.layout_code,
                direction_conflict=False,
                phrase_key="",
                phrase_conflict=False,
                side="",
                anchor_color="",
            ))

        # ── Type D: ordinal and exclusion preservation ────────────────────────
        for p in ordinal_prompts_for_episode(meta):
            cases.append(S1Case(
                frame_id=frame_id,
                split=split,
                sample_type=TYPE_D,
                student_prompt=p,
                ref_prompt=p,  # same prompt; student ≈ ref
                scaffold_prompt="",
                target_slot_id=meta.target_slot_id,
                layout_id=meta.layout_id,
                layout_code=meta.layout_code,
                direction_conflict=False,
                phrase_key="",
                phrase_conflict=False,
                side="",
                anchor_color="",
            ))
        for p in exclusion_prompts_for_episode(meta):
            cases.append(S1Case(
                frame_id=frame_id,
                split=split,
                sample_type=TYPE_D,
                student_prompt=p,
                ref_prompt=p,
                scaffold_prompt="",
                target_slot_id=meta.target_slot_id,
                layout_id=meta.layout_id,
                layout_code=meta.layout_code,
                direction_conflict=False,
                phrase_key="",
                phrase_conflict=False,
                side="",
                anchor_color="",
            ))

    return cases


def summarize_cases(frames: list[FrameRecord], cases: list[S1Case], split: str) -> dict:
    sc = [c for c in cases if c.split == split]
    fids = {c.frame_id for c in sc}
    return {
        "split": split,
        "num_frames": len(fids),
        "num_cases": len(sc),
        "by_type": dict(Counter(c.sample_type for c in sc)),
        "by_target_slot": dict(Counter(c.target_slot_id for c in sc)),
        "type_A_phrase_conflict": sum(1 for c in sc if c.sample_type == TYPE_A and c.phrase_conflict),
        "type_A_direction_conflict": sum(1 for c in sc if c.sample_type == TYPE_A and c.direction_conflict),
        "type_A_total": sum(1 for c in sc if c.sample_type == TYPE_A),
        "type_A_by_slot": dict(Counter(
            ID_TO_SLOT[c.target_slot_id] for c in sc if c.sample_type == TYPE_A
        )),
        "layouts": dict(Counter(c.layout_code for c in sc if c.sample_type == TYPE_A)),
    }


def mark_phrase_conflicts(cases: list[S1Case]) -> list[S1Case]:
    """Mark phrase_conflict=True for TYPE_A and TYPE_C cases where the same
    phrase_key resolves to ≥2 different target_slot_ids within the same split.

    This is the true anti-shortcut test:
      same phrase ("right_of_red") + different layout → different target slot.

    Example:
      layout = red|green|blue (rgb):  right_of_red → middle  (slot 1)
      layout = green|red|blue (grb):  right_of_red → right   (slot 2)
      → phrase_key "right_of_red" has phrase_conflict=True in any split where
        both layouts appear.

    direction_conflict (always-middle cases) is a separate, weaker concept and
    is not changed by this function.
    """
    # Collect all target slots seen per (split, phrase_key) across TYPE_A cases
    slots_by_split_phrase: dict[tuple, set] = defaultdict(set)
    for c in cases:
        if c.sample_type == TYPE_A:
            slots_by_split_phrase[(c.split, c.phrase_key)].add(c.target_slot_id)

    # A phrase_key is conflicting if it resolves to >=2 different slots in that split
    conflict_keys: set[tuple] = {
        key for key, slots in slots_by_split_phrase.items()
        if len(slots) >= 2
    }

    out: list[S1Case] = []
    for c in cases:
        if c.sample_type in (TYPE_A, TYPE_C) and c.phrase_key:
            is_conflict = (c.split, c.phrase_key) in conflict_keys
            out.append(replace(c, phrase_conflict=is_conflict))
        else:
            out.append(c)
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Semantic heads
# ─────────────────────────────────────────────────────────────────────────────

class SemanticHeads(nn.Module):
    """Lightweight auxiliary classification heads for semantic grounding.

    Input: vlm_last_nonpad_text_hidden [B, D]

    slot_head:   3-way CE  (left / middle / right)
    layout_head: 6-way CE  (bgr / brg / gbr / grb / rbg / rgb)
    """

    def __init__(self, hidden_size: int, dropout: float = 0.05):
        super().__init__()
        self.slot_head = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, 256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, len(SLOTS)),
        )
        self.layout_head = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, 256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, len(LAYOUT_CODES)),
        )

    def forward(self, hidden: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns (slot_logits [B,3], layout_logits [B,6])."""
        return self.slot_head(hidden), self.layout_head(hidden)


# ─────────────────────────────────────────────────────────────────────────────
# VLM hidden state extraction
# ─────────────────────────────────────────────────────────────────────────────

def vlm_last_nonpad_text_hidden(policy, batch: dict) -> torch.Tensor:
    """Extract vlm_last_nonpad_text_hidden: the final VLM hidden state at the
    last non-padding language token, after image+prompt prefix processing.

    This bypasses the connector, lm_expert, and all action modules entirely.
    Only the VLM text model is run.

    Distillation tensor spec:
      name:         vlm_last_nonpad_text_hidden
      module path:  model.vlm_with_expert.vlm.model.text_model
      shape:        [B, hidden_size]
      before connector: YES
      gradients:    YES for student; NO for frozen reference
    """
    raw_policy = policy.get_base_model() if hasattr(policy, "get_base_model") else policy
    flow = raw_policy.model
    images, img_masks = raw_policy.prepare_images(batch)
    lang_tokens = batch[OBS_LANGUAGE_TOKENS]
    lang_masks = batch[OBS_LANGUAGE_ATTENTION_MASK]

    embs = []
    pad_masks = []

    for img, img_mask in zip(images, img_masks, strict=False):
        if flow.add_image_special_tokens:
            image_start = (
                flow.vlm_with_expert.embed_language_tokens(
                    flow.global_image_start_token.to(device=flow.vlm_with_expert.vlm.device)
                )
                .unsqueeze(0)
                .expand(img.shape[0], -1, -1)
            )
            embs.append(image_start)
            pad_masks.append(
                torch.ones_like(image_start[:, :, 0], dtype=torch.bool, device=image_start.device)
            )

        img_emb = flow.vlm_with_expert.embed_image(img)
        img_emb = img_emb * torch.tensor(
            img_emb.shape[-1] ** 0.5, dtype=img_emb.dtype, device=img_emb.device
        )
        img_mask_expanded = img_mask[:, None].expand(img_emb.shape[0], img_emb.shape[1])
        embs.append(img_emb)
        pad_masks.append(img_mask_expanded)

        if flow.add_image_special_tokens:
            image_end = (
                flow.vlm_with_expert.embed_language_tokens(
                    flow.image_end_token.to(device=flow.vlm_with_expert.vlm.device)
                )
                .unsqueeze(0)
                .expand(img.shape[0], -1, -1)
            )
            embs.append(image_end)
            pad_masks.append(
                torch.ones_like(image_end[:, :, 0], dtype=torch.bool, device=image_end.device)
            )

    lang_start = sum(x.shape[1] for x in embs)
    lang_emb = flow.vlm_with_expert.embed_language_tokens(lang_tokens)
    lang_emb = lang_emb * math.sqrt(lang_emb.shape[-1])
    embs.append(lang_emb)
    pad_masks.append(lang_masks)

    inputs_embeds = torch.cat(embs, dim=1)
    attention_mask = torch.cat(pad_masks, dim=1)
    position_ids = torch.cumsum(attention_mask.long(), dim=1) - 1
    position_ids = position_ids.clamp_min(0)

    outputs = flow.vlm_with_expert.get_vlm_model().text_model(
        inputs_embeds=inputs_embeds,
        attention_mask=attention_mask,
        position_ids=position_ids,
        use_cache=False,
    )
    prefix_out = outputs.last_hidden_state  # [B, seq_len, D]
    batch_size = prefix_out.shape[0]
    last_lang_offset = lang_masks.long().sum(dim=1).clamp_min(1) - 1
    last_lang_pos = lang_start + last_lang_offset
    rows = torch.arange(batch_size, device=prefix_out.device)
    hidden = prefix_out[rows, last_lang_pos].to(dtype=torch.float32)

    return hidden  # [B, hidden_size]


# ─────────────────────────────────────────────────────────────────────────────
# Dataset and balanced sampler
# ─────────────────────────────────────────────────────────────────────────────

class ColorNeighborS1Dataset(Dataset):
    """Returns case metadata; images are fetched in the training loop."""

    def __init__(
        self,
        frames: list[FrameRecord],
        cases: list[S1Case],
        split: str,
    ):
        self.frames = frames
        self.cases = [c for c in cases if c.split == split]
        if not self.cases:
            raise ValueError(f"No S1 cases for split={split!r}")

        # Build per-type, per-slot pools for balanced sampling
        self.pools: dict[str, dict[str, list[int]]] = {
            TYPE_A: {s: [] for s in SLOTS},
            TYPE_B: {s: [] for s in SLOTS},
            TYPE_C: {s: [] for s in SLOTS},
            TYPE_D: {s: [] for s in SLOTS},
        }
        for idx, case in enumerate(self.cases):
            if case.sample_type in self.pools:
                slot = ID_TO_SLOT[case.target_slot_id]
                self.pools[case.sample_type][slot].append(idx)

        # Warn about empty pools
        for stype, by_slot in self.pools.items():
            for slot, rows in by_slot.items():
                if not rows:
                    print(f"  WARNING: empty pool for {stype}/{slot}")

    def __len__(self) -> int:
        return len(self.cases)

    def __getitem__(self, idx: int) -> dict:
        case = self.cases[idx]
        return {
            "global_idx": torch.tensor(self.frames[case.frame_id].global_idx, dtype=torch.long),
            "student_prompt": case.student_prompt,
            "ref_prompt": case.ref_prompt,
            "scaffold_prompt": case.scaffold_prompt,
            "target_slot_id": torch.tensor(case.target_slot_id, dtype=torch.long),
            "layout_id": torch.tensor(case.layout_id, dtype=torch.long),
            "sample_type": case.sample_type,
            "phrase_key": case.phrase_key,
            "direction_conflict": int(case.direction_conflict),
            "phrase_conflict": int(case.phrase_conflict),
            "_case_index": torch.tensor(idx, dtype=torch.long),
        }


class BalancedS1Sampler(Sampler):
    """Batch sampler that enforces:

    1. Approximate type fractions (A/B/C/D)
    2. Balanced left/middle/right within Type A (round-robin)
    3. ~PHRASE_CONFLICT_FRAC of Type A drawn from phrase-conflict cases per slot

    phrase_conflict cases are the true anti-shortcut test:
      same phrase_key maps to different target slots across layouts.
    They are balanced per slot, so conflict draws are evenly spread across
    left/middle/right (unlike the old direction_conflict which was middle-only).
    """

    def __init__(
        self,
        dataset: ColorNeighborS1Dataset,
        batch_size: int,
        steps: int,
        frac_a: float = FRAC_A,
        frac_b: float = FRAC_B,
        frac_c: float = FRAC_C,
        phrase_conflict_frac: float = PHRASE_CONFLICT_FRAC,
        seed: int = 47,
    ):
        self.dataset = dataset
        self.batch_size = batch_size
        self.steps = steps
        self.frac_a = frac_a
        self.frac_b = frac_b
        self.frac_c = frac_c
        self.frac_d = max(0.0, 1.0 - frac_a - frac_b - frac_c)
        self.phrase_conflict_frac = phrase_conflict_frac
        self.rng = random.Random(seed)

        # Separate phrase-conflict and normal Type A pools, indexed by target slot
        self.conflict_pools: dict[str, list[int]] = {s: [] for s in SLOTS}
        self.normal_pools: dict[str, list[int]] = {s: [] for s in SLOTS}
        for idx, case in enumerate(dataset.cases):
            if case.sample_type == TYPE_A:
                slot = ID_TO_SLOT[case.target_slot_id]
                if case.phrase_conflict:
                    self.conflict_pools[slot].append(idx)
                else:
                    self.normal_pools[slot].append(idx)

        self._slot_rr = 0

    def _draw_type_a(self) -> int:
        si = self._slot_rr % 3
        self._slot_rr += 1
        slot = SLOTS[si]
        # Try to draw from phrase-conflict pool at rate phrase_conflict_frac
        if self.rng.random() < self.phrase_conflict_frac and self.conflict_pools[slot]:
            return self.rng.choice(self.conflict_pools[slot])
        # Fall back to normal pool, then conflict pool, then any TYPE_A pool
        pool = self.normal_pools[slot] or self.conflict_pools[slot]
        if not pool:
            pool = self.dataset.pools[TYPE_A][slot]
        return self.rng.choice(pool) if pool else 0

    def _draw_from(self, sample_type: str) -> int:
        si = self.rng.randint(0, 2)
        slot = SLOTS[si]
        pool = self.dataset.pools[sample_type][slot]
        if not pool:
            # fallback: any slot
            for s in SLOTS:
                if self.dataset.pools[sample_type][s]:
                    return self.rng.choice(self.dataset.pools[sample_type][s])
            return 0
        return self.rng.choice(pool)

    def __len__(self) -> int:
        return self.steps

    def __iter__(self) -> Iterator[list[int]]:
        n_a = max(1, round(self.batch_size * self.frac_a))
        n_b = max(1, round(self.batch_size * self.frac_b))
        n_c = max(1, round(self.batch_size * self.frac_c))
        n_d = max(0, self.batch_size - n_a - n_b - n_c)

        for _ in range(self.steps):
            batch = []
            for _ in range(n_a):
                batch.append(self._draw_type_a())
            for _ in range(n_b):
                batch.append(self._draw_from(TYPE_B))
            for _ in range(n_c):
                batch.append(self._draw_from(TYPE_C))
            for _ in range(n_d):
                batch.append(self._draw_from(TYPE_D))
            self.rng.shuffle(batch)
            yield batch


# ─────────────────────────────────────────────────────────────────────────────
# Batch helpers
# ─────────────────────────────────────────────────────────────────────────────

def collate_s1(items: list[dict]) -> dict:
    batch: dict = {}
    # Keys that are lists of strings — do not torch.stack
    str_keys = {"student_prompt", "ref_prompt", "scaffold_prompt", "sample_type", "phrase_key"}
    # Keys that are lists of ints (bool flags) — keep as plain lists, not tensors
    list_int_keys = {"direction_conflict", "phrase_conflict"}
    for key in items[0]:
        if key in str_keys:
            batch[key] = [item[key] for item in items]
        elif key in list_int_keys:
            batch[key] = [item[key] for item in items]
        else:
            batch[key] = torch.stack([item[key] for item in items])
    return batch


def fetch_and_build_batch(
    ds: LeRobotDataset,
    global_indices: list[int],
    prompts: list[str],
    camera_keys: list[str],
) -> dict:
    """Fetch raw items from dataset, assign prompts, and collate into a batch dict."""
    items = []
    for gi, prompt in zip(global_indices, prompts):
        item = dict(ds[gi])
        item.pop(ACTION, None)
        item.pop("action_is_pad", None)
        item.pop("task_index", None)
        item["task"] = prompt
        for key in camera_keys:
            if key in item and isinstance(item[key], torch.Tensor) and item[key].dtype == torch.uint8:
                item[key] = item[key].float() / 255.0
        items.append(item)

    # Collate: handle list[str] "task" separately
    tasks = [it.pop("task") for it in items]
    from torch.utils.data import default_collate
    batch = default_collate(items)
    batch["task"] = tasks
    return batch


# ─────────────────────────────────────────────────────────────────────────────
# Model setup
# ─────────────────────────────────────────────────────────────────────────────

def configure_s1_trainability(policy) -> None:
    """S1 frozen: vision encoder, connector, action expert, action projections.
    S1 trainable: VLM q/v LoRA weights only (semantic heads managed separately).
    """
    raw = policy.get_base_model() if hasattr(policy, "get_base_model") else policy
    for param in raw.parameters():
        param.requires_grad = False
    for name, param in policy.named_parameters():
        if "lora_" in name:
            param.requires_grad = True

    # Explicitly set frozen modules to eval mode (prevents BN/dropout updates)
    m = raw.model
    m.vlm_with_expert.get_vlm_model().vision_model.eval()
    m.vlm_with_expert.get_vlm_model().connector.eval()
    m.vlm_with_expert.lm_expert.eval()
    m.state_proj.eval()
    m.action_in_proj.eval()
    m.action_out_proj.eval()
    m.action_time_mlp_in.eval()
    m.action_time_mlp_out.eval()


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


def save_s1_ckpt(
    out_dir: Path,
    step: int,
    policy,
    heads: SemanticHeads,
    init_path: Path,
    run_report: dict,
) -> Path:
    ckpt = out_dir / "checkpoints" / f"{step:06d}" / "pretrained_model"
    ckpt.mkdir(parents=True, exist_ok=True)
    policy.save_pretrained(ckpt)
    policy.config.save_pretrained(ckpt)
    copy_processors(init_path, ckpt)
    torch.save(heads.state_dict(), ckpt / "s1_semantic_heads.pt")
    (ckpt / "s1_run_report.json").write_text(json.dumps(run_report, indent=2, default=_json_default) + "\n")
    return ckpt


def lr_lambda(step: int, warmup: int, total: int) -> float:
    if step < warmup:
        return max(1e-7, step / max(1, warmup))
    progress = (step - warmup) / max(1, total - warmup)
    return max(0.0, 1.0 - progress)


# ─────────────────────────────────────────────────────────────────────────────
# Evaluation
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def evaluate_s1(
    policy,
    frozen_ref,
    heads: SemanticHeads,
    preprocessor,
    ds: LeRobotDataset,
    dataset: ColorNeighborS1Dataset,
    device: str,
    batch_size: int,
    use_amp: bool,
) -> dict:
    """Evaluate semantic accuracy and representation distances on the validation set.

    KEY S1 SUCCESS CRITERION (fix 5):
    ──────────────────────────────────
    S1 is considered successful when:
      distill_repr_dist_color_neighbor_to_slot  DECREASES  (main signal)
      preservation_repr_dist_slot               stays LOW   (no forgetting)
      preservation_repr_dist_ord_excl           stays LOW   (no forgetting)

    Head accuracies (color_neighbor_slot_acc, layout_acc) are auxiliary diagnostics.
    A model can achieve high head accuracy while the underlying VLM representations
    remain far from the canonical slot representation.  The primary training objective
    is representation alignment, not head accuracy.
    """
    policy.eval()
    heads.eval()
    camera_keys = list(ds.meta.camera_keys)

    totals = Counter()
    by_type: dict[str, Counter] = defaultdict(Counter)

    # Phrase-conflict counter: same phrase_key → different slots across layouts (true anti-shortcut)
    phrase_conflict_counter = Counter()
    # Per-slot counters for phrase-conflict (needed for balanced accuracy)
    phrase_conflict_by_slot: dict[str, Counter] = {s: Counter() for s in SLOTS}
    # Direction-conflict counter: right_of(leftmost)/left_of(rightmost) → always middle
    direction_conflict_counter = Counter()

    # layout_acc: ONLY from TYPE_A plain color-neighbor prompts.
    layout_counter = Counter()

    # Representation distance accumulators
    distill_dists: list[float] = []     # Type A: student[color-neighbor] vs frozen_ref[canonical]
    preserve_b_dists: list[float] = []  # Type B: student[slot] vs frozen_ref[slot]
    preserve_d_dists: list[float] = []  # Type D: student[ord/excl] vs frozen_ref[same]

    # Distribution accumulators
    type_a_slot_dist: Counter = Counter()           # resolved target slot counts for Type A
    phrase_conflict_slot_dist: Counter = Counter()  # resolved target slot counts for phrase-conflict
    direction_conflict_slot_dist: Counter = Counter()  # target slot dist for direction-conflict
    phrase_to_target_slots: dict[str, Counter] = defaultdict(Counter)  # phrase_key → target slots seen
    phrase_to_pred_slots: dict[str, Counter] = defaultdict(Counter)    # phrase_key → predicted slots
    layout_dist: Counter = Counter()                # layout code counts for Type A

    # Prediction histograms — what the model actually outputs (detect collapse to one class)
    # These are PREDICTED slot distributions (not ground-truth).
    pred_hist_type_a: Counter = Counter()            # all Type A predictions
    pred_hist_phrase_conflict: Counter = Counter()   # phrase-conflict subset predictions
    pred_hist_direction_conflict: Counter = Counter() # direction-conflict subset predictions

    # Per-slot correct/total for ALL Type A (for overall balanced accuracy)
    type_a_by_slot: dict[str, Counter] = {s: Counter() for s in SLOTS}

    # Norm accumulators for distill tensor spec logging
    student_norms: list[float] = []
    ref_norms: list[float] = []

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_s1,
    )

    for meta_batch in loader:
        global_indices = meta_batch["global_idx"].tolist()
        target_slot_ids = meta_batch["target_slot_id"].to(device)
        layout_ids = meta_batch["layout_id"].to(device)
        sample_types = meta_batch["sample_type"]
        phrase_keys = meta_batch["phrase_key"]               # list[str]
        direction_conflict_flags = meta_batch["direction_conflict"]  # list[int]
        phrase_conflict_flags_eval = meta_batch["phrase_conflict"]   # list[int]
        student_prompts = meta_batch["student_prompt"]       # list[str]
        ref_prompts = meta_batch["ref_prompt"]

        with torch.autocast(device_type="cuda", dtype=torch.bfloat16,
                            enabled=use_amp and device == "cuda"):
            # Student forward — always uses plain prompts (student_prompt field):
            #   TYPE_A: plain color-neighbor
            #   TYPE_B: direct slot
            #   TYPE_C: plain color-neighbor (same as TYPE_A; scaffold is NOT used in eval)
            #   TYPE_D: ordinal / exclusion
            s_batch = fetch_and_build_batch(ds, global_indices, student_prompts, camera_keys)
            s_batch = preprocessor(s_batch)
            s_hiddens = vlm_last_nonpad_text_hidden(policy, s_batch)   # [N, D]

            # Reference forward — frozen; uses canonical slot / same prompts
            r_batch = fetch_and_build_batch(ds, global_indices, ref_prompts, camera_keys)
            r_batch = preprocessor(r_batch)
            r_hiddens = vlm_last_nonpad_text_hidden(frozen_ref, r_batch).detach()  # [N, D]

            # Heads operate on plain student hidden states
            slot_logits, layout_logits = heads(s_hiddens)

        pred_slots = slot_logits.argmax(dim=-1)    # [N]
        pred_layouts = layout_logits.argmax(dim=-1)  # [N]

        for i, (stype, pkey, dir_conf, phr_conf, s_prompt, pred_s, tgt_s, pred_l, tgt_l) in enumerate(
            zip(sample_types, phrase_keys, direction_conflict_flags, phrase_conflict_flags_eval,
                student_prompts, pred_slots, target_slot_ids, pred_layouts, layout_ids)
        ):
            slot_ok    = int(pred_s.item() == tgt_s.item())
            slot_name  = ID_TO_SLOT[tgt_s.item()]
            pred_slot_name = ID_TO_SLOT[pred_s.item()]
            lay_code   = ID_TO_LAYOUT[tgt_l.item()]

            totals["total"]   += 1
            totals["slot_ok"] += slot_ok
            by_type[stype]["total"]   += 1
            by_type[stype]["slot_ok"] += slot_ok

            per_sample_dist = float(F.mse_loss(
                s_hiddens[i].float(), r_hiddens[i].float()
            ).item())

            if stype == TYPE_A:
                layout_counter["total"]    += 1
                layout_counter["layout_ok"] += int(pred_l.item() == tgt_l.item())

                distill_dists.append(per_sample_dist)
                student_norms.append(float(s_hiddens[i].float().norm()))
                ref_norms.append(float(r_hiddens[i].float().norm()))

                type_a_slot_dist[slot_name] += 1
                layout_dist[lay_code]        += 1

                # Prediction histogram for all Type A (detects collapse)
                pred_hist_type_a[pred_slot_name] += 1
                type_a_by_slot[slot_name]["total"]   += 1
                type_a_by_slot[slot_name]["slot_ok"] += slot_ok

                # phrase-level tracking
                if pkey:
                    phrase_to_target_slots[pkey][slot_name] += 1
                    phrase_to_pred_slots[pkey][pred_slot_name] += 1

                # Phrase-conflict (true anti-shortcut): same phrase → different slots
                if phr_conf:
                    phrase_conflict_counter["total"]   += 1
                    phrase_conflict_counter["slot_ok"] += slot_ok
                    phrase_conflict_slot_dist[slot_name] += 1
                    phrase_conflict_by_slot[slot_name]["total"]   += 1
                    phrase_conflict_by_slot[slot_name]["slot_ok"] += slot_ok
                    pred_hist_phrase_conflict[pred_slot_name] += 1

                # Direction-conflict: right_of(leftmost)/left_of(rightmost) → always middle
                if dir_conf:
                    direction_conflict_counter["total"]   += 1
                    direction_conflict_counter["slot_ok"] += slot_ok
                    direction_conflict_slot_dist[slot_name] += 1
                    pred_hist_direction_conflict[pred_slot_name] += 1

            elif stype == TYPE_B:
                preserve_b_dists.append(per_sample_dist)

            elif stype == TYPE_D:
                preserve_d_dists.append(per_sample_dist)

    def acc(c: Counter) -> float:
        return c["slot_ok"] / max(1, c["total"])

    def mean_f(lst: list[float]) -> float:
        return float(sum(lst) / max(1, len(lst)))

    hidden_size = s_hiddens.shape[-1] if len(s_hiddens) else 0

    # ── Balanced (macro-averaged) accuracy ────────────────────────────────────
    # avg(acc_left, acc_middle, acc_right) — penalises middle-collapse correctly.
    # Example: model predicts middle for everything in phrase-conflict subset:
    #   acc_left=0/36, acc_middle=72/72, acc_right=0/36 → balanced=0.333 (not 0.50 raw)
    # Healthy target: >= 0.50 beats random collapse; >= 0.67 means two slots are working.

    # Phrase-conflict per-slot accuracy
    pc_per_slot_acc = {s: acc(phrase_conflict_by_slot[s]) for s in SLOTS}
    pc_balanced_acc = float(sum(pc_per_slot_acc.values()) / 3.0)

    # Overall Type A per-slot accuracy (uses type_a_by_slot collected in loop)
    cn_per_slot_acc = {s: acc(type_a_by_slot[s]) for s in SLOTS}
    cn_balanced_acc = float(sum(cn_per_slot_acc.values()) / 3.0)

    metrics = {
        # ═══════════════════════════════════════════════════════════════════════
        # PRIMARY S1 SIGNAL — representation alignment
        # ═══════════════════════════════════════════════════════════════════════
        "distill_repr_dist_color_neighbor_to_slot": mean_f(distill_dists),
        "preservation_repr_dist_slot":              mean_f(preserve_b_dists),
        "preservation_repr_dist_ord_excl":          mean_f(preserve_d_dists),

        # ── Distillation tensor spec — logged every eval ───────────────────────
        "distill_tensor_name":              DISTILLATION_TENSOR_NAME,
        "distill_tensor_module_path":       DISTILLATION_MODULE_PATH,
        "distill_tensor_shape":             f"[N, {hidden_size}]",
        "distill_tensor_before_connector":  DISTILLATION_IS_BEFORE_CONNECTOR,
        "distill_tensor_grad_student":      True,
        "distill_tensor_grad_ref":          False,
        "distill_tensor_student_norm_mean": mean_f(student_norms),
        "distill_tensor_ref_norm_mean":     mean_f(ref_norms),

        # ── Head accuracy — v2 success criteria ───────────────────────────────
        # SUCCESS = phrase_conflict_acc_balanced >= 0.50 (beats naive middle-predict)
        #           AND prediction_histogram_phrase_conflict is NOT collapsed to middle
        # color_neighbor_slot_acc_raw  >= 0.80: overall slot correct
        # phrase_conflict_acc_raw      >= 0.70: good raw
        # phrase_conflict_acc_balanced >= 0.50: first bar (beats collapse); 0.67+ is good
        # direction_conflict_slot_acc: secondary, always-middle cases
        # layout_acc >= 0.90: visual grounding intact
        # preservation: pres_slot <= 0.02, pres_ord <= 0.05
        "color_neighbor_slot_acc_raw":         acc(by_type[TYPE_A]),
        "color_neighbor_slot_acc_balanced":    cn_balanced_acc,
        "color_neighbor_slot_acc_by_slot":     cn_per_slot_acc,
        "phrase_conflict_acc_raw":             acc(phrase_conflict_counter),
        "phrase_conflict_acc_balanced":        pc_balanced_acc,
        "phrase_conflict_acc_by_slot":         pc_per_slot_acc,
        "direction_conflict_slot_acc":         acc(direction_conflict_counter),
        "layout_acc":                          layout_counter["layout_ok"] / max(1, layout_counter["total"]),
        "preserve_slot_acc":                   acc(by_type[TYPE_B]),
        "preserve_ord_excl_acc":               acc(by_type[TYPE_D]),
        "overall_slot_acc":                    acc(totals),

        # ── Prediction histograms — CRITICAL for diagnosing collapse ──────────
        # If pred_hist_type_a is {"middle": 144}, model is collapsed.
        # Healthy = roughly uniform across left/middle/right.
        "prediction_histogram_type_a":         dict(pred_hist_type_a),
        "prediction_histogram_phrase_conflict": dict(pred_hist_phrase_conflict),
        "prediction_histogram_dir_conflict":    dict(pred_hist_direction_conflict),

        # ── Ground-truth distributions (for reference) ───────────────────────
        "dist_type_a_resolved_slot":      dict(type_a_slot_dist),
        "dist_phrase_conflict_slot":      dict(phrase_conflict_slot_dist),
        "dist_direction_conflict_slot":   dict(direction_conflict_slot_dist),
        "dist_layout":                    dict(layout_dist),
        "phrase_to_target_slots":         {k: dict(v) for k, v in phrase_to_target_slots.items()},
        "phrase_to_pred_slots":           {k: dict(v) for k, v in phrase_to_pred_slots.items()},
    }

    policy.train()
    heads.train()
    return metrics


# ─────────────────────────────────────────────────────────────────────────────
# Training loop
# ─────────────────────────────────────────────────────────────────────────────

def run_s1(
    policy,
    frozen_ref,
    heads: SemanticHeads,
    preprocessor,
    ds: LeRobotDataset,
    train_ds: ColorNeighborS1Dataset,
    val_ds: ColorNeighborS1Dataset,
    args,
    out_dir: Path,
    init_path: Path,
    run_report: dict,
    device: str,
) -> None:
    """Run Stage S1: semantic-only color-neighbor alignment."""
    camera_keys = list(ds.meta.camera_keys)
    use_amp = args.use_amp and device == "cuda"

    lora_params = [p for n, p in policy.named_parameters() if p.requires_grad and "lora_" in n]
    head_params = list(heads.parameters())
    all_trainable = lora_params + head_params

    print(f"  S1 trainable — LoRA: {sum(p.numel() for p in lora_params):,}  "
          f"Heads: {sum(p.numel() for p in head_params):,}",
          flush=True)

    save_steps = {int(x) for x in args.save_steps.split(",") if x.strip()}

    opt = torch.optim.AdamW(
        [
            {"params": lora_params, "lr": args.lora_lr, "name": "lora"},
            {"params": head_params, "lr": args.head_lr, "name": "heads"},
        ],
        weight_decay=args.weight_decay,
        betas=(0.9, 0.95),
        eps=1e-8,
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        opt,
        lr_lambda=lambda step: lr_lambda(step, args.warmup_steps, args.steps),
    )

    sampler = BalancedS1Sampler(
        train_ds,
        batch_size=args.batch_size,
        steps=args.steps,
        frac_a=FRAC_A,
        frac_b=FRAC_B,
        frac_c=FRAC_C,
        phrase_conflict_frac=PHRASE_CONFLICT_FRAC,
        seed=args.seed,
    )
    loader = DataLoader(
        train_ds,
        batch_sampler=sampler,
        num_workers=0,  # metadata-only dataset; images fetched inline
        collate_fn=collate_s1,
    )

    loss_ema: Optional[float] = None
    ema_alpha = 0.1
    t0 = time.time()
    last_ckpt = None

    # Log distillation tensor spec (required)
    hidden_size_at_init = _get_hidden_size(policy)
    distill_spec = {
        "distillation_tensor_name":      DISTILLATION_TENSOR_NAME,
        "module_path":                   DISTILLATION_MODULE_PATH,
        "tensor_shape":                  f"[batch_size, {hidden_size_at_init}]",
        "before_connector":              DISTILLATION_IS_BEFORE_CONNECTOR,
        "grad_student":                  True,
        "grad_ref":                      False,
        "action_consumed_hidden":        (
            "NOT computed in S1 — S1 is semantic-only "
            "(no action forward pass, no action tensors)."
        ),
        "norms_and_distance": "logged every log_every steps in DISTILL_TENSOR lines",
    }
    print("DISTILLATION_TENSOR_SPEC:", json.dumps(distill_spec, indent=2), flush=True)
    run_report["distillation_tensor_spec"] = distill_spec
    (out_dir / "s1_run_report.json").write_text(json.dumps(run_report, indent=2, default=_json_default) + "\n")

    wandb_run = None
    if args.wandb:
        import wandb as _wandb
        wandb_run = _wandb.init(
            project=args.wandb_project,
            name=args.wandb_name,
            config=vars(args),
        )

    policy.train()
    heads.train()

    for step, meta_batch in enumerate(loader, start=1):
        global_indices = meta_batch["global_idx"].tolist()
        target_slot_ids = meta_batch["target_slot_id"].to(device)
        layout_ids = meta_batch["layout_id"].to(device)
        sample_types = meta_batch["sample_type"]
        student_prompts = meta_batch["student_prompt"]
        ref_prompts = meta_batch["ref_prompt"]
        scaffold_prompts = meta_batch["scaffold_prompt"]
        phrase_conflict_flags = meta_batch["phrase_conflict"]  # list[int] 0/1

        # Identify sub-batch indices by type
        a_idx = [i for i, t in enumerate(sample_types) if t == TYPE_A]
        b_idx = [i for i, t in enumerate(sample_types) if t == TYPE_B]
        c_idx = [i for i, t in enumerate(sample_types) if t == TYPE_C]
        d_idx = [i for i, t in enumerate(sample_types) if t == TYPE_D]
        # All four types need a frozen-reference forward.
        # TYPE_C needs it for the scaffold triplet anchor:
        #   student(plain) ≈ student(scaffold).detach()  AND
        #   student(plain) ≈ frozen_ref(canonical slot).detach()
        abcd_idx = a_idx + b_idx + c_idx + d_idx

        opt.zero_grad(set_to_none=True)
        total_loss = torch.tensor(0.0, device=device)
        loss_distill_val = 0.0
        loss_preserve_b_val = 0.0
        loss_scaffold_val = 0.0
        loss_preserve_d_val = 0.0
        loss_slot_ce_val = 0.0
        loss_layout_ce_val = 0.0
        loss_phrase_conflict_ce_val = 0.0

        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=use_amp):

            # ── Student forward (all samples with student prompts) ────────────
            s_batch = fetch_and_build_batch(ds, global_indices, student_prompts, camera_keys)
            s_batch = preprocessor(s_batch)
            s_hiddens = vlm_last_nonpad_text_hidden(policy, s_batch)  # [N, D]

            # ── Frozen reference forward (all types: A, B, C, D) ─────────────
            # TYPE_C needs ref to provide the scaffold triplet anchor:
            #   student_plain[C] ≈ frozen_ref_canonical[C].detach()
            ref_hiddens = None
            if abcd_idx:
                ref_gi = [global_indices[i] for i in abcd_idx]
                ref_ps = [ref_prompts[i] for i in abcd_idx]
                r_batch = fetch_and_build_batch(ds, ref_gi, ref_ps, camera_keys)
                r_batch = preprocessor(r_batch)
                with torch.no_grad():
                    ref_hiddens = vlm_last_nonpad_text_hidden(frozen_ref, r_batch)  # [|abcd|, D]
            # Map original batch position → ref_hiddens row
            abcd_to_ref: dict[int, int] = {orig_i: ri for ri, orig_i in enumerate(abcd_idx)}

            # ── Scaffold forward (Type C) — WITH gradients ───────────────────
            # Gradients must flow into scaffold_hiddens so the scaffold branch
            # can receive the scaffold-to-canonical term:
            #   MSE(student_scaffold, frozen_ref_canonical.detach())
            # Do NOT wrap in torch.no_grad().
            scaffold_hiddens = None
            if c_idx:
                sc_gi = [global_indices[i] for i in c_idx]
                sc_ps = [scaffold_prompts[i] for i in c_idx]
                sc_batch = fetch_and_build_batch(ds, sc_gi, sc_ps, camera_keys)
                sc_batch = preprocessor(sc_batch)
                scaffold_hiddens = vlm_last_nonpad_text_hidden(policy, sc_batch)  # [|C|, D]

            # ── L_repr_distill: Type A  student[color-neighbor] ≈ frozen_ref[canonical slot] ──
            if a_idx and ref_hiddens is not None:
                s_a = s_hiddens[[i for i in a_idx]]
                r_a = ref_hiddens[[abcd_to_ref[i] for i in a_idx]].detach()
                l_distill = F.mse_loss(s_a, r_a)
                total_loss = total_loss + WEIGHT_DISTILL * l_distill
                loss_distill_val = float(WEIGHT_DISTILL * l_distill)

            # ── L_slot_ce + L_layout_ce: auxiliary heads — TYPE_A PLAIN ONLY ───
            # These are auxiliary — S1 success is measured by repr distance, not head acc.
            #
            # layout_CE MUST be computed only on plain color-neighbor hidden states.
            # Scaffold prompts contain explicit layout text ("The bowls from left to
            # right are …"); feeding scaffold hiddens into layout_CE would make the
            # task trivial and invalidate the diagnostic.
            #
            # Guarantee: only a_idx (TYPE_A) rows are selected from s_hiddens.
            #   a_idx   = TYPE_A indices  →  student_prompt = plain color-neighbor
            #   c_idx   = TYPE_C indices  →  scaffold_hiddens computed separately
            # scaffold_hiddens is NEVER passed to the heads anywhere in this script.
            if a_idx:
                _a_set = set(map(int, a_idx.tolist() if hasattr(a_idx, "tolist") else a_idx))
                _c_set = set(map(int, c_idx.tolist() if hasattr(c_idx, "tolist") else c_idx))
                assert _a_set.isdisjoint(_c_set), (
                    "TYPE_A plain indices and TYPE_C scaffold indices overlap. "
                    "Layout/slot heads must never receive scaffold prompts."
                )
                # Plain color-neighbor hidden states only — scaffolds excluded.
                s_a_plain = s_hiddens[[i for i in a_idx]]
                slot_logits, layout_logits = heads(s_a_plain)
                tgt_slots_a = target_slot_ids[[i for i in a_idx]]
                tgt_layouts_a = layout_ids[[i for i in a_idx]]
                l_slot = F.cross_entropy(slot_logits, tgt_slots_a)
                l_layout = F.cross_entropy(layout_logits, tgt_layouts_a)
                total_loss = total_loss + WEIGHT_SLOT_CE * l_slot + WEIGHT_LAYOUT_CE * l_layout
                loss_slot_ce_val = float(WEIGHT_SLOT_CE * l_slot)
                loss_layout_ce_val = float(WEIGHT_LAYOUT_CE * l_layout)

                # Extra CE on phrase-conflict (hard) subset of Type A
                # phrase_conflict_flags is indexed by batch position (orig_i)
                conflict_local = [
                    j for j, orig_i in enumerate(a_idx)
                    if int(phrase_conflict_flags[orig_i]) == 1
                ]
                if conflict_local:
                    conflict_idx_t = torch.tensor(
                        conflict_local, device=slot_logits.device, dtype=torch.long
                    )
                    l_phrase_conflict = F.cross_entropy(
                        slot_logits.index_select(0, conflict_idx_t),
                        tgt_slots_a.index_select(0, conflict_idx_t),
                    )
                    total_loss = total_loss + WEIGHT_PHRASE_CONFLICT_CE * l_phrase_conflict
                    loss_phrase_conflict_ce_val = float(WEIGHT_PHRASE_CONFLICT_CE * l_phrase_conflict)

            # ── L_preserve_B: direct slot preservation ───────────────────────
            if b_idx and ref_hiddens is not None:
                s_b = s_hiddens[[i for i in b_idx]]
                r_b = ref_hiddens[[abcd_to_ref[i] for i in b_idx]].detach()
                l_pres_b = F.mse_loss(s_b, r_b)
                total_loss = total_loss + WEIGHT_PRESERVE_B * l_pres_b
                loss_preserve_b_val = float(WEIGHT_PRESERVE_B * l_pres_b)

            # ── L_scaffold: full triplet ──────────────────────────────────────
            #
            #   student(plain color-neighbor)
            #   ≈  student(scaffold)
            #   ≈  frozen_reference(canonical slot)
            #
            # Three terms:
            #   (1) MSE(student_plain,    student_scaffold.detach()) — plain toward scaffold
            #   (2) MSE(student_plain,    frozen_ref_canonical.detach()) — plain toward canonical
            #   (3) MSE(student_scaffold, frozen_ref_canonical.detach()) — scaffold toward canonical
            #
            # Term (1): sc_c is detached so the plain branch does not drag scaffold away.
            # Term (3): sc_c is NOT detached — gradients flow into scaffold so the scaffold
            #   branch is also pulled toward the canonical representation.
            #   Scaffold prompts are training-only; receiving gradients here is intentional.
            #
            # scaffold_hiddens is used ONLY here and in no other loss term.
            # It is NEVER passed to slot_head or layout_head (see plain-only guard above).
            if c_idx and scaffold_hiddens is not None:
                s_c  = s_hiddens[[i for i in c_idx]]     # student[plain],    grad flows
                sc_c = scaffold_hiddens                   # student[scaffold], grad flows for term 3
                l_scaffold_consistency  = F.mse_loss(s_c, sc_c.detach())       # (1)
                l_scaffold_plain_anchor = torch.tensor(0.0, device=device)
                l_scaffold_to_canonical = torch.tensor(0.0, device=device)
                if ref_hiddens is not None:
                    r_c = ref_hiddens[[abcd_to_ref[i] for i in c_idx]].detach()
                    l_scaffold_plain_anchor = F.mse_loss(s_c,  r_c)            # (2)
                    l_scaffold_to_canonical = F.mse_loss(sc_c, r_c)            # (3)
                l_scaffold = (
                    l_scaffold_consistency     # plain  → scaffold  (scaffold stop-grad)
                    + l_scaffold_plain_anchor  # plain  → canonical
                    + l_scaffold_to_canonical  # scaffold → canonical  (scaffold gets grad)
                )
                total_loss = total_loss + WEIGHT_SCAFFOLD * l_scaffold
                loss_scaffold_val = float(WEIGHT_SCAFFOLD * l_scaffold)

            # ── L_preserve_D: ordinal/exclusion preservation ─────────────────
            if d_idx and ref_hiddens is not None:
                s_d = s_hiddens[[i for i in d_idx]]
                r_d = ref_hiddens[[abcd_to_ref[i] for i in d_idx]].detach()
                l_pres_d = F.mse_loss(s_d, r_d)
                total_loss = total_loss + WEIGHT_PRESERVE_D * l_pres_d
                loss_preserve_d_val = float(WEIGHT_PRESERVE_D * l_pres_d)

        # ── Backward + step ───────────────────────────────────────────────────
        if total_loss.requires_grad:
            total_loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(all_trainable, args.grad_clip_norm)
        opt.step()
        scheduler.step()

        step_loss = (
            loss_distill_val + loss_preserve_b_val + loss_scaffold_val
            + loss_preserve_d_val + loss_slot_ce_val + loss_layout_ce_val
            + loss_phrase_conflict_ce_val
        )
        loss_ema = step_loss if loss_ema is None else (1 - ema_alpha) * loss_ema + ema_alpha * step_loss

        # ── Logging ───────────────────────────────────────────────────────────
        if step == 1 or step % args.log_every == 0:
            # ── Distillation tensor spec — logged every log step (fix 2) ─────
            # Required fields: name, module path, shape, before/after connector,
            # grad-flow status, student norm, ref norm, student-ref distance.
            if a_idx and ref_hiddens is not None:
                with torch.no_grad():
                    s_a_log = s_hiddens[[i for i in a_idx]].float()
                    r_a_log = ref_hiddens[[abcd_to_ref[i] for i in a_idx]].float()
                    s_norm = float(s_a_log.norm(dim=-1).mean())
                    r_norm = float(r_a_log.norm(dim=-1).mean())
                    dist   = float(F.mse_loss(s_a_log, r_a_log))
                    hidden_size = s_a_log.shape[-1]
                print(
                    f"  DISTILL_TENSOR"
                    f" name={DISTILLATION_TENSOR_NAME}"
                    f" module=\"{DISTILLATION_MODULE_PATH}\""
                    f" shape=[{len(a_idx)},{hidden_size}]"
                    f" before_connector={DISTILLATION_IS_BEFORE_CONNECTOR}"
                    f" grad_student=True grad_ref=False"
                    f" student_norm={s_norm:.4f} ref_norm={r_norm:.4f}"
                    f" student_ref_dist={dist:.6f}",
                    flush=True,
                )

            type_counts = Counter(sample_types)
            lrs = {pg["name"]: pg["lr"] for pg in opt.param_groups}

            # Batch sampler diagnostics — target slot and phrase-conflict distributions
            # within Type A draws (verifies sampler is balanced, not collapsed to middle)
            batch_a_slot_dist = Counter(
                ID_TO_SLOT[int(target_slot_ids[i].item())]
                for i, t in enumerate(sample_types) if t == TYPE_A
            )
            batch_pc_slot_dist = Counter(
                ID_TO_SLOT[int(target_slot_ids[i].item())]
                for i, t in enumerate(sample_types)
                if t == TYPE_A and int(phrase_conflict_flags[i]) == 1
            )

            print(
                f"S1 step={step:5d}/{args.steps} loss_ema={loss_ema:.4f} "
                f"distill={loss_distill_val:.4f} pres_b={loss_preserve_b_val:.4f} "
                f"scaffold={loss_scaffold_val:.4f} pres_d={loss_preserve_d_val:.4f} "
                f"slot_ce={loss_slot_ce_val:.4f} layout_ce={loss_layout_ce_val:.4f} "
                f"phrase_conflict_ce={loss_phrase_conflict_ce_val:.4f} "
                f"grad={float(grad_norm):.2f} "
                f"lr_lora={lrs['lora']:.1e} "
                f"types={dict(type_counts)} "
                f"batch_A_slot={dict(batch_a_slot_dist)} "
                f"batch_pc_slot={dict(batch_pc_slot_dist)}",
                flush=True,
            )
            if wandb_run is not None:
                import wandb
                wandb.log({
                    "S1/loss_ema": loss_ema,
                    "S1/loss_distill": loss_distill_val,
                    "S1/loss_preserve_b": loss_preserve_b_val,
                    "S1/loss_scaffold": loss_scaffold_val,
                    "S1/loss_preserve_d": loss_preserve_d_val,
                    "S1/loss_slot_ce": loss_slot_ce_val,
                    "S1/loss_layout_ce": loss_layout_ce_val,
                    "S1/loss_phrase_conflict_ce": loss_phrase_conflict_ce_val,
                    "S1/grad_norm": float(grad_norm),
                    "S1/lr_lora": lrs["lora"],
                    # Batch sampler diagnostics
                    "S1/batch_A_slot_left":    batch_a_slot_dist.get("left", 0),
                    "S1/batch_A_slot_middle":  batch_a_slot_dist.get("middle", 0),
                    "S1/batch_A_slot_right":   batch_a_slot_dist.get("right", 0),
                    "S1/batch_pc_slot_left":   batch_pc_slot_dist.get("left", 0),
                    "S1/batch_pc_slot_middle": batch_pc_slot_dist.get("middle", 0),
                    "S1/batch_pc_slot_right":  batch_pc_slot_dist.get("right", 0),
                }, step=step)

        # ── Evaluation ────────────────────────────────────────────────────────
        if step % args.eval_every == 0 or step == args.steps:
            eval_m = evaluate_s1(
                policy, frozen_ref, heads, preprocessor, ds,
                val_ds, device, args.eval_batch_size, args.use_amp
            )
            # Primary repr-distance line (the real S1 signal)
            print(
                f"EVAL step={step} [REPR_DIST]"
                f" distill={eval_m['distill_repr_dist_color_neighbor_to_slot']:.6f}"
                f" pres_slot={eval_m['preservation_repr_dist_slot']:.6f}"
                f" pres_ord={eval_m['preservation_repr_dist_ord_excl']:.6f}"
                f" student_norm={eval_m['distill_tensor_student_norm_mean']:.4f}"
                f" ref_norm={eval_m['distill_tensor_ref_norm_mean']:.4f}",
                flush=True,
            )
            # Head accuracy line — v2 success criteria
            # SUCCESS: phrase_conflict_acc_balanced >= 0.50 (beats middle-collapse)
            #          AND prediction_histogram_phrase_conflict not all-middle
            pc_by_slot = eval_m["phrase_conflict_acc_by_slot"]
            cn_by_slot = eval_m["color_neighbor_slot_acc_by_slot"]
            print(
                f"EVAL step={step} [HEAD_ACC]"
                f" cn_raw={eval_m['color_neighbor_slot_acc_raw']*100:.1f}%"
                f" cn_bal={eval_m['color_neighbor_slot_acc_balanced']*100:.1f}%"
                f" pc_raw={eval_m['phrase_conflict_acc_raw']*100:.1f}%"
                f" pc_bal={eval_m['phrase_conflict_acc_balanced']*100:.1f}%"
                f" pc_by_slot=L{pc_by_slot['left']*100:.0f}%M{pc_by_slot['middle']*100:.0f}%R{pc_by_slot['right']*100:.0f}%"
                f" dir_conf={eval_m['direction_conflict_slot_acc']*100:.1f}%"
                f" layout={eval_m['layout_acc']*100:.1f}%"
                f" pres_slot={eval_m['preserve_slot_acc']*100:.1f}%"
                f" pres_ord={eval_m['preserve_ord_excl_acc']*100:.1f}%",
                flush=True,
            )
            # Prediction histograms — critical collapse check
            print(
                f"EVAL step={step} [PRED_HIST]"
                f" typeA={dict(eval_m['prediction_histogram_type_a'])}"
                f" phrase_conflict={dict(eval_m['prediction_histogram_phrase_conflict'])}"
                f" dir_conflict={dict(eval_m['prediction_histogram_dir_conflict'])}",
                flush=True,
            )
            # Ground-truth distributions
            print(
                f"EVAL step={step} [DIST]"
                f" type_A_slot={dict(eval_m['dist_type_a_resolved_slot'])}"
                f" phrase_conflict_slot={dict(eval_m['dist_phrase_conflict_slot'])}"
                f" layouts={dict(eval_m['dist_layout'])}",
                flush=True,
            )
            # per-phrase prediction breakdown (detect direction-heuristic collapse)
            print(f"EVAL step={step} [PHRASE_PRED] {dict(eval_m['phrase_to_pred_slots'])}",
                  flush=True)
            # Full JSON dump of scalar metrics
            scalar_metrics = {
                k: v for k, v in eval_m.items()
                if not isinstance(v, (dict, str, bool))
            }
            print("EVAL_METRICS_JSON:", json.dumps(scalar_metrics, indent=2), flush=True)
            if wandb_run is not None:
                import wandb
                wandb.log({f"eval/{k}": v for k, v in scalar_metrics.items()}, step=step)
                # Key v2 success metrics
                wandb.log({
                    "eval/phrase_conflict_acc_balanced":         eval_m["phrase_conflict_acc_balanced"],
                    "eval/phrase_conflict_acc_raw":              eval_m["phrase_conflict_acc_raw"],
                    "eval/color_neighbor_slot_acc_balanced":     eval_m["color_neighbor_slot_acc_balanced"],
                    "eval/color_neighbor_slot_acc_raw":          eval_m["color_neighbor_slot_acc_raw"],
                    "eval/direction_conflict_slot_acc":          eval_m["direction_conflict_slot_acc"],
                    # Prediction histogram (collapse detection)
                    "eval/pred_hist_pc_left":    eval_m["prediction_histogram_phrase_conflict"].get("left", 0),
                    "eval/pred_hist_pc_middle":  eval_m["prediction_histogram_phrase_conflict"].get("middle", 0),
                    "eval/pred_hist_pc_right":   eval_m["prediction_histogram_phrase_conflict"].get("right", 0),
                    "eval/pred_hist_typeA_left":   eval_m["prediction_histogram_type_a"].get("left", 0),
                    "eval/pred_hist_typeA_middle": eval_m["prediction_histogram_type_a"].get("middle", 0),
                    "eval/pred_hist_typeA_right":  eval_m["prediction_histogram_type_a"].get("right", 0),
                    # Per-slot phrase-conflict accuracy
                    "eval/pc_acc_left":   pc_by_slot["left"],
                    "eval/pc_acc_middle": pc_by_slot["middle"],
                    "eval/pc_acc_right":  pc_by_slot["right"],
                }, step=step)
            policy.train()
            heads.train()

        # ── Checkpoint ────────────────────────────────────────────────────────
        if step in save_steps or step == args.steps:
            last_ckpt = save_s1_ckpt(out_dir, step, policy, heads, init_path, run_report)
            print(f"CHECKPOINT step={step} path={last_ckpt}", flush=True)

    print(f"S1_DONE last_checkpoint={last_ckpt}", flush=True)
    if wandb_run is not None:
        wandb_run.finish()


def _get_hidden_size(policy) -> int:
    """Detect VLM text hidden size."""
    raw = policy.get_base_model() if hasattr(policy, "get_base_model") else policy
    return raw.model.vlm_with_expert.config.text_config.hidden_size


# ─────────────────────────────────────────────────────────────────────────────
# Argument parsing
# ─────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Stage S1: Semantic-only color-neighbor alignment from Stage2-20k"
    )
    # Paths
    p.add_argument("--init", default=DEFAULT_INIT,
                   help="Stage2-20k checkpoint (used as both frozen reference and student init)")
    p.add_argument("--dataset", default=DEFAULT_DATASET)
    p.add_argument("--dataset-root", type=Path, default=Path(DEFAULT_DATASET_ROOT))
    p.add_argument("--out-dir", type=Path, default=Path(DEFAULT_OUT))

    # Split
    p.add_argument("--train-reps", default="0,1,2,3",
                   help="Replicate indices to use for training")
    p.add_argument("--val-reps", default="4,5",
                   help="Replicate indices to use for validation")

    # LoRA
    p.add_argument("--lora-rank", type=int, default=16)
    p.add_argument("--lora-alpha", type=int, default=32)
    p.add_argument("--lora-dropout", type=float, default=0.05)

    # Training schedule
    p.add_argument("--steps", type=int, default=1000)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--eval-batch-size", type=int, default=32)
    p.add_argument("--lora-lr", type=float, default=1e-5)
    p.add_argument("--head-lr", type=float, default=5e-5)
    p.add_argument("--weight-decay", type=float, default=1e-10)
    p.add_argument("--warmup-steps", type=int, default=100)
    p.add_argument("--grad-clip-norm", type=float, default=10.0)
    p.add_argument("--save-steps", default="100,200,300,500,1000",
                   help="Comma-separated step numbers at which to save checkpoints")
    p.add_argument("--eval-every", type=int, default=200)
    p.add_argument("--log-every", type=int, default=25)

    # Misc
    p.add_argument("--seed", type=int, default=47)
    p.add_argument("--use-amp", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--dry-inspect", action="store_true",
                   help="Print dataset stats and case distributions, then exit")
    p.add_argument("--wandb", action="store_true")
    p.add_argument("--wandb-project", default="lerobot")
    p.add_argument("--wandb-name", default="eval2-s1-color-neighbor-semantic-align-v1")

    return p.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    init_path = Path(args.init)
    out_dir = Path(args.out_dir)
    train_reps = {int(x) for x in args.train_reps.split(",") if x.strip()}
    val_reps = {int(x) for x in args.val_reps.split(",") if x.strip()}

    # ── Dataset ───────────────────────────────────────────────────────────────
    print(f"Loading dataset: {args.dataset} from {args.dataset_root}", flush=True)
    base_ds = LeRobotDataset(args.dataset, root=args.dataset_root, video_backend="pyav")
    truth = load_episode_truth(args.dataset, args.dataset_root)
    frames = first_frames(base_ds, truth)
    cases = build_s1_cases(frames, train_reps, val_reps)
    cases = mark_phrase_conflicts(cases)   # fill phrase_conflict=True for true anti-shortcuts
    train_summary = summarize_cases(frames, cases, "train")
    val_summary = summarize_cases(frames, cases, "val")
    print("TRAIN_SUMMARY:", json.dumps(train_summary, indent=2), flush=True)
    print("VAL_SUMMARY:  ", json.dumps(val_summary, indent=2), flush=True)

    # ── Phrase-conflict (true anti-shortcut) validation ──────────────────────
    # Show 20 phrase-conflict val cases: same phrase_key → different target slots
    print("\nPHRASE_CONFLICT CASE EXAMPLES (true anti-shortcut — same phrase, different layout):")
    pc_cases = [(c.phrase_key, c.layout_code, ID_TO_SLOT[c.target_slot_id], c.student_prompt)
                for c in cases if c.split == "val" and c.sample_type == TYPE_A and c.phrase_conflict]
    # Group by phrase_key to show contrast
    from collections import defaultdict as _dd
    pc_by_phrase: dict = _dd(list)
    for pk, lc, sl, sp in pc_cases:
        pc_by_phrase[pk].append((lc, sl, sp))
    shown = 0
    for pk, entries in sorted(pc_by_phrase.items()):
        if shown >= 6:
            break
        for lc, sl, sp in entries[:2]:
            print(f"  phrase={pk!r} layout={lc} target={sl!r} | {sp[:70]}", flush=True)
        shown += 1

    # Also show direction-conflict examples for reference (always-middle)
    print("\nDIRECTION_CONFLICT CASE EXAMPLES (always-middle, secondary metric):")
    dc_cases = [(c.student_prompt, c.layout_code, ID_TO_SLOT[c.target_slot_id])
                for c in cases if c.split == "val" and c.sample_type == TYPE_A
                and c.direction_conflict][:6]
    for prompt, lc, slot in dc_cases:
        print(f"  layout={lc} target={slot!r} | {prompt[:80]}", flush=True)

    # Verify phrase_conflict target-slot distribution (should NOT be all-middle)
    pc_slot_dist = Counter(ID_TO_SLOT[c.target_slot_id]
                           for c in cases if c.split == "val"
                           and c.sample_type == TYPE_A and c.phrase_conflict)
    dc_slot_dist = Counter(ID_TO_SLOT[c.target_slot_id]
                           for c in cases if c.split == "val"
                           and c.sample_type == TYPE_A and c.direction_conflict)
    print(f"\nPhrase-conflict val slot distribution:   {dict(pc_slot_dist)}", flush=True)
    print(f"Direction-conflict val slot distribution: {dict(dc_slot_dist)}", flush=True)

    if args.dry_inspect:
        print("DRY_INSPECT_DONE", flush=True)
        return

    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Frozen reference model (Stage2-20k, all frozen) ───────────────────────
    print(f"\nLoading frozen reference: {args.init}", flush=True)
    frozen_ref = SmolVLAPolicy.from_pretrained(str(init_path))
    frozen_ref.config.device = device
    frozen_ref = frozen_ref.to(device).eval()
    for p in frozen_ref.parameters():
        p.requires_grad = False
    n_ref = sum(p.numel() for p in frozen_ref.parameters())
    print(f"  Frozen reference: {n_ref:,} params (all frozen)", flush=True)

    # ── Student policy (Stage2-20k + VLM q/v LoRA) ───────────────────────────
    print(f"\nLoading student (from same init + LoRA): {args.init}", flush=True)
    policy = SmolVLAPolicy.from_pretrained(str(init_path))
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
    configure_s1_trainability(policy)
    policy = policy.to(device)

    hidden_size = _get_hidden_size(policy)
    heads = SemanticHeads(hidden_size).to(device)
    print(f"  SemanticHeads hidden_size={hidden_size}", flush=True)

    n_lora = sum(p.numel() for n, p in policy.named_parameters() if "lora_" in n and p.requires_grad)
    n_heads = sum(p.numel() for p in heads.parameters())
    print(f"  Student trainable — LoRA: {n_lora:,}  Heads: {n_heads:,}", flush=True)

    # ── Preprocessor ──────────────────────────────────────────────────────────
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

    # ── Run report ────────────────────────────────────────────────────────────
    run_report = {
        "stage": "S1-semantic-only-color-neighbor-alignment",
        "purpose": (
            "Fix color-neighbor relative prompts only. "
            "No action loss. No action tensors. No connector/bridge training. "
            "Semantic-only S1."
        ),
        "init": str(init_path),
        "dataset": args.dataset,
        "dataset_root": str(args.dataset_root),
        "distillation_tensor_name": DISTILLATION_TENSOR_NAME,
        "distillation_module_path": DISTILLATION_MODULE_PATH,
        "distillation_is_before_connector": DISTILLATION_IS_BEFORE_CONNECTOR,
        "trainable_modules": ["VLM q/v LoRA", "SemanticHeads (slot_head, layout_head)"],
        "frozen_modules": [
            "vision encoder",
            "connector / bridge",
            "action expert / lm_expert",
            "state_proj",
            "action_in_proj",
            "action_out_proj",
            "action_time_mlp_in",
            "action_time_mlp_out",
            "base VLM weights outside LoRA",
        ],
        "loss_weights": {
            "L_repr_distill (Type A)": WEIGHT_DISTILL,
            "L_preserve_B (slot)": WEIGHT_PRESERVE_B,
            "L_scaffold (Type C)": WEIGHT_SCAFFOLD,
            "L_preserve_D (ord/excl)": WEIGHT_PRESERVE_D,
            "L_slot_CE (Type A heads)": WEIGHT_SLOT_CE,
            "L_layout_CE (Type A heads)": WEIGHT_LAYOUT_CE,
            "L_phrase_conflict_CE (hard subset)": WEIGHT_PHRASE_CONFLICT_CE,
        },
        "sample_fractions": {
            "Type A (color_neighbor_distill)": FRAC_A,
            "Type B (slot_preserve)": FRAC_B,
            "Type C (scaffold)": FRAC_C,
            "Type D (ord_excl_preserve)": FRAC_D,
        },
        "phrase_conflict_frac_within_type_A": PHRASE_CONFLICT_FRAC,
        "anti_shortcut_note": (
            "anti_shortcut redefined as phrase_conflict: same phrase_key resolves "
            "to >=2 different target slots across layouts. Old direction_conflict "
            "(always-middle) is still tracked but is not the primary metric."
        ),
        "hard_restrictions_verified": [
            "NO action loss",
            "NO action tensors (noise/time/action_out_proj)",
            "NO VelocityCapture",
            "NO velocity MSE",
            "NO phase2-start / phase2-end",
            "NO branch-point frame selection",
            "NO connector/bridge training",
        ],
        "train_summary": train_summary,
        "val_summary": val_summary,
        "args": vars(args),
    }

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "s1_run_report.json").write_text(json.dumps(run_report, indent=2, default=_json_default) + "\n")

    # ── Build datasets ────────────────────────────────────────────────────────
    train_ds = ColorNeighborS1Dataset(frames, cases, "train")
    val_ds = ColorNeighborS1Dataset(frames, cases, "val")

    # ── Run S1 ────────────────────────────────────────────────────────────────
    print(f"\n{'='*60}", flush=True)
    print(f"  STAGE S1: Semantic-Only Color-Neighbor Alignment", flush=True)
    print(f"  Steps:       {args.steps}", flush=True)
    print(f"  LoRA LR:     {args.lora_lr}", flush=True)
    print(f"  Head LR:     {args.head_lr}", flush=True)
    print(f"  Batch size:  {args.batch_size}", flush=True)
    print(f"  Save steps:  {args.save_steps}", flush=True)
    print(f"  Output:      {out_dir}", flush=True)
    print(f"{'='*60}\n", flush=True)

    run_s1(
        policy=policy,
        frozen_ref=frozen_ref,
        heads=heads,
        preprocessor=preprocessor,
        ds=base_ds,
        train_ds=train_ds,
        val_ds=val_ds,
        args=args,
        out_dir=out_dir,
        init_path=init_path,
        run_report=run_report,
        device=device,
    )


if __name__ == "__main__":
    main()
