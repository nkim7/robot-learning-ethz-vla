"""Bridge loss patch for SmolVLA end-to-end celebrity→position training.

Two losses added on top of the action prediction loss:
  1. Alignment: MSE(student_suffix_out, teacher_suffix_out.detach())
     Pulls the celebrity-prompt action hidden state toward the spatial-prompt one.
  2. Spatial CE: CrossEntropy(spatial_head(suffix_out.mean(1)), target_slot_id)
     Directly supervises slot prediction from the action hidden state.

VLM LoRA must be loaded frozen (run1_lora_patch with EVAL3_VLM_LORA_TRAINABLE=0).
Teacher forward uses spatial prompts; student uses celebrity prompts from the batch.

Slot label rule:  within_block = episode_index % 30
    0–9   → left (id=0)   → teacher prompt: "Put the coke on left."
    10–19 → middle (id=1) → teacher prompt: "Put the coke on middle."
    20–29 → right (id=2)  → teacher prompt: "Put the coke on right."

Env vars:
  EVAL3_BRIDGE_SIDECAR_PATH    Path to target_slot_by_episode.json
  EVAL3_BRIDGE_ALIGN_LAMBDA    Weight of alignment loss (default: 0.1)
  EVAL3_BRIDGE_SPATIAL_LAMBDA  Weight of spatial CE loss (default: 1.0)
  EVAL3_BRIDGE_SPATIAL_HEAD_LR Learning rate for spatial_head (default: 1e-4)
  EVAL3_BRIDGE_MAX_LANG_LEN    Language token max length (default: 48)
"""
from __future__ import annotations

import json
import logging
import os
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)

ALIGN_LAMBDA = float(os.environ.get("EVAL3_BRIDGE_ALIGN_LAMBDA", "0.1"))
SPATIAL_LAMBDA = float(os.environ.get("EVAL3_BRIDGE_SPATIAL_LAMBDA", "1.0"))
SPATIAL_HEAD_LR = float(os.environ.get("EVAL3_BRIDGE_SPATIAL_HEAD_LR", "1e-4"))
MAX_LANG_LEN = int(os.environ.get("EVAL3_BRIDGE_MAX_LANG_LEN", "48"))
_SIDECAR_PATH = os.environ.get(
    "EVAL3_BRIDGE_SIDECAR_PATH",
    "data/eval3_90_permutation_spatial_labels/target_slot_by_episode.json",
)

SLOT_TO_ID = {"left": 0, "middle": 1, "right": 2}
ID_TO_SLOT = {v: k for k, v in SLOT_TO_ID.items()}
SLOT_TO_ACTION_PROMPT = {
    "left": "Put the coke on left.",
    "middle": "Put the coke on middle.",
    "right": "Put the coke on right.",
}
EXPERT_HIDDEN_DIM = None  # resolved at init time from action_out_proj.in_features


def _slot_from_episode(episode_index: int) -> str:
    """Arithmetic label rule — does not require the sidecar file."""
    wb = episode_index % 30
    if wb <= 9:
        return "left"
    if wb <= 19:
        return "middle"
    return "right"


class _SidecarLabels:
    def __init__(self, path: str) -> None:
        resolved = Path(path)
        if not resolved.is_absolute():
            resolved = Path.cwd() / resolved
        if resolved.exists():
            raw = json.loads(resolved.read_text())
            self._map: dict[int, int] = {int(k): SLOT_TO_ID[v] for k, v in raw.items()}
            logger.info("Bridge sidecar loaded: %d episodes from %s", len(self._map), resolved)
        else:
            logger.warning(
                "Bridge sidecar not found at %s — falling back to arithmetic rule.", resolved
            )
            self._map = {}

    def slot_id(self, episode_index: int) -> int:
        # Arithmetic rule works for all indices including repeated episodes.
        # Sidecar is only used as a cross-check; fall back cleanly if missing.
        return SLOT_TO_ID[_slot_from_episode(episode_index)]


_labels: _SidecarLabels | None = None


def _get_labels() -> _SidecarLabels:
    global _labels
    if _labels is None:
        _labels = _SidecarLabels(_SIDECAR_PATH)
    return _labels


def apply_bridge_loss_patch() -> None:
    from lerobot.policies.smolvla import modeling_smolvla
    from lerobot.utils.constants import OBS_LANGUAGE_ATTENTION_MASK, OBS_LANGUAGE_TOKENS

    if getattr(modeling_smolvla.SmolVLAPolicy, "_eval3_bridge_patch", False):
        return

    VLAFlowMatching = modeling_smolvla.VLAFlowMatching

    # ── 1. Register spatial_head on VLAFlowMatching ──────────────────────────
    _orig_vla_init = VLAFlowMatching.__init__

    def _patched_vla_init(self, *args, **kwargs):
        _orig_vla_init(self, *args, **kwargs)
        expert_dim = self.action_out_proj.in_features  # e.g. 1536 for 2.2B, 720 for 500M
        self.spatial_head = nn.Linear(expert_dim, 3)
        nn.init.normal_(self.spatial_head.weight, std=0.02)
        nn.init.zeros_(self.spatial_head.bias)
        logger.info("Bridge patch: spatial_head(%d→3) registered on VLAFlowMatching", expert_dim)

    VLAFlowMatching.__init__ = _patched_vla_init

    # ── 2. Separate LR group for spatial_head ────────────────────────────────
    _orig_get_optim_params = modeling_smolvla.SmolVLAPolicy.get_optim_params

    def _patched_get_optim_params(self):
        spatial_params, other_params = [], []
        for name, param in self.named_parameters():
            if not param.requires_grad:
                continue
            if "spatial_head" in name:
                spatial_params.append(param)
            else:
                other_params.append(param)
        groups = []
        if other_params:
            groups.append({"params": other_params, "name": "action_expert"})
        if spatial_params:
            groups.append({"params": spatial_params, "lr": SPATIAL_HEAD_LR, "name": "spatial_head"})
        logger.info(
            "Bridge optimizer groups: action_expert=%d params  spatial_head=%d params (lr=%s)",
            sum(p.numel() for p in other_params),
            sum(p.numel() for p in spatial_params),
            SPATIAL_HEAD_LR,
        )
        return groups

    modeling_smolvla.SmolVLAPolicy.get_optim_params = _patched_get_optim_params

    # ── 3. Patch forward with bridge losses ───────────────────────────────────
    _orig_forward = modeling_smolvla.SmolVLAPolicy.forward

    def _patched_forward(self, batch, **kwargs):
        if not torch.is_grad_enabled():
            return _orig_forward(self, batch, **kwargs)

        episode_indices = batch.get("episode_index")
        if episode_indices is None:
            logger.warning("Bridge patch: 'episode_index' not in batch; skipping bridge losses")
            return _orig_forward(self, batch, **kwargs)

        labels = _get_labels()
        processor = self.model.vlm_with_expert.processor
        device = next(self.parameters()).device

        ep_list = episode_indices.view(-1).tolist()
        slot_ids = [labels.slot_id(int(ep)) for ep in ep_list]
        spatial_tasks = [SLOT_TO_ACTION_PROMPT[ID_TO_SLOT[sid]] for sid in slot_ids]
        target_slots = torch.tensor(slot_ids, dtype=torch.long, device=device)

        # ── Student forward (celebrity prompts) ──────────────────────────────
        _student: dict[str, torch.Tensor] = {}

        def _student_hook(module, inp, out):
            _student["suffix_out"] = inp[0]

        hook_s = self.model.action_out_proj.register_forward_hook(_student_hook)
        try:
            action_loss, loss_dict = _orig_forward(self, batch, **kwargs)
        finally:
            hook_s.remove()

        student_suffix = _student.get("suffix_out")  # (B, n_action_steps, 720)
        if student_suffix is None:
            logger.warning("Bridge patch: student hook did not fire; skipping bridge losses")
            return action_loss, loss_dict

        # ── Teacher forward (spatial prompts, no grad) ────────────────────────
        tok = processor.tokenizer(
            spatial_tasks,
            padding="max_length",
            max_length=MAX_LANG_LEN,
            truncation=True,
            return_tensors="pt",
        )
        teacher_batch = dict(batch)
        teacher_batch[OBS_LANGUAGE_TOKENS] = tok["input_ids"].to(device)
        teacher_batch[OBS_LANGUAGE_ATTENTION_MASK] = tok["attention_mask"].to(device, dtype=torch.bool)

        _teacher: dict[str, torch.Tensor] = {}

        def _teacher_hook(module, inp, out):
            _teacher["suffix_out"] = inp[0]

        hook_t = self.model.action_out_proj.register_forward_hook(_teacher_hook)
        try:
            with torch.no_grad():
                _orig_forward(self, teacher_batch, **kwargs)
        finally:
            hook_t.remove()

        teacher_suffix = _teacher.get("suffix_out")
        if teacher_suffix is None:
            logger.warning("Bridge patch: teacher hook did not fire; skipping bridge losses")
            return action_loss, loss_dict

        # ── Compute bridge losses ─────────────────────────────────────────────
        align_loss = F.mse_loss(student_suffix, teacher_suffix.detach().to(student_suffix.dtype))

        pooled = student_suffix.mean(dim=1)  # (B, 720)
        spatial_logits = self.model.spatial_head(
            pooled.to(self.model.spatial_head.weight.dtype)
        )  # (B, 3)
        spatial_loss = F.cross_entropy(spatial_logits, target_slots)

        total_loss = (
            action_loss
            + ALIGN_LAMBDA * align_loss.to(action_loss.dtype)
            + SPATIAL_LAMBDA * spatial_loss.to(action_loss.dtype)
        )

        loss_dict["action_loss_only"] = action_loss.item()
        loss_dict["align_loss"] = align_loss.item()
        loss_dict["spatial_loss"] = spatial_loss.item()
        loss_dict["spatial_acc"] = (spatial_logits.argmax(1) == target_slots).float().mean().item()

        return total_loss, loss_dict

    modeling_smolvla.SmolVLAPolicy.forward = _patched_forward
    modeling_smolvla.SmolVLAPolicy._eval3_bridge_patch = True

    logger.info(
        "Bridge loss patch applied | align_lambda=%.3f | spatial_lambda=%.3f | spatial_head_lr=%s",
        ALIGN_LAMBDA,
        SPATIAL_LAMBDA,
        SPATIAL_HEAD_LR,
    )
