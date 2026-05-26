"""Auxiliary VLM Q&A loss patch for end-to-end SmolVLA training.

Patches SmolVLAPolicy.forward to add a position Q&A cross-entropy loss
on top of the action prediction loss. This preserves VLM position accuracy
while co-training the action expert and VLM LoRA.

Env vars:
  EVAL3_AUX_VLM_LAMBDA       Weight of the auxiliary loss (default: 0.5)
  EVAL3_AUX_VLM_QA_DATASET   Path to the position QA jsonl (default: data/eval3_position_qa_3frames/all.jsonl)
  EVAL3_AUX_VLM_BATCH_SIZE   Q&A examples per training step (default: 2)
"""
from __future__ import annotations

import json
import logging
import os
import random
from pathlib import Path

import torch
from PIL import Image

logger = logging.getLogger(__name__)

AUX_VLM_LAMBDA = float(os.environ.get("EVAL3_AUX_VLM_LAMBDA", "0.5"))
AUX_VLM_BATCH_SIZE = int(os.environ.get("EVAL3_AUX_VLM_BATCH_SIZE", "2"))
_AUX_VLM_QA_DATASET_PATH = os.environ.get(
    "EVAL3_AUX_VLM_QA_DATASET",
    "data/eval3_position_qa_3frames/all.jsonl",
)

_PROBE_PROMPT = (
    "Place the Coke on {name}, there are three celebrity cards in this robot scene, "
    "arranged in the left, middle, and right slots. "
    "Which slot is {name} in? "
    "Answer with exactly one word: left, center, or right."
)
# ground_truth in jsonl uses "left"/"middle"/"right"; VLM was trained to answer "center" for middle
_GT_TO_ANSWER = {"left": "left", "middle": "center", "right": "right"}


class _QAIterator:
    def __init__(self, path: str) -> None:
        resolved = Path(path)
        if not resolved.is_absolute():
            resolved = Path.cwd() / resolved
        self.records = [json.loads(l) for l in resolved.read_text().splitlines()]
        random.shuffle(self.records)
        self._idx = 0
        logger.info("Aux VLM Q&A dataset: %d records from %s", len(self.records), resolved)

    def sample(self, n: int) -> list[dict]:
        out = []
        for _ in range(n):
            out.append(self.records[self._idx % len(self.records)])
            self._idx += 1
            if self._idx % len(self.records) == 0:
                random.shuffle(self.records)
        return out


_qa_iter: _QAIterator | None = None


def _get_qa_iter() -> _QAIterator:
    global _qa_iter
    if _qa_iter is None:
        _qa_iter = _QAIterator(_AUX_VLM_QA_DATASET_PATH)
    return _qa_iter


def _build_prompt_text(processor, name: str) -> str:
    prompt = _PROBE_PROMPT.format(name=name)
    messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": prompt}]}]
    return processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def _compute_aux_vlm_loss(policy, device: torch.device, dtype: torch.dtype) -> torch.Tensor | None:
    vlm_with_expert = policy.model.vlm_with_expert
    processor = vlm_with_expert.processor
    vlm = vlm_with_expert.vlm

    records = _get_qa_iter().sample(AUX_VLM_BATCH_SIZE)
    total_loss = torch.tensor(0.0, device=device, dtype=torch.float32)
    n_valid = 0

    for rec in records:
        try:
            image = Image.open(rec["image"]).convert("RGB")
            name = rec["target_person"]
            answer = _GT_TO_ANSWER.get(rec["ground_truth"], rec["ground_truth"])

            prompt_text = _build_prompt_text(processor, name)
            full_text = prompt_text + answer

            # Find prompt token length (needed to build labels)
            p_inp = processor(text=[prompt_text], images=[[image]], return_tensors="pt")
            prompt_len = p_inp["input_ids"].shape[-1]

            # Full sequence: prompt + answer
            f_inp = processor(text=[full_text], images=[[image]], return_tensors="pt")

            # Move inputs to device/dtype
            vlm_kwargs = {}
            for k, v in f_inp.items():
                if torch.is_floating_point(v):
                    vlm_kwargs[k] = v.to(device=device, dtype=dtype)
                else:
                    vlm_kwargs[k] = v.to(device=device)

            # Labels: mask prompt with -100, supervise answer tokens only
            labels = vlm_kwargs["input_ids"].clone()
            labels[:, :prompt_len] = -100
            vlm_kwargs["labels"] = labels

            out = vlm(**vlm_kwargs)
            if out.loss is not None and not out.loss.isnan() and not out.loss.isinf():
                total_loss = total_loss + out.loss.float()
                n_valid += 1
        except Exception as exc:
            logger.debug("Aux VLM Q&A loss skipped for one record: %s", exc)

    if n_valid == 0:
        return None
    return total_loss / n_valid


def apply_aux_vlm_loss_patch() -> None:
    from lerobot.policies.smolvla import modeling_smolvla

    if getattr(modeling_smolvla.SmolVLAPolicy, "_eval3_aux_vlm_patch", False):
        return

    original_forward = modeling_smolvla.SmolVLAPolicy.forward

    def patched_forward(self, batch, **kwargs):
        action_loss, loss_dict = original_forward(self, batch, **kwargs)

        # Only add auxiliary loss during training (grad enabled)
        if not torch.is_grad_enabled():
            return action_loss, loss_dict

        try:
            vlm = self.model.vlm_with_expert.vlm
            device = next(vlm.parameters()).device
            dtype = next((p.dtype for p in vlm.parameters() if p.is_floating_point()), torch.float32)

            vlm_qa_loss = _compute_aux_vlm_loss(self, device, dtype)
            if vlm_qa_loss is not None:
                total_loss = action_loss + AUX_VLM_LAMBDA * vlm_qa_loss.to(action_loss.dtype)
                loss_dict["vlm_qa_loss"] = vlm_qa_loss.item()
                loss_dict["action_loss_only"] = action_loss.item()
            else:
                total_loss = action_loss
        except Exception as exc:
            logger.warning("Aux VLM loss failed, using action loss only: %s", exc)
            total_loss = action_loss

        return total_loss, loss_dict

    modeling_smolvla.SmolVLAPolicy.forward = patched_forward
    modeling_smolvla.SmolVLAPolicy._eval3_aux_vlm_patch = True
    logger.info(
        "Aux VLM Q&A loss patch applied | lambda=%.3f | qa_batch=%d",
        AUX_VLM_LAMBDA,
        AUX_VLM_BATCH_SIZE,
    )
