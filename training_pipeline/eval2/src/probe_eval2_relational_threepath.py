#!/usr/bin/env python3
"""Three-path Eval2 relational diagnostic.

Reports, per fixed image/state frame and prompt:
  1. Standalone strict10 VLM target-slot answer.
  2. VLA-loaded VLM target-slot answer from the actual checkpoint.
  3. Slot-head prediction and action-nearest ordinal anchor.

No robot interaction and no training.
"""

from __future__ import annotations

import argparse
import os
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch import nn
from transformers import AutoModelForImageTextToText, AutoProcessor

from eval2_layout_utils import SLOTS, layout_id, normalize_layout, resolve_prompt_target_color, target_slot
from lerobot.common.control_utils import predict_action
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies.factory import make_pre_post_processors
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy


STRICT10_ID = "robot-learning-group47/smolvlm2-500m-vqa-init-position-strict10-color-slot-merged"
DATASET_REPO = "robot-learning-group47/eval2_compositional_augmented"
DATASET_ROOT = "/home/shadeform/lerobot_cache/robot-learning-group47/eval2_compositional_augmented"
DEFAULT_CKPT_BASE = "/home/shadeform/outputs/eval2_stage2_compositional_015000_balanced_r2/checkpoints"
CHUNK = 50
SLOT_TO_ID = {slot: i for i, slot in enumerate(SLOTS)}
ID_TO_SLOT = {i: slot for slot, i in SLOT_TO_ID.items()}

SYSTEM = (
    "You are looking at a robot tabletop scene with exactly three bowls: blue, red, and green. "
    "Given the image and the robot instruction, identify the target bowl. "
    "Reply with exactly two short lines, not JSON and not prose. "
    "The exact format is:\n"
    "target color: green\n"
    "target slot: right"
)

PROMPTS = [
    "Put the banana into the bowl on the left of the red bowl from the robot perspective.",
    "Put the banana into the bowl on the right of the red bowl from the robot perspective.",
    "Put the banana into the bowl on the left of the green bowl from the robot perspective.",
    "Put the banana into the bowl on the right of the blue bowl from the robot perspective.",
]

COMBOS = [
    ("bgr", "blue", 0),
    ("bgr", "green", 6),
    ("bgr", "red", 12),
    ("brg", "blue", 18),
    ("brg", "green", 24),
    ("brg", "red", 30),
    ("gbr", "blue", 36),
    ("gbr", "green", 42),
    ("gbr", "red", 48),
    ("grb", "blue", 54),
    ("grb", "green", 60),
    ("grb", "red", 66),
    ("rbg", "blue", 72),
    ("rbg", "green", 78),
    ("rbg", "red", 84),
    ("rgb", "blue", 90),
    ("rgb", "green", 96),
    ("rgb", "red", 102),
]


@dataclass(frozen=True)
class FrameInfo:
    global_index: int
    episode_index: int
    orig_episode_index: int
    layout: list[str]
    layout_id: str


def parse_vlm_output(raw: str) -> dict[str, str]:
    text = raw.lower()
    out: dict[str, str] = {"raw": raw.strip()}
    m_color = re.search(r"target\s+color\s*[:\-]\s*(blue|red|green)", text)
    m_slot = re.search(r"target\s+slot\s*[:\-]\s*(left|middle|right)", text)
    if m_color:
        out["color"] = m_color.group(1)
    if m_slot:
        out["slot"] = m_slot.group(1)
    if "color" not in out:
        colors = sorted(set(re.findall(r"\b(blue|red|green)\b", text)))
        if len(colors) == 1:
            out["color"] = colors[0]
    if "slot" not in out:
        slots = sorted(set(re.findall(r"\b(left|middle|right)\b", text)))
        if len(slots) == 1:
            out["slot"] = slots[0]
    return out


def to_pil(image) -> Image.Image:
    if isinstance(image, Image.Image):
        return image.convert("RGB")
    if isinstance(image, torch.Tensor):
        image = image.detach().cpu().numpy()
    if image.ndim == 3 and image.shape[0] in (1, 3):
        image = image.transpose(1, 2, 0)
    if image.dtype != np.uint8:
        image = (np.clip(image, 0, 1) * 255).astype(np.uint8)
    return Image.fromarray(image).convert("RGB")


def to_numpy_obs(sample: dict) -> dict:
    image = sample["observation.images.front"]
    state = sample["observation.state"]
    if isinstance(image, torch.Tensor):
        image = image.detach().cpu().numpy()
    if isinstance(state, torch.Tensor):
        state = state.detach().cpu().numpy()
    if image.ndim == 3 and image.shape[0] in (1, 3):
        image = image.transpose(1, 2, 0)
    if image.dtype != np.uint8:
        image = (np.clip(image, 0, 1) * 255).astype(np.uint8)
    return {
        "observation.images.front": image,
        "observation.state": state.astype(np.float32),
        "observation.images.empty_camera_0": np.zeros((480, 640, 3), np.uint8),
        "observation.images.empty_camera_1": np.zeros((480, 640, 3), np.uint8),
    }


def augmented_episode_info(episode_index: int) -> tuple[list[str], int]:
    cursor = 0
    for layout_code, color, orig_start in COMBOS:
        layout = normalize_layout(layout_code)
        prompts_per_demo = 8 if color == layout[1] else 6
        next_cursor = cursor + 6 * prompts_per_demo
        if cursor <= episode_index < next_cursor:
            orig_episode = orig_start + ((episode_index - cursor) // prompts_per_demo)
            return layout, orig_episode
        cursor = next_cursor
    raise ValueError(f"Episode {episode_index} outside augmented mapping")


def pick_frames(ds: LeRobotDataset) -> list[FrameInfo]:
    # bgr makes all requested example prompts valid. Add two more layouts for contrast.
    desired = ["bgr", "gbr", "rgb"]
    found: dict[str, FrameInfo] = {}
    for global_idx, ep_value in enumerate(ds.hf_dataset["episode_index"]):
        ep = int(ep_value)
        layout, orig = augmented_episode_info(ep)
        code = "".join(color[0] for color in layout)
        if code in desired and code not in found:
            found[code] = FrameInfo(global_idx, ep, orig, layout, layout_id(layout))
        if len(found) == len(desired):
            break
    return [found[code] for code in desired if code in found]


def expected_for_prompt(prompt: str, layout: list[str]) -> tuple[str, str] | None:
    try:
        color = resolve_prompt_target_color(prompt, layout)
    except Exception:
        return None
    return color, target_slot(layout, color)


def build_vlm_prompt(processor, instruction: str) -> str:
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": f"{SYSTEM}\n\nInstruction: {instruction}"},
            ],
        }
    ]
    return processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def ask_vlm(model, processor, image: Image.Image, prompt: str, device, max_new_tokens: int) -> dict[str, str]:
    text = build_vlm_prompt(processor, prompt)
    inputs = processor(text=[text], images=[[image]], return_tensors="pt", padding=True)
    inputs = {key: value.to(device) for key, value in inputs.items()}
    with torch.inference_mode():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=processor.tokenizer.eos_token_id,
        )
    input_len = inputs["input_ids"].shape[1]
    raw = processor.batch_decode(out[:, input_len:], skip_special_tokens=True)[0]
    return parse_vlm_output(raw)


def add_slot_hook(policy, ckpt_dir: Path, storage: dict):
    slot_head = nn.Linear(policy.model.vlm_with_expert.expert_hidden_size, 3).to(policy.config.device)
    slot_head.load_state_dict(torch.load(ckpt_dir / "target_slot_head.pt", map_location=policy.config.device))
    slot_head.eval()

    def hook(_module, inputs, _output):
        suffix_out = inputs[0].float()
        with torch.no_grad():
            storage["slot_logits"] = slot_head(suffix_out.mean(dim=1))

    return policy.model.action_out_proj.register_forward_hook(hook), slot_head


def run_action_prompt(policy, pre, post, obs, prompt: str, storage: dict, device) -> dict:
    policy.reset()
    pre.reset()
    post.reset()
    storage.clear()
    actions = []
    slot_pred = None
    for i in range(CHUNK):
        action = predict_action(
            observation=obs,
            policy=policy,
            device=device,
            preprocessor=pre,
            postprocessor=post,
            use_amp=False,
            task=prompt,
        )
        arr = action.detach().cpu().numpy()
        if arr.ndim > 1:
            arr = arr.squeeze(0)
        actions.append(arr)
        if i == 0 and "slot_logits" in storage:
            logits = storage["slot_logits"].detach().cpu().numpy()
            if logits.ndim > 1:
                logits = logits[0]
            slot_pred = int(np.argmax(logits))
    return {
        "actions": np.stack(actions),
        "slot_pred": ID_TO_SLOT.get(slot_pred, "?"),
    }


def nearest_anchor(pan: float, anchors: dict[str, float]) -> str:
    return min(anchors, key=lambda slot: abs(anchors[slot] - pan))


def diagnose(expected: str | None, standalone: str, vla_vlm: str, slot: str, action: str) -> str:
    if expected is None:
        return "invalid prompt for layout"
    if standalone != expected:
        return "standalone VLM relational failure"
    if vla_vlm != expected:
        return "VLA-loaded VLM relational failure"
    if slot != expected:
        return "slot-head/latent relational failure"
    if action != expected:
        return "action-side routing failure"
    return "all paths correct"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpts", nargs="+", default=["010000", "020000", "030000"])
    parser.add_argument("--ckpt-base", default=DEFAULT_CKPT_BASE)
    parser.add_argument("--dataset-root", default=DATASET_ROOT)
    parser.add_argument("--max-new-tokens", type=int, default=32)
    args = parser.parse_args()

    os.environ.setdefault("HF_LEROBOT_HOME", "/home/shadeform/lerobot_cache")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    print(f"Device: {device}")

    print("Loading dataset")
    ds = LeRobotDataset(DATASET_REPO, root=args.dataset_root, video_backend="pyav")
    frames = pick_frames(ds)
    samples = [(fi, ds[fi.global_index]) for fi in frames]
    for fi, _ in samples:
        print(f"Frame global={fi.global_index} ep={fi.episode_index} orig={fi.orig_episode_index} layout={fi.layout_id}")

    print(f"Loading standalone strict10 VLM: {STRICT10_ID}")
    standalone_processor = AutoProcessor.from_pretrained(STRICT10_ID)
    standalone_processor.tokenizer.padding_side = "left"
    standalone_vlm = AutoModelForImageTextToText.from_pretrained(
        STRICT10_ID,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
    ).to(device).eval()

    print(
        "checkpoint\tlayout\tprompt\texpected_target_position\t"
        "standalone_vlm_answer\tvla_loaded_vlm_answer\tslot_head_predicted_target_position\t"
        "action_nearest_ordinal_anchor\tdiagnosis"
    )

    for step in args.ckpts:
        ckpt_dir = Path(args.ckpt_base) / step / "pretrained_model"
        if not (ckpt_dir / "model.safetensors").exists():
            print(f"{step}\tMISSING\t-\t-\t-\t-\t-\t-\tmissing checkpoint")
            continue

        policy = SmolVLAPolicy.from_pretrained(str(ckpt_dir))
        policy.config.device = str(device)
        policy = policy.to(device).eval()
        vla_processor = policy.model.vlm_with_expert.processor
        vla_processor.tokenizer.padding_side = "left"
        vla_vlm = policy.model.vlm_with_expert.vlm.eval()
        pre, post = make_pre_post_processors(
            policy.config,
            pretrained_path=str(ckpt_dir),
            preprocessor_overrides={"device_processor": {"device": str(device)}},
        )
        storage: dict = {}
        hook, _slot_head = add_slot_hook(policy, ckpt_dir, storage)

        for fi, sample in samples:
            image = to_pil(sample["observation.images.front"])
            obs = to_numpy_obs(sample)
            anchors = {}
            for idx, slot in enumerate(SLOTS):
                anchor_prompt = (
                    f"Put the banana into the {('1st', '2nd', '3rd')[idx]} "
                    "bowl from the left from the robot perspective."
                )
                r = run_action_prompt(policy, pre, post, obs, anchor_prompt, storage, device)
                anchors[slot] = float(r["actions"][:10, 0].mean())

            for prompt in PROMPTS:
                expected = expected_for_prompt(prompt, fi.layout)
                if expected is None:
                    expected_slot = None
                    expected_text = "INVALID"
                else:
                    _expected_color, expected_slot = expected
                    expected_text = expected_slot

                standalone = ask_vlm(
                    standalone_vlm,
                    standalone_processor,
                    image,
                    prompt,
                    device,
                    args.max_new_tokens,
                )
                vla_answer = ask_vlm(
                    vla_vlm,
                    vla_processor,
                    image,
                    prompt,
                    device,
                    args.max_new_tokens,
                )
                action_res = run_action_prompt(policy, pre, post, obs, prompt, storage, device)
                pan_m10 = float(action_res["actions"][:10, 0].mean())
                action_anchor = nearest_anchor(pan_m10, anchors)
                slot_pred = action_res["slot_pred"]
                diagnosis = diagnose(
                    expected_slot,
                    standalone.get("slot", "?"),
                    vla_answer.get("slot", "?"),
                    slot_pred,
                    action_anchor,
                )
                prompt_short = prompt.replace("Put the banana into the ", "").replace(" from the robot perspective.", "")
                standalone_text = f"{standalone.get('slot','?')}/{standalone.get('color','?')}"
                vla_text = f"{vla_answer.get('slot','?')}/{vla_answer.get('color','?')}"
                print(
                    f"{step}\t{fi.layout_id}\t{prompt_short}\t{expected_text}\t"
                    f"{standalone_text}\t{vla_text}\t{slot_pred}\t{action_anchor}\t{diagnosis}"
                )

        hook.remove()
        del policy
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
