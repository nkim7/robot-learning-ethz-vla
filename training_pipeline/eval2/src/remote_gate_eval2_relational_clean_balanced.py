#!/usr/bin/env python3
"""Expanded held-out gate for clean Eval2 relational repair checkpoints."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from peft import PeftModel
from torch import nn

sys.path.insert(0, "/home/shadeform")

from eval2_layout_utils import SLOTS, normalize_layout
from lerobot.common.control_utils import predict_action
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies.factory import make_pre_post_processors
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy


LAYOUT_CODES = ("bgr", "brg", "gbr", "grb", "rbg", "rgb")
COLORS = ("blue", "green", "red")
ID_TO_SLOT = {i: slot for i, slot in enumerate(SLOTS)}
DEFAULT_DATASET = "robot-learning-group47/eval2_108_position_only"
DEFAULT_DATASET_ROOT = "/home/shadeform/lerobot_cache/robot-learning-group47/eval2_108_position_only"
DEFAULT_BASE = "/home/shadeform/outputs/eval2_stage2_compositional_015000_balanced_r2/checkpoints/020000/pretrained_model"


@dataclass(frozen=True)
class Case:
    family: str
    key: str
    prompt: str
    slot_id: int
    side: str = ""
    anchor: str = ""
    target_color: str = ""


def make_cases(code: str) -> dict[str, list[Case]]:
    layout = normalize_layout(code)
    out: dict[str, list[Case]] = {k: [] for k in ("relative", "ordinal", "exclusion", "direct")}
    for ref_idx, anchor in enumerate(layout):
        if ref_idx > 0:
            out["relative"].append(Case(
                "relative", f"left_of_{anchor}",
                f"Put the banana into the bowl on the left of the {anchor} bowl from the robot perspective.",
                ref_idx - 1, side="left", anchor=anchor, target_color=layout[ref_idx - 1],
            ))
        if ref_idx < 2:
            out["relative"].append(Case(
                "relative", f"right_of_{anchor}",
                f"Put the banana into the bowl on the right of the {anchor} bowl from the robot perspective.",
                ref_idx + 1, side="right", anchor=anchor, target_color=layout[ref_idx + 1],
            ))
    for slot_id, ord_text in enumerate(("1st", "2nd", "3rd")):
        out["ordinal"].append(Case(
            "ordinal", f"ord_{SLOTS[slot_id]}",
            f"Put the banana into the {ord_text} bowl from the left from the robot perspective.",
            slot_id, target_color=layout[slot_id],
        ))
    for target in COLORS:
        excluded = [c for c in COLORS if c != target]
        out["exclusion"].append(Case(
            "exclusion", f"not_{excluded[0]}_not_{excluded[1]}",
            f"Put the banana into the bowl that is not {excluded[0]} and not {excluded[1]}.",
            layout.index(target), target_color=target,
        ))
        out["direct"].append(Case(
            "direct", f"direct_{target}",
            f"Put the banana into the {target} bowl.",
            layout.index(target), target_color=target,
        ))
    return out


def load_policy(ckpt: Path, base_policy: str, device: torch.device):
    if (ckpt / "adapter_config.json").exists():
        base = SmolVLAPolicy.from_pretrained(base_policy)
        base.config.device = str(device)
        peft = PeftModel.from_pretrained(base, str(ckpt))
        policy = peft.merge_and_unload()
        processors_path = str(ckpt)
        print(f"Loaded merged adapter {ckpt} on base {base_policy}")
    else:
        policy = SmolVLAPolicy.from_pretrained(str(ckpt))
        processors_path = str(ckpt)
        print(f"Loaded full policy {ckpt}")
    policy.config.device = str(device)
    return policy.to(device).eval(), processors_path


def add_slot_hook(policy, ckpt: Path, storage: dict):
    slot_head = nn.Linear(policy.model.vlm_with_expert.expert_hidden_size, len(SLOTS)).to(policy.config.device)
    slot_head.load_state_dict(torch.load(ckpt / "target_slot_head.pt", map_location=policy.config.device))
    slot_head.eval()

    def hook(_module, inputs, _output):
        with torch.no_grad():
            storage["slot_logits"] = slot_head(inputs[0].float().mean(dim=1))

    return policy.model.action_out_proj.register_forward_hook(hook), slot_head


def frame_to_obs(ds: LeRobotDataset, global_idx: int) -> dict:
    sample = ds[global_idx]
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


def run_prompt(policy, preprocessor, postprocessor, obs, prompt: str, storage: dict, device: torch.device) -> str:
    policy.reset()
    preprocessor.reset()
    postprocessor.reset()
    storage.clear()
    _ = predict_action(
        observation=obs,
        policy=policy,
        device=device,
        preprocessor=preprocessor,
        postprocessor=postprocessor,
        use_amp=False,
        task=prompt,
    )
    if "slot_logits" not in storage:
        return "?"
    logits = storage["slot_logits"].detach().cpu().numpy()
    if logits.ndim > 1:
        logits = logits[0]
    return ID_TO_SLOT[int(np.argmax(logits))]


def inc(bucket: Counter, ok: bool) -> None:
    bucket["total"] += 1
    bucket["ok"] += int(ok)


def acc(bucket: Counter) -> float:
    return bucket["ok"] / max(1, bucket["total"])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--base-policy", default=DEFAULT_BASE)
    ap.add_argument("--dataset", default=DEFAULT_DATASET)
    ap.add_argument("--dataset-root", default=DEFAULT_DATASET_ROOT)
    ap.add_argument("--split-path", default="/home/shadeform/eval3_repair_work/eval2_clean_fixed_frame_split.json")
    ap.add_argument("--n-gate-frames", type=int, default=5)
    ap.add_argument("--threshold", type=float, default=0.95)
    ap.add_argument("--print-cases", action="store_true")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = Path(args.ckpt)
    with Path(args.split_path).open() as f:
        split = json.load(f)
    ds = LeRobotDataset(args.dataset, root=args.dataset_root, video_backend="pyav")
    policy, processors_path = load_policy(ckpt, args.base_policy, device)
    pre, post = make_pre_post_processors(
        policy.config,
        pretrained_path=processors_path,
        preprocessor_overrides={"device_processor": {"device": str(device)}},
    )
    storage: dict = {}
    hook, _ = add_slot_hook(policy, ckpt, storage)

    overall = Counter()
    by_family = defaultdict(Counter)
    by_side = defaultdict(Counter)
    by_anchor = defaultdict(Counter)
    by_layout = defaultdict(Counter)
    by_target_slot = defaultdict(Counter)

    print("layout\tframe\tfamily\tkey\texpected\tpred\tok")
    for code in LAYOUT_CODES:
        frames = split[code]["gate"][: args.n_gate_frames]
        cases_by_family = make_cases(code)
        for frame in frames:
            obs = frame_to_obs(ds, int(frame))
            for family in ("relative", "ordinal", "exclusion", "direct"):
                for case in cases_by_family[family]:
                    pred = run_prompt(policy, pre, post, obs, case.prompt, storage, device)
                    expected = SLOTS[case.slot_id]
                    ok = pred == expected
                    inc(overall, ok)
                    inc(by_family[family], ok)
                    inc(by_layout[code], ok)
                    inc(by_target_slot[expected], ok)
                    if family == "relative":
                        inc(by_side[case.side], ok)
                        inc(by_anchor[case.anchor], ok)
                    if args.print_cases or family == "relative":
                        print(f"{code}\t{frame}\t{family}\t{case.key}\t{expected}\t{pred}\t{ok}")

    print("\nSUMMARY")
    for family in ("relative", "ordinal", "exclusion", "direct"):
        c = by_family[family]
        print(f"{family}_slot_head_accuracy={acc(c)*100:.1f}% ({c['ok']}/{c['total']})")
    print("relative_by_side")
    for key in ("left", "right"):
        c = by_side[key]
        print(f"  {key}: {acc(c)*100:.1f}% ({c['ok']}/{c['total']})")
    print("relative_by_anchor")
    for key in ("blue", "green", "red"):
        c = by_anchor[key]
        print(f"  {key}: {acc(c)*100:.1f}% ({c['ok']}/{c['total']})")
    print("by_layout")
    for key in LAYOUT_CODES:
        c = by_layout[key]
        print(f"  {key}: {acc(c)*100:.1f}% ({c['ok']}/{c['total']})")
    print("by_target_slot")
    for key in SLOTS:
        c = by_target_slot[key]
        print(f"  {key}: {acc(c)*100:.1f}% ({c['ok']}/{c['total']})")
    rel_ok = acc(by_family["relative"]) >= args.threshold
    anchors_ok = all(acc(by_anchor[c]) >= 0.90 for c in ("blue", "green", "red"))
    sides_ok = all(acc(by_side[s]) >= 0.90 for s in ("left", "right"))
    layout_ok = all(acc(by_layout[c]) >= 0.90 for c in LAYOUT_CODES)
    sanity_ok = all(acc(by_family[f]) >= 0.90 for f in ("ordinal", "exclusion", "direct"))
    print(f"GATE_PASS={rel_ok and anchors_ok and sides_ok and layout_ok and sanity_ok}")
    hook.remove()


if __name__ == "__main__":
    main()
