#!/usr/bin/env python3
"""Offline gate for Eval2 relational VLM repair checkpoints.

Gate criteria:
  - relative slot-head accuracy > 85%
  - left-of and right-of accuracies reported separately
  - anchor colors red/green/blue reported separately
  - all six layouts tested
  - action-nearest ordinal anchor reported as secondary evidence
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from pathlib import Path

from probe_eval2_relational_diagnosis import (
    DATASET_REPO,
    ID_TO_SLOT,
    SLOTS,
    CHUNK,
    frame_to_obs,
    make_cases,
    nearest_anchor_slot,
)
import torch
import numpy as np
from peft import PeftModel
from torch import nn
from eval2_layout_utils import normalize_layout
from lerobot.common.control_utils import predict_action
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies.factory import make_pre_post_processors
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy


LAYOUT_CODES = ("bgr", "brg", "gbr", "grb", "rbg", "rgb")
DEFAULT_BASE_POLICY = "/home/shadeform/outputs/eval2_stage2_compositional_015000_balanced_r2/checkpoints/020000/pretrained_model"


def pick_layout_frames(ds: LeRobotDataset) -> list[tuple[str, int]]:
    from probe_eval2_relational_diagnosis import augmented_episode_info

    found: dict[str, int] = {}
    for global_idx, ep_value in enumerate(ds.hf_dataset["episode_index"]):
        layout, _lid, _color, _slot, _orig = augmented_episode_info(int(ep_value))
        code = "".join(c[0] for c in layout)
        if code not in found:
            found[code] = global_idx
        if len(found) == len(LAYOUT_CODES):
            break
    missing = [code for code in LAYOUT_CODES if code not in found]
    if missing:
        raise RuntimeError(f"Missing layouts in probe dataset: {missing}")
    return [(code, found[code]) for code in LAYOUT_CODES]


def load_policy_for_gate(ckpt: Path, base_policy: str, device: torch.device):
    if (ckpt / "adapter_config.json").exists():
        base = SmolVLAPolicy.from_pretrained(base_policy)
        base.config.device = str(device)
        peft_policy = PeftModel.from_pretrained(base, str(ckpt))
        inference_policy = peft_policy.get_base_model()
        keepalive = peft_policy
        pretrained_for_processors = str(ckpt)
        print(f"Loaded repair adapter: {ckpt}")
        print(f"Adapter base policy: {base_policy}")
    else:
        inference_policy = SmolVLAPolicy.from_pretrained(str(ckpt))
        inference_policy.config.device = str(device)
        keepalive = inference_policy
        pretrained_for_processors = str(ckpt)
        print(f"Loaded full policy checkpoint: {ckpt}")
    inference_policy.config.device = str(device)
    inference_policy = inference_policy.to(device).eval()
    return inference_policy, keepalive, pretrained_for_processors


def add_slot_hook(policy, ckpt_dir: Path, storage: dict):
    slot_head = nn.Linear(policy.model.vlm_with_expert.expert_hidden_size, len(SLOTS)).to(policy.config.device)
    slot_head.load_state_dict(torch.load(ckpt_dir / "target_slot_head.pt", map_location=policy.config.device))
    slot_head.eval()

    def hook(_module, inputs, _output):
        suffix_out = inputs[0].float()
        with torch.no_grad():
            storage["slot_logits"] = slot_head(suffix_out.mean(dim=1))

    return policy.model.action_out_proj.register_forward_hook(hook), slot_head


def run_prompt(policy, preprocessor, postprocessor, obs, prompt: str, storage: dict, device) -> dict:
    policy.reset()
    preprocessor.reset()
    postprocessor.reset()
    storage.clear()
    actions = []
    slot_pred = None
    logits_np = None
    for i in range(CHUNK):
        action = predict_action(
            observation=obs,
            policy=policy,
            device=device,
            preprocessor=preprocessor,
            postprocessor=postprocessor,
            use_amp=False,
            task=prompt,
        )
        arr = action.detach().cpu().numpy()
        if arr.ndim > 1:
            arr = arr.squeeze(0)
        actions.append(arr)
        if i == 0 and "slot_logits" in storage:
            logits_np = storage["slot_logits"].detach().cpu().numpy()
            if logits_np.ndim > 1:
                logits_np = logits_np[0]
            slot_pred = int(np.argmax(logits_np))
    return {
        "actions": np.stack(actions),
        "slot_pred": slot_pred,
        "slot_pred_str": ID_TO_SLOT.get(slot_pred, "?"),
        "slot_logits": logits_np,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--base-policy", default=DEFAULT_BASE_POLICY)
    parser.add_argument("--dataset-root", default="/home/shadeform/lerobot_cache/robot-learning-group47/eval2_compositional_augmented")
    parser.add_argument("--threshold", type=float, default=0.85)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ds = LeRobotDataset(DATASET_REPO, root=args.dataset_root, video_backend="pyav")
    frames = pick_layout_frames(ds)

    ckpt = Path(args.ckpt)
    policy, _keepalive, pretrained_for_processors = load_policy_for_gate(ckpt, args.base_policy, device)
    pre, post = make_pre_post_processors(
        policy.config,
        pretrained_path=pretrained_for_processors,
        preprocessor_overrides={"device_processor": {"device": str(device)}},
    )
    storage: dict = {}
    hook, _slot_head = add_slot_hook(policy, ckpt, storage)

    overall = Counter()
    by_side = defaultdict(Counter)
    by_anchor = defaultdict(Counter)
    by_layout = defaultdict(Counter)
    action_overall = Counter()

    print("layout\tanchor\tside\tprompt\texpected_slot\tslot_pred\tslot_ok\taction_near\taction_ok")
    for code, global_idx in frames:
        layout = normalize_layout(code)
        obs = frame_to_obs(ds, global_idx)
        cases = make_cases(layout)
        anchors = {}
        for slot in SLOTS:
            res = run_prompt(
                policy,
                pre,
                post,
                obs,
                f"Put the banana into the {('1st','2nd','3rd')[SLOTS.index(slot)]} bowl from the left from the robot perspective.",
                storage,
                device,
            )
            anchors[slot] = float(res["actions"][:10, 0].mean())
        for case in cases:
            if case.family != "relative":
                continue
            side, _, anchor = case.key.partition("_of_")
            res = run_prompt(policy, pre, post, obs, case.prompt, storage, device)
            pred = res["slot_pred_str"]
            slot_ok = pred == case.expected_slot
            pan_m10 = float(res["actions"][:10, 0].mean())
            action_near = nearest_anchor_slot(pan_m10, anchors)
            action_ok = action_near == case.expected_slot
            for bucket in (overall, by_side[side], by_anchor[anchor], by_layout[code]):
                bucket["total"] += 1
                bucket["slot_ok"] += int(slot_ok)
            action_overall["total"] += 1
            action_overall["ok"] += int(action_ok)
            print(
                f"{code}\t{anchor}\t{side}\t{case.key}\t{case.expected_slot}\t"
                f"{pred}\t{slot_ok}\t{action_near}\t{action_ok}"
            )

    def acc(counter: Counter, key: str = "slot_ok") -> float:
        return counter[key] / max(1, counter["total"])

    print("\nSUMMARY")
    print(f"relative_slot_head_accuracy={acc(overall)*100:.1f}% ({overall['slot_ok']}/{overall['total']})")
    print(f"relative_action_nearest_accuracy={action_overall['ok']/max(1, action_overall['total'])*100:.1f}% ({action_overall['ok']}/{action_overall['total']})")
    print("by_side")
    for key in sorted(by_side):
        c = by_side[key]
        print(f"  {key}: {acc(c)*100:.1f}% ({c['slot_ok']}/{c['total']})")
    print("by_anchor")
    for key in sorted(by_anchor):
        c = by_anchor[key]
        print(f"  {key}: {acc(c)*100:.1f}% ({c['slot_ok']}/{c['total']})")
    print("by_layout")
    for key in LAYOUT_CODES:
        c = by_layout[key]
        print(f"  {key}: {acc(c)*100:.1f}% ({c['slot_ok']}/{c['total']})")
    passed = acc(overall) > args.threshold and all(acc(c) > args.threshold for c in by_side.values())
    print(f"GATE_PASS={passed}")
    hook.remove()


if __name__ == "__main__":
    main()
