#!/usr/bin/env python3
"""
Action-conditioning probe for Stage 2 target-routing checkpoints.

Tests whether the same image/state + different prompts produce different
shoulder_pan trajectories pointing toward the correct target bowl.

Also captures aux slot head predictions for each prompt.

Usage:
    python probe_stage2_action_conditioning.py --ckpts 004000 006000
    python probe_stage2_action_conditioning.py --ckpts 006000 --watch
"""
from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

os.environ["HF_LEROBOT_HOME"] = "/home/shadeform/.cache/lerobot"

# ── Config ────────────────────────────────────────────────────────────────────

CKPT_BASE = "/home/shadeform/outputs/eval2_strict10_targetrouting_aux_grouped_stage2/checkpoints"
INIT_PATH = "/home/shadeform/outputs/smolvla-eval2-strict10-eval1-init"
DATASET_REPO = "robot-learning-group47/eval2_fixed_bowl_fixed_banana_108"
DATASET_ROOT = "/home/shadeform/.cache/lerobot/robot-learning-group47/eval2_fixed_bowl_fixed_banana_108"
TARGET_INDEX = "/home/shadeform/data/eval2_fixed_target_branching/target_branching_index.csv"

SLOTS = ("left", "middle", "right")
SLOT_TO_ID = {s: i for i, s in enumerate(SLOTS)}
CHUNK = 50  # action chunk size

# All test prompts — model sees only these; no target_slot or target_color fed
PROMPTS = {
    # ── color-based ──────────────────────────────────────────
    "blue_color":   "Put the banana in the blue colored bowl.",
    "red_color":    "Put the banana in the red colored bowl.",
    "green_color":  "Put the banana in the green colored bowl.",
    # ── ordinal-based ─────────────────────────────────────────
    "2nd_from_L":   "Put the banana into the 2nd bowl from the left from the robot perspective.",
    "2nd_from_R":   "Put the banana into the 2nd bowl from the right from the robot perspective.",
    # ── relative/relational ───────────────────────────────────
    "right_of_red": "Put the banana into the bowl on the right of the red bowl from the robot perspective.",
    # ── negation ─────────────────────────────────────────────
    "not_grn_blue": "Put the banana into the bowl that is not green and not blue.",
}

COLOR_KEYS   = ["blue_color", "red_color", "green_color"]
ORDINAL_KEYS = ["2nd_from_L", "2nd_from_R"]
RELAT_KEYS   = ["right_of_red"]
NEGAT_KEYS   = ["not_grn_blue"]


# ── Dataset helpers ───────────────────────────────────────────────────────────

def pick_branch_frames(n: int = 3):
    """
    Pick n frames from the target_branching_index covering different layouts
    and different ground-truth target slots (left / middle / right).
    Returns list of dicts with metadata.
    """
    import pandas as pd
    df = pd.read_csv(TARGET_INDEX)
    # One frame per ground-truth slot, from distinct layouts
    selected = []
    used_slots = set()
    used_layouts = set()
    for slot in ["left", "right", "middle"]:  # priority order
        if len(selected) >= n:
            break
        sub = df[(df["target_slot"] == slot) & (~df["layout_id"].isin(used_layouts))]
        if len(sub) == 0:
            continue
        row = sub.iloc[0]
        selected.append({
            "global_index": int(row["global_index"]),
            "episode_index": int(row["episode_index"]),
            "layout_id": str(row["layout_id"]),
            "layout_left_to_right": str(row["layout_left_to_right"]),
            "true_slot": str(row["target_slot"]),
            "true_color": str(row["target_color"]),
            "prompt_orig": str(row["prompt"]),
        })
        used_slots.add(slot)
        used_layouts.add(str(row["layout_id"]))
    return selected


def frame_to_obs(ds, global_idx: int) -> dict:
    sample = ds[global_idx]
    img = sample["observation.images.front"]
    if isinstance(img, torch.Tensor):
        img = img.numpy()
    if img.ndim == 3 and img.shape[0] in (1, 3):
        img = img.transpose(1, 2, 0)   # CHW → HWC
    if img.dtype != np.uint8:
        img = (np.clip(img, 0, 1) * 255).astype(np.uint8)
    state = sample["observation.state"].numpy().astype(np.float32)
    return {
        "observation.images.camera1":        img,
        "observation.state":                 state,
        "observation.images.empty_camera_0": np.zeros((480, 640, 3), np.uint8),
        "observation.images.empty_camera_1": np.zeros((480, 640, 3), np.uint8),
    }


# ── Model loading ─────────────────────────────────────────────────────────────

def load_stage2_model(ckpt_step: str, device):
    """
    Load SmolVLAPolicy from init + apply saved LoRA adapter + load slot head.
    Returns (peft_policy, preprocessor, postprocessor, slot_head, hook_storage).
    """
    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
    from lerobot.policies.factory import make_pre_post_processors
    from peft import PeftModel

    ckpt_dir = Path(f"{CKPT_BASE}/{ckpt_step}/pretrained_model")

    print(f"  Loading base from init: {INIT_PATH}")
    base = SmolVLAPolicy.from_pretrained(INIT_PATH)

    print(f"  Applying LoRA adapter from {ckpt_dir}")
    policy = PeftModel.from_pretrained(base, str(ckpt_dir))
    policy = policy.to(device).eval()

    raw = policy.get_base_model()
    expert_hidden = raw.model.vlm_with_expert.expert_hidden_size
    print(f"  expert_hidden_size: {expert_hidden}")

    # Reconstruct slot head and load weights
    slot_head = nn.Linear(expert_hidden, 3).to(device)
    slot_state = torch.load(str(ckpt_dir / "target_slot_head.pt"), map_location=device)
    slot_head.load_state_dict(slot_state)
    slot_head.eval()
    print(f"  Slot head loaded from target_slot_head.pt")

    # Register hook on action_out_proj (same as training)
    storage: dict = {}
    action_out = raw.model.action_out_proj

    def hook(_module, inputs, _output):
        suffix_out = inputs[0].float()
        with torch.no_grad():
            storage["slot_logits"] = slot_head(suffix_out.mean(dim=1))

    hook_handle = action_out.register_forward_hook(hook)

    # Preprocessor/postprocessor from saved files in checkpoint
    preprocessor, postprocessor = make_pre_post_processors(
        policy.config,
        pretrained_path=str(ckpt_dir),
        preprocessor_overrides={"device_processor": {"device": str(device)}},
    )

    return policy, preprocessor, postprocessor, slot_head, storage, hook_handle


# ── Inference ─────────────────────────────────────────────────────────────────

def run_all_prompts(policy, preprocessor, postprocessor, obs, storage, device):
    """
    For each prompt, reset policy and collect the full CHUNK-step action sequence.
    Returns dict: prompt_key -> {actions: np.array(CHUNK, 6), slot_pred: int, slot_logits: np.array(3)}
    """
    from lerobot.utils.control_utils import predict_action

    results = {}
    for key, prompt_text in PROMPTS.items():
        policy.reset()
        storage.clear()
        steps = []
        slot_pred_captured = False
        slot_logits_np = None
        slot_pred = None

        for t in range(CHUNK):
            with torch.no_grad():
                a = predict_action(
                    observation=obs,
                    policy=policy,
                    device=device,
                    preprocessor=preprocessor,
                    postprocessor=postprocessor,
                    use_amp=False,
                    task=prompt_text,
                )
            a_np = a.cpu().numpy()
            if a_np.ndim > 1:
                a_np = a_np.squeeze()
            steps.append(a_np)

            # Capture slot prediction on first step (first forward pass)
            if not slot_pred_captured and "slot_logits" in storage:
                logits = storage["slot_logits"].cpu().numpy()
                if logits.ndim > 1:
                    logits = logits[0]
                slot_logits_np = logits
                slot_pred = int(np.argmax(logits))
                slot_pred_captured = True

        results[key] = {
            "actions":       np.stack(steps),       # (CHUNK, 6)
            "slot_pred":     slot_pred,              # int 0/1/2 or None
            "slot_logits":   slot_logits_np,         # np.array(3) or None
            "slot_pred_str": SLOTS[slot_pred] if slot_pred is not None else "?",
        }
    return results


# ── Metrics ───────────────────────────────────────────────────────────────────

def window_mean(actions, sl):
    return float(actions[sl, 0].mean())


def pan_range(results, keys, sl):
    vals = [results[k]["actions"][sl, 0].mean() for k in keys if k in results]
    return (max(vals) - min(vals)) if len(vals) >= 2 else 0.0


def interpret(r):
    if r < 1:    return "COLLAPSED <1°"
    elif r < 3:  return "weak 1–3°"
    elif r < 6:  return "weak 3–6°"
    elif r < 10: return "moderate 6–10°"
    elif r < 15: return "strong 10–15°"
    elif r < 25: return "STRONG 15–25°"
    else:        return "VERY STRONG >25°"


# ── Print ─────────────────────────────────────────────────────────────────────

W = 88

def print_frame_results(frame_info, results):
    layout = frame_info["layout_left_to_right"]
    print(f"\n  Layout : {frame_info['layout_id']}  →  {layout}")
    print(f"  True   : slot={frame_info['true_slot']:6s}  color={frame_info['true_color']}")
    print(f"  Orig   : {frame_info['prompt_orig']}")
    print(f"  Frame  : global_idx={frame_info['global_index']}  ep={frame_info['episode_index']}")
    print()

    slices = [
        ("pan[0]",        slice(0, 1)),
        ("m[0-9]",        slice(0, 10)),
        ("m[10-19]",      slice(10, 20)),
        ("m[20-29]",      slice(20, 30)),
        ("m[30-39]",      slice(30, 40)),
        ("m[40-49]",      slice(40, 50)),
    ]

    hdr = f"  {'prompt':15s}  {'exp_slot':8s}  {'aux_pred':8s}"
    for name, _ in slices:
        hdr += f"  {name:>8s}"
    print(hdr)
    print(f"  {'-'*(len(hdr)-2)}")

    # Expected slot per prompt — based on layout
    def expected_slot(key, layout_str):
        """Derive expected slot given layout (left_to_right list string) and prompt key."""
        import ast
        try:
            order = ast.literal_eval(layout_str)   # ["blue","green","red"]
        except Exception:
            return "?"
        if key == "blue_color":
            color = "blue"
        elif key == "red_color":
            color = "red"
        elif key == "green_color":
            color = "green"
        elif key == "2nd_from_L":
            idx = 1  # 0-based
            return SLOTS[idx] if idx < 3 else "?"
        elif key == "2nd_from_R":
            idx = len(order) - 2
            return SLOTS[idx] if idx < 3 else "?"
        elif key == "right_of_red":
            ri = order.index("red") if "red" in order else -1
            idx = ri + 1
            return SLOTS[idx] if 0 <= idx < 3 else "?"
        elif key == "not_grn_blue":
            # Bowl that is not green and not blue → must be red
            color = "red"
        else:
            return "?"
        idx = order.index(color) if color in order else -1
        return SLOTS[idx] if 0 <= idx < 3 else "?"

    for key in PROMPTS:
        r = results[key]
        exp = expected_slot(key, frame_info["layout_left_to_right"])
        pred_str = r["slot_pred_str"]
        correct = "✓" if pred_str == exp else "✗"
        row = f"  {key:15s}  {exp:8s}  {pred_str:6s} {correct}"
        for _, sl in slices:
            row += f"  {window_mean(r['actions'], sl):+8.3f}"
        print(row)

    # Pan ranges
    s_all  = slice(0, 50)
    s_m10  = slice(0, 10)
    s_fin  = slice(40, 50)

    print()
    range_rows = [
        ("color",   COLOR_KEYS),
        ("ordinal", ORDINAL_KEYS),
        ("relative",RELAT_KEYS),
        ("negation", NEGAT_KEYS),
    ]
    print(f"  {'prompt type':12s}  {'step[0]':>8}  {'mean[0-9]':>10}  {'mean[40-49]':>12}  interpretation")
    print(f"  {'-'*75}")
    for rtype, keys in range_rows:
        avail = [k for k in keys if k in results]
        if not avail:
            continue
        if len(avail) == 1:
            # can't compute a range with 1 key, skip
            continue
        r0   = pan_range(results, avail, slice(0, 1))
        rm10 = pan_range(results, avail, s_m10)
        rfin = pan_range(results, avail, s_fin)
        print(f"  {rtype:12s}  {r0:>8.3f}°  {rm10:>10.3f}°  {rfin:>12.3f}°  {interpret(rm10)}")

    return {
        "color_rm10":   pan_range(results, COLOR_KEYS,   s_m10),
        "color_rfin":   pan_range(results, COLOR_KEYS,   s_fin),
        "ordinal_rm10": pan_range(results, ORDINAL_KEYS, s_m10),
        "ordinal_rfin": pan_range(results, ORDINAL_KEYS, s_fin),
    }


def print_ckpt_summary(ckpt_step, frame_infos, all_results, all_metrics):
    print(f"\n{'='*W}")
    print(f"  CHECKPOINT {ckpt_step}  —  SUMMARY  (avg over {len(frame_infos)} frames)")
    print(f"{'='*W}")
    for key in ["color_rm10", "ordinal_rm10", "color_rfin", "ordinal_rfin"]:
        avg = float(np.mean([m[key] for m in all_metrics]))
        label = key.replace("_", " ")
        print(f"    {label:25s}: {avg:.3f}°  →  {interpret(avg)}")

    # Per-frame slot acc
    slot_hits = []
    for fi, results in zip(frame_infos, all_results):
        import ast
        try:
            order = ast.literal_eval(fi["layout_left_to_right"])
        except Exception:
            order = []
        for key in COLOR_KEYS:
            color = key.replace("_color", "")
            exp_idx = order.index(color) if color in order else -1
            exp_slot = SLOTS[exp_idx] if 0 <= exp_idx < 3 else "?"
            pred = results[key]["slot_pred_str"]
            slot_hits.append(pred == exp_slot)
    if slot_hits:
        acc = sum(slot_hits) / len(slot_hits)
        print(f"    {'color-slot aux accuracy':25s}: {acc*100:.1f}%  ({sum(slot_hits)}/{len(slot_hits)} color prompts correct)")

    avg_color_rm10 = float(np.mean([m["color_rm10"] for m in all_metrics]))
    avg_ord_rm10   = float(np.mean([m["ordinal_rm10"] for m in all_metrics]))

    print()
    if avg_color_rm10 >= 15 and avg_ord_rm10 >= 10:
        verdict = "✅ STRONG — robot test candidate if direction is correct"
    elif avg_color_rm10 >= 8 or avg_ord_rm10 >= 8:
        verdict = "⚠️  MODERATE — meaningful but direction check needed"
    elif avg_color_rm10 >= 3:
        verdict = "⚠️  WEAK — slot head learning but action not yet coupled"
    else:
        verdict = "❌ COLLAPSED — action does not respond to prompt"
    print(f"  Verdict: {verdict}")
    return {"color_rm10": avg_color_rm10, "ordinal_rm10": avg_ord_rm10}


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpts", nargs="+", default=["004000", "006000"])
    ap.add_argument("--watch", action="store_true")
    ap.add_argument("--poll_interval", type=int, default=120)
    ap.add_argument("--n_frames", type=int, default=3)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load dataset once
    print("\nLoading dataset ...")
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    ds = LeRobotDataset(repo_id=DATASET_REPO, root=DATASET_ROOT, video_backend="pyav")
    print(f"  {len(ds)} frames, {ds.num_episodes} episodes")

    # Pick branching-phase frames
    print(f"\nPicking {args.n_frames} branch-precommit frames (different layouts, L/M/R slots) ...")
    frame_infos = pick_branch_frames(n=args.n_frames)
    for i, fi in enumerate(frame_infos):
        print(f"  Frame {i+1}: global_idx={fi['global_index']}  layout={fi['layout_id']}"
              f"  true_slot={fi['true_slot']}  true_color={fi['true_color']}")
        print(f"           orig: {fi['prompt_orig']}")

    # Pre-load observations
    frame_obs_list = [frame_to_obs(ds, fi["global_index"]) for fi in frame_infos]

    # Probe checkpoints
    summary_table = {}  # ckpt -> avg metrics

    for ckpt_step in args.ckpts:
        adapter_file = Path(f"{CKPT_BASE}/{ckpt_step}/pretrained_model/adapter_model.safetensors")

        if args.watch:
            waited = 0
            while not adapter_file.exists():
                print(f"\n  [{ckpt_step}] waiting (elapsed {waited//60}m) ...")
                time.sleep(args.poll_interval)
                waited += args.poll_interval
            print(f"  [{ckpt_step}] detected — waiting 30s for write to finish ...")
            time.sleep(30)
        else:
            if not adapter_file.exists():
                print(f"\n  [{ckpt_step}] not found at {adapter_file} — skipping")
                continue

        print(f"\n{'='*W}")
        print(f"  PROBING CHECKPOINT {ckpt_step}")
        print(f"{'='*W}")

        policy, pre, post, slot_head, storage, hook_handle = load_stage2_model(ckpt_step, device)

        all_results = []
        all_metrics = []
        for i, (fi, obs) in enumerate(zip(frame_infos, frame_obs_list)):
            print(f"\n  --- Frame {i+1}/{len(frame_infos)}: layout={fi['layout_id']}"
                  f"  true_slot={fi['true_slot']} ---")
            results = run_all_prompts(policy, pre, post, obs, storage, device)
            all_results.append(results)
            metrics = print_frame_results(fi, results)
            all_metrics.append(metrics)

        avg = print_ckpt_summary(ckpt_step, frame_infos, all_results, all_metrics)
        summary_table[ckpt_step] = avg

        hook_handle.remove()
        del policy, slot_head
        torch.cuda.empty_cache()

    # Cross-checkpoint trend
    if len(summary_table) >= 2:
        print(f"\n{'='*W}")
        print("  TREND ACROSS CHECKPOINTS")
        print(f"{'='*W}")
        print(f"  {'ckpt':8s}  {'color mean[0-9]':>16}  {'ordinal mean[0-9]':>18}  verdict")
        print(f"  {'-'*70}")
        for ckpt, m in summary_table.items():
            print(f"  {ckpt:8s}  {m['color_rm10']:>16.3f}°  {m['ordinal_rm10']:>18.3f}°")
        steps = list(summary_table.keys())
        d_c = summary_table[steps[-1]]["color_rm10"]   - summary_table[steps[0]]["color_rm10"]
        d_o = summary_table[steps[-1]]["ordinal_rm10"] - summary_table[steps[0]]["ordinal_rm10"]
        print(f"\n  Δ color   {steps[0]}→{steps[-1]}: {d_c:+.3f}°  {'↑' if d_c > 1 else ('↓' if d_c < -1 else '→')}")
        print(f"  Δ ordinal {steps[0]}→{steps[-1]}: {d_o:+.3f}°  {'↑' if d_o > 1 else ('↓' if d_o < -1 else '→')}")

    print("\nDone.")


if __name__ == "__main__":
    main()
