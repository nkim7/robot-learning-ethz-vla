#!/usr/bin/env python3
"""
Offline action-conditioning probe — multi-frame, watcher mode.

For each checkpoint:
  - Uses 3 placement-phase frames from DIFFERENT episodes/tasks.
  - Reports per-frame and averaged pan ranges at step[0], mean[0-9], mean[40-49].
  - In --watch mode: polls for new checkpoints and probes them when they appear.

Usage:
    python probe_ckpt.py --ckpts 006000 008000 --watch
    python probe_ckpt.py --ckpts 002000 004000          # single-shot
"""
from __future__ import annotations
import argparse, os, time
import numpy as np
import torch

os.environ["HF_LEROBOT_HOME"] = "/home/shadeform/.cache/lerobot"

CKPT_BASE    = "/home/shadeform/outputs/smolvla-eval2-lora-vqa-eval1expert/checkpoints"
DATASET_REPO = "robot-learning-group47/eval2_compositional_augmented"
DATASET_ROOT = "/home/shadeform/.cache/lerobot/robot-learning-group47/eval2_compositional_augmented"

PROMPTS = {
    "blue":   "Put the banana in the blue bowl.",
    "red":    "Put the banana in the red bowl.",
    "green":  "Put the banana in the green bowl.",
    "2nd_L":  "Put the banana into the second bowl from the left.",
    "2nd_R":  "Put the banana into the second bowl from the right.",
}
COLOR_KEYS   = ["blue", "red", "green"]
ORDINAL_KEYS = ["2nd_L", "2nd_R"]
CHUNK        = 50


# ── Find 3 diverse placement-phase frames ─────────────────────────────────────

def find_placement_frames(ds, n=3):
    """
    Return n frame indices from DIFFERENT episodes with different task labels,
    each in placement phase (gripper closed, arm elevated).

    Strategy:
      - Scan episodes in strides, collect candidate placement frames.
      - Prefer episodes whose task label covers different bowl targets
        (red, blue/green, ordinal) so we sample diverse ground-truth contexts.
    """
    seen_task_types = set()
    candidates = []   # (score, frame_idx, task_type)

    # Scan through dataset with stride to find placement candidates
    stride = max(1, len(ds) // 3000)   # ~3000 samples to check
    for i in range(0, min(len(ds), stride * 3000), stride):
        s     = ds[i]
        state = s["observation.state"].numpy()
        fr    = s["frame_index"].item()
        task  = s["task"]
        if isinstance(task, (list, tuple)):
            task = task[0]
        task = str(task).strip().lower()

        shoulder_lift = state[1]
        gripper       = state[5]

        # Must be in placement phase
        if gripper < 25 or shoulder_lift < -30:
            continue
        # Must be mid-episode (not first 10 frames)
        if fr < 10:
            continue

        # Classify task type
        if "red" in task and "not" not in task:
            ttype = "red"
        elif "blue" in task and "not" not in task:
            ttype = "blue"
        elif "green" in task and "not" not in task:
            ttype = "green"
        elif "left" in task or "right" in task or "1st" in task or "2nd" in task or "3rd" in task:
            ttype = "ordinal"
        else:
            ttype = "other"

        score = gripper + shoulder_lift
        candidates.append((score, i, ttype, task[:60]))

    if not candidates:
        # Fallback: use known good frame from episode 0
        return [{"frame_idx": 113, "task_type": "blue",
                 "task_label": "Put the banana in the blue colored bowl.",
                 "state": ds[113]["observation.state"].numpy()}]

    # Sort by score, then pick diverse task types
    candidates.sort(key=lambda x: -x[0])

    selected = []
    used_ttypes = set()
    # Priority order: pick one red/blue/green color episode, one ordinal, one other
    priority = ["red", "blue", "ordinal", "green", "other"]
    for ptype in priority:
        if len(selected) >= n:
            break
        for score, fidx, ttype, tlabel in candidates:
            if ttype == ptype and ttype not in used_ttypes:
                s     = ds[fidx]
                state = s["observation.state"].numpy()
                selected.append({
                    "frame_idx":  fidx,
                    "episode":    s["episode_index"].item(),
                    "frame_in_ep": s["frame_index"].item(),
                    "task_type":  ttype,
                    "task_label": tlabel,
                    "state":      state,
                })
                used_ttypes.add(ttype)
                break

    # If still not enough, fill with best remaining candidates
    for score, fidx, ttype, tlabel in candidates:
        if len(selected) >= n:
            break
        if not any(f["frame_idx"] == fidx for f in selected):
            s = ds[fidx]
            selected.append({
                "frame_idx":  fidx,
                "episode":    s["episode_index"].item(),
                "frame_in_ep": s["frame_index"].item(),
                "task_type":  ttype,
                "task_label": tlabel,
                "state":      s["observation.state"].numpy(),
            })

    return selected[:n]


def frame_to_obs(ds, frame_idx):
    """Convert dataset frame to the observation dict expected by predict_action."""
    sample = ds[frame_idx]
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


# ── Load checkpoint ───────────────────────────────────────────────────────────

def load_model(ckpt_step: str, device):
    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
    from lerobot.policies.factory import make_pre_post_processors
    from peft import PeftConfig, PeftModel

    ckpt_dir = f"{CKPT_BASE}/{ckpt_step}/pretrained_model"
    peft_cfg = PeftConfig.from_pretrained(ckpt_dir)
    policy   = SmolVLAPolicy.from_pretrained(peft_cfg.base_model_name_or_path)
    model    = PeftModel.from_pretrained(policy, ckpt_dir, config=peft_cfg)
    model    = model.to(device).eval()
    preprocessor, postprocessor = make_pre_post_processors(
        policy.config,
        pretrained_path=ckpt_dir,
        preprocessor_overrides={"device_processor": {"device": str(device)}},
    )
    return model, preprocessor, postprocessor


# ── Run probe: collect full 50-step chunk per prompt ─────────────────────────

def run_probe_one_frame(model, preprocessor, postprocessor, frame_obs, device):
    """Returns dict: prompt_key -> np.array (CHUNK, 6) in unnormalized degrees."""
    from lerobot.utils.control_utils import predict_action
    results = {}
    for key, prompt_text in PROMPTS.items():
        model.reset()
        steps = []
        for _ in range(CHUNK):
            with torch.no_grad():
                a = predict_action(
                    observation=frame_obs,
                    policy=model,
                    device=device,
                    preprocessor=preprocessor,
                    postprocessor=postprocessor,
                    use_amp=False,
                    task=prompt_text,
                )
            a = a.cpu().numpy()
            if a.ndim > 1:
                a = a.squeeze()
            steps.append(a)
        results[key] = np.stack(steps)   # (50, 6)
    return results


# ── Compute pan range metrics ─────────────────────────────────────────────────

def pan_ranges(results):
    """Return dict of pan-range metrics."""
    def rng(d, keys, sl):
        vals = [d[k][sl, 0].mean() for k in keys]
        return max(vals) - min(vals)

    s0  = slice(0, 1)
    m10 = slice(0, 10)
    fin = slice(40, 50)

    return {
        "color_r0":   rng(results, COLOR_KEYS, s0),
        "color_rm10": rng(results, COLOR_KEYS, m10),
        "color_rfin": rng(results, COLOR_KEYS, fin),
        "ord_r0":     abs(results["2nd_L"][s0,  0].mean() - results["2nd_R"][s0,  0].mean()),
        "ord_rm10":   abs(results["2nd_L"][m10, 0].mean() - results["2nd_R"][m10, 0].mean()),
        "ord_rfin":   abs(results["2nd_L"][fin, 0].mean() - results["2nd_R"][fin, 0].mean()),
    }


# ── Print results for one checkpoint ─────────────────────────────────────────

def print_checkpoint_results(ckpt_step, frame_infos, all_results):
    W = 80
    print(f"\n{'='*W}")
    print(f"  CHECKPOINT {ckpt_step}")
    print(f"{'='*W}")

    # Per-frame table
    for fi, (info, results) in enumerate(zip(frame_infos, all_results)):
        m = pan_ranges(results)
        print(f"\n  Frame {fi+1}: idx={info['frame_idx']}  ep={info['episode']}"
              f"  fr_in_ep={info['frame_in_ep']}  task_type={info['task_type']}")
        print(f"           label: {info['task_label'][:65]}")
        print(f"           state: pan={info['state'][0]:.1f}°  lift={info['state'][1]:.1f}°"
              f"  grip={info['state'][5]:.1f}°")
        print(f"\n  {'prompt':10s}  {'pan[0]':>8}  {'pan_m[0-9]':>11}  {'pan_m[40-49]':>13}"
              f"  {'lift[0]':>8}  {'grip[0]':>8}")
        print(f"  {'-'*72}")
        for key in PROMPTS:
            a = results[key]
            print(f"  {key:10s}  {a[0,0]:+8.3f}  {a[:10,0].mean():+11.3f}  {a[40:,0].mean():+13.3f}"
                  f"  {a[0,1]:+8.3f}  {a[0,5]:+8.3f}")
        print(f"\n  Ranges this frame:")
        print(f"    color  step[0]={m['color_r0']:.3f}°  mean[0-9]={m['color_rm10']:.3f}°  mean[40-49]={m['color_rfin']:.3f}°")
        print(f"    ordinal step[0]={m['ord_r0']:.3f}°   mean[0-9]={m['ord_rm10']:.3f}°   mean[40-49]={m['ord_rfin']:.3f}°")

    # Averaged across frames
    avg_metrics = {}
    for k in ["color_r0","color_rm10","color_rfin","ord_r0","ord_rm10","ord_rfin"]:
        avg_metrics[k] = np.mean([pan_ranges(r)[k] for r in all_results])

    def interp(r):
        if r < 1:    return "COLLAPSED <1°"
        elif r < 3:  return "weak 1-3°"
        elif r < 5:  return "weak 3-5°"
        elif r < 10: return "moderate 5-10°"
        elif r < 20: return "strong 10-20°"
        else:        return "STRONG >20°"

    print(f"\n  {'─'*W}")
    print(f"  AVERAGED ACROSS {len(frame_infos)} FRAMES")
    print(f"  {'─'*W}")
    print(f"  {'metric':25s}  {'step[0]':>8}  {'mean[0-9]':>10}  {'mean[40-49]':>12}")
    print(f"  {'─'*60}")
    print(f"  {'color pan range':25s}  {avg_metrics['color_r0']:>8.3f}°  {avg_metrics['color_rm10']:>10.3f}°  {avg_metrics['color_rfin']:>12.3f}°")
    print(f"  {'ordinal pan range':25s}  {avg_metrics['ord_r0']:>8.3f}°  {avg_metrics['ord_rm10']:>10.3f}°  {avg_metrics['ord_rfin']:>12.3f}°")
    print(f"\n  KEY METRIC — mean[0-9] pan range:")
    print(f"    color:   {avg_metrics['color_rm10']:.3f}°  →  {interp(avg_metrics['color_rm10'])}")
    print(f"    ordinal: {avg_metrics['ord_rm10']:.3f}°  →  {interp(avg_metrics['ord_rm10'])}")

    return avg_metrics


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpts", nargs="+", default=["006000", "008000"])
    ap.add_argument("--watch", action="store_true",
                    help="Poll until each checkpoint appears, then probe it")
    ap.add_argument("--poll_interval", type=int, default=120,
                    help="Seconds between polls in watch mode (default: 120)")
    ap.add_argument("--n_frames", type=int, default=3,
                    help="Number of diverse placement frames to use (default: 3)")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ── Load dataset and find placement frames once ───────────────────────────
    print("\nLoading dataset ...")
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    ds = LeRobotDataset(repo_id=DATASET_REPO, root=DATASET_ROOT, video_backend="pyav")
    print(f"  {len(ds)} frames, {ds.num_episodes} episodes")

    print(f"\nFinding {args.n_frames} diverse placement-phase frames ...")
    frame_infos = find_placement_frames(ds, n=args.n_frames)
    print(f"  Selected {len(frame_infos)} frames:")
    for i, fi in enumerate(frame_infos):
        print(f"    Frame {i+1}: idx={fi['frame_idx']}  ep={fi['episode']}"
              f"  task_type={fi['task_type']}  grip={fi['state'][5]:.1f}°")

    # Pre-load observations (images + state) from dataset
    frame_obs_list = [frame_to_obs(ds, fi["frame_idx"]) for fi in frame_infos]

    # ── Probe each checkpoint ─────────────────────────────────────────────────
    summary = {}   # ckpt_step -> avg_metrics dict

    for ckpt_step in args.ckpts:
        ckpt_dir = f"{CKPT_BASE}/{ckpt_step}/pretrained_model"
        adapter_file = f"{ckpt_dir}/adapter_model.safetensors"

        if args.watch:
            waited = 0
            while not os.path.exists(adapter_file):
                print(f"\n  [{ckpt_step}] not ready yet — waiting "
                      f"({waited//60}m elapsed, checking every {args.poll_interval}s) ...")
                time.sleep(args.poll_interval)
                waited += args.poll_interval
            # Extra 30s buffer for file to finish writing
            print(f"  [{ckpt_step}] checkpoint detected — waiting 30s for write to complete ...")
            time.sleep(30)
        else:
            if not os.path.exists(adapter_file):
                print(f"\n  [{ckpt_step}] not found — skipping")
                continue

        print(f"\nLoading checkpoint {ckpt_step} ...")
        model, pre, post = load_model(ckpt_step, device)

        all_results = []
        for i, (fi, frame_obs) in enumerate(zip(frame_infos, frame_obs_list)):
            print(f"  Probing frame {i+1}/{len(frame_infos)} "
                  f"(ep={fi['episode']} task={fi['task_type']}) ...")
            r = run_probe_one_frame(model, pre, post, frame_obs, device)
            all_results.append(r)

        avg = print_checkpoint_results(ckpt_step, frame_infos, all_results)
        summary[ckpt_step] = avg

        del model
        torch.cuda.empty_cache()

    # ── Cross-checkpoint trend ────────────────────────────────────────────────
    if len(summary) >= 2:
        W = 80
        print(f"\n{'='*W}")
        print("  TREND ACROSS CHECKPOINTS  (avg over frames, KEY = mean[0-9])")
        print(f"{'='*W}")
        print(f"  {'ckpt':8s}  {'color[0]':>9}  {'color_m10':>10}  {'color_fin':>10}"
              f"  {'ord[0]':>8}  {'ord_m10':>8}  {'ord_fin':>8}")
        print(f"  {'-'*70}")
        for s, m in summary.items():
            print(f"  {s:8s}  {m['color_r0']:+9.3f}  {m['color_rm10']:+10.3f}  {m['color_rfin']:+10.3f}"
                  f"  {m['ord_r0']:+8.3f}  {m['ord_rm10']:+8.3f}  {m['ord_rfin']:+8.3f}")

        steps = list(summary.keys())
        d_color = summary[steps[-1]]["color_rm10"] - summary[steps[0]]["color_rm10"]
        d_ord   = summary[steps[-1]]["ord_rm10"]   - summary[steps[0]]["ord_rm10"]
        print(f"\n  mean[0-9] change  {steps[0]} → {steps[-1]}:")
        print(f"    color   : {d_color:+.3f}°  {'↑ INCREASING' if d_color>1 else ('↓ DECREASING' if d_color<-1 else '→ flat')}")
        print(f"    ordinal : {d_ord:+.3f}°  {'↑ INCREASING' if d_ord>1 else ('↓ DECREASING' if d_ord<-1 else '→ flat')}")
        print(f"\n  Thresholds for mean[0-9]:")
        print(f"    <1°  still collapsed  |  3-5° weak  |  5-10° meaningful  |  15°+ separating bowls")

    print("\nDone.")


if __name__ == "__main__":
    main()
