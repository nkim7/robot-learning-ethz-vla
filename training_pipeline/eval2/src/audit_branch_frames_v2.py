#!/usr/bin/env python3
"""
Branch frame audit v2 — strict full-state ambiguity check.

Criteria:
  - ALL 6 joints must be similar across left/middle/right demos
  - State-only target classification check (can a classifier predict slot from state?)
  - Visual frame grids at the best candidate per layout

For each layout group, this script:
  1. Sweeps ALL local frames 0..ep_len
  2. Computes pairwise full-6D state distance between L/M/R at every frame
  3. Finds the frame window where max(dist_LM, dist_LR, dist_MR) is minimized
  4. Reports per-joint diffs at the best 3 candidate frames
  5. Runs a LOO state-only classifier: at each candidate frame, can state
     alone predict left/middle/right? (using all episodes in the layout)
  6. Saves frame images at the best candidate
"""
from __future__ import annotations

import os
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd
import torch

os.environ["HF_LEROBOT_HOME"] = "/home/shadeform/.cache/lerobot"

DATASET_REPO = "robot-learning-group47/eval2_fixed_bowl_fixed_banana_108"
DATASET_ROOT = "/home/shadeform/.cache/lerobot/robot-learning-group47/eval2_fixed_bowl_fixed_banana_108"
OLD_INDEX    = "/home/shadeform/data/eval2_fixed_target_branching/target_branching_index.csv"
OUT_DIR      = Path("/home/shadeform/branch_frame_audit_v2")
IMG_DIR      = OUT_DIR / "frames"

JOINT_NAMES = ["shoulder_pan", "shoulder_lift", "elbow", "wrist_flex", "wrist_roll", "gripper"]
SLOTS = ["left", "middle", "right"]

# Thresholds for flagging
GRIP_CLOSED_THRESH   =  20.0   # gripper < this = closed
LIFT_MIN             = -20.0   # shoulder_lift > this = elevated (relaxed)
PAN_THRESH           =   5.0   # per-joint pan diff flag
JOINT_THRESH         =  10.0   # per-joint diff flag for lift/elbow/wrist
FULL_STATE_GOOD      =   8.0   # max pairwise 6D dist = "good"
FULL_STATE_MARGINAL  =  15.0   # max pairwise 6D dist = "marginal"


# ── Dataset helpers ───────────────────────────────────────────────────────────

def build_ep_index(ds):
    ep_col = ds.hf_dataset["episode_index"]
    ep_to_frames: dict[int, list[int]] = defaultdict(list)
    for i, ep in enumerate(ep_col):
        ep_to_frames[int(ep)].append(i)
    ep_to_frames = {ep: sorted(v) for ep, v in ep_to_frames.items()}
    states_raw = ds.hf_dataset["observation.state"]
    all_states = np.array(states_raw, dtype=np.float64)   # (N, 6)
    return ep_to_frames, all_states


def save_frame_image(ds, ep_to_frames, ep, local_fr, path: Path):
    frames = ep_to_frames.get(ep, [])
    if local_fr >= len(frames):
        return False
    sample = ds[frames[local_fr]]
    img = sample["observation.images.front"]
    if isinstance(img, torch.Tensor):
        img = img.numpy()
    if img.ndim == 3 and img.shape[0] in (1, 3):
        img = img.transpose(1, 2, 0)
    if img.dtype != np.uint8:
        img = (np.clip(img, 0, 1) * 255).astype(np.uint8)
    from PIL import Image
    Image.fromarray(img).save(str(path))
    return True


# ── State distance ────────────────────────────────────────────────────────────

def full_dist(s1, s2):
    """Euclidean distance across all 6 joints."""
    return float(np.linalg.norm(s1 - s2))


def partial_dist(s1, s2, joints=(0,1,2,3)):
    """Euclidean on selected joints (default: pan+lift+elbow+wrist_flex)."""
    return float(np.linalg.norm(s1[list(joints)] - s2[list(joints)]))


# ── State-only target classification ─────────────────────────────────────────

def state_only_classif(all_states, ep_to_frames, layout_group, lf,
                       n_neighbors=1, n_context_frames=5):
    """
    At local_frame=lf, collect 6D state vectors from all episodes in each slot.
    Use leave-one-out k-NN (k=1) to check if state alone predicts slot.

    For robustness, use a small context window: states at lf-n..lf+n.
    Returns:
        accuracy: float (0.0-1.0)
        detail: dict slot -> list of (true_slot, predicted_slot)
    """
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.preprocessing import StandardScaler

    X, y = [], []
    # Collect from all episodes for this layout (multiple episodes per slot)
    for slot_idx, slot in enumerate(SLOTS):
        eps_for_slot = layout_group[slot]["all_episodes"]
        for ep in eps_for_slot:
            frames = ep_to_frames.get(ep, [])
            # context window
            for delta in range(-n_context_frames, n_context_frames + 1):
                flf = lf + delta
                if 0 <= flf < len(frames):
                    X.append(all_states[frames[flf]])
                    y.append(slot_idx)

    X = np.array(X, dtype=np.float64)
    y = np.array(y)

    if len(X) < 6:
        return None, {}

    # LOO k-NN
    scaler = StandardScaler()
    X_sc = scaler.fit_transform(X)

    n = len(X)
    correct = 0
    for i in range(n):
        X_train = np.concatenate([X_sc[:i], X_sc[i+1:]])
        y_train = np.concatenate([y[:i], y[i+1:]])
        X_test  = X_sc[i:i+1]
        clf = KNeighborsClassifier(n_neighbors=min(n_neighbors, len(X_train)))
        clf.fit(X_train, y_train)
        pred = clf.predict(X_test)[0]
        if pred == y[i]:
            correct += 1

    acc = correct / n
    return acc, {}


# ── Per-layout sweep ──────────────────────────────────────────────────────────

def sweep_layout(all_states, ep_to_frames, layout_group):
    """
    Sweep all local frames. Return DataFrame with per-frame state metrics.
    """
    eps = {slot: layout_group[slot]["episode_index"] for slot in SLOTS}
    ep_len = min(len(ep_to_frames.get(eps[s], [])) for s in SLOTS)

    rows = []
    for lf in range(ep_len):
        s = {slot: all_states[ep_to_frames[eps[slot]][lf]] for slot in SLOTS}

        fd = {
            "LM": full_dist(s["left"], s["middle"]),
            "LR": full_dist(s["left"], s["right"]),
            "MR": full_dist(s["middle"], s["right"]),
        }
        pd6 = {  # partial: pan+lift+elbow+wrist_flex
            "LM": partial_dist(s["left"], s["middle"]),
            "LR": partial_dist(s["left"], s["right"]),
            "MR": partial_dist(s["middle"], s["right"]),
        }
        max_full = max(fd.values())
        max_part = max(pd6.values())
        max_pan  = max(abs(s[a][0]-s[b][0]) for a,b in [("left","middle"),("left","right"),("middle","right")])
        max_lift = max(abs(s[a][1]-s[b][1]) for a,b in [("left","middle"),("left","right"),("middle","right")])
        max_elbow= max(abs(s[a][2]-s[b][2]) for a,b in [("left","middle"),("left","right"),("middle","right")])
        max_wrist= max(abs(s[a][3]-s[b][3]) for a,b in [("left","middle"),("left","right"),("middle","right")])
        max_grip = max(abs(s[a][5]-s[b][5]) for a,b in [("left","middle"),("left","right"),("middle","right")])
        grip_max = max(s[sl][5] for sl in SLOTS)
        lift_min = min(s[sl][1] for sl in SLOTS)

        rows.append({
            "lf": lf,
            "max_full_dist": max_full,
            "max_part_dist": max_part,
            "max_pan":  max_pan,
            "max_lift": max_lift,
            "max_elbow":max_elbow,
            "max_wrist":max_wrist,
            "max_grip": max_grip,
            "grip_max_val": grip_max,
            "lift_min_val": lift_min,
            "dist_LM_full": fd["LM"],
            "dist_LR_full": fd["LR"],
            "dist_MR_full": fd["MR"],
            "pan_L":  s["left"][0],  "pan_M":  s["middle"][0],  "pan_R":  s["right"][0],
            "lift_L": s["left"][1],  "lift_M": s["middle"][1],  "lift_R": s["right"][1],
            "elbow_L":s["left"][2],  "elbow_M":s["middle"][2],  "elbow_R":s["right"][2],
            "wrist_L":s["left"][3],  "wrist_M":s["middle"][3],  "wrist_R":s["right"][3],
            "roll_L": s["left"][4],  "roll_M": s["middle"][4],  "roll_R": s["right"][4],
            "grip_L": s["left"][5],  "grip_M": s["middle"][5],  "grip_R": s["right"][5],
        })
    return pd.DataFrame(rows)


# ── Print helpers ─────────────────────────────────────────────────────────────

W = 110

def print_sweep_summary(layout_id, df):
    """Print every 5th frame plus any row that is a local minimum in max_full_dist."""
    print(f"\n{'─'*W}")
    print(f"  Layout: {layout_id}   sweep of full-state distance (all 6 joints)")
    print(f"{'─'*W}")
    hdr = (f"  {'lf':>5}  {'max_full':>9}  {'max_pan':>8}  {'max_lift':>9}  "
           f"{'max_elbow':>10}  {'max_wrist':>10}  {'max_grip':>9}  "
           f"{'grip_max':>9}  {'lift_min':>9}")
    print(hdr)
    print(f"  {'─'*W}")

    # Find local minima of max_full_dist
    vals = df["max_full_dist"].values
    local_min_lfs = set()
    for i in range(1, len(vals)-1):
        if vals[i] < vals[i-1] and vals[i] < vals[i+1]:
            local_min_lfs.add(df.iloc[i]["lf"])

    shown = set()
    for _, r in df.iterrows():
        lf = int(r["lf"])
        is_min = lf in local_min_lfs
        show = (lf % 5 == 0) or is_min
        if not show:
            continue
        mark = " ★" if is_min else "  "
        gflag = "  grip?" if r["grip_max_val"] < GRIP_CLOSED_THRESH else ""
        line = (f"{mark}{int(r['lf']):>5}  {r['max_full_dist']:>9.2f}  {r['max_pan']:>8.2f}  "
                f"{r['max_lift']:>9.2f}  {r['max_elbow']:>10.2f}  {r['max_wrist']:>10.2f}  "
                f"{r['max_grip']:>9.2f}  {r['grip_max_val']:>9.2f}  {r['lift_min_val']:>9.2f}{gflag}")
        print(line)


def print_candidate_detail(layout_id, df, lf, classif_acc):
    r = df[df["lf"] == lf].iloc[0]
    print(f"\n  ★ Candidate local_frame={lf}  (full-state max_dist={r['max_full_dist']:.2f}°)")
    print(f"  {'joint':>14}   {'left':>10}  {'middle':>10}  {'right':>10}  {'max_diff':>10}  flag")
    print(f"  {'─'*66}")
    joint_data = [
        ("shoulder_pan",  r["pan_L"],   r["pan_M"],   r["pan_R"]),
        ("shoulder_lift", r["lift_L"],  r["lift_M"],  r["lift_R"]),
        ("elbow",         r["elbow_L"], r["elbow_M"], r["elbow_R"]),
        ("wrist_flex",    r["wrist_L"], r["wrist_M"], r["wrist_R"]),
        ("wrist_roll",    r["roll_L"],  r["roll_M"],  r["roll_R"]),
        ("gripper",       r["grip_L"],  r["grip_M"],  r["grip_R"]),
    ]
    for name, vl, vm, vr in joint_data:
        diff = max(vl,vm,vr) - min(vl,vm,vr)
        thresh = PAN_THRESH if name=="shoulder_pan" else JOINT_THRESH
        flag = "⚠" if diff > thresh else "✓"
        print(f"  {name:>14}   {vl:>10.3f}  {vm:>10.3f}  {vr:>10.3f}  {diff:>10.3f}  {flag}")

    print(f"\n  Pairwise full-6D distances:")
    for pair, key in [("L–M","dist_LM_full"),("L–R","dist_LR_full"),("M–R","dist_MR_full")]:
        d = r[key]
        tag = ("✓ good" if d < FULL_STATE_GOOD else
               ("~ marginal" if d < FULL_STATE_MARGINAL else "⚠ diverged"))
        print(f"    {pair}: {d:.2f}°  {tag}")

    if classif_acc is not None:
        pct = classif_acc * 100
        chance = 33.3
        flag = "⚠ state leaks target" if pct > 50 else "✓ near-chance"
        print(f"\n  State-only LOO k-NN classification accuracy: {pct:.1f}%  "
              f"(chance=33.3%)  {flag}")
    print()


def interp_dist(d):
    if d < 5:    return "✓ GOOD <5°"
    elif d < 10: return "~ OK 5–10°"
    elif d < 15: return "⚠ marginal 10–15°"
    else:        return "✗ DIVERGED >15°"


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    IMG_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading dataset...")
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    ds = LeRobotDataset(repo_id=DATASET_REPO, root=DATASET_ROOT, video_backend="pyav")
    print(f"  {len(ds)} frames, {ds.num_episodes} episodes")

    print("Building episode index from Arrow (no video load)...")
    ep_to_frames, all_states = build_ep_index(ds)
    print(f"  {len(ep_to_frames)} episodes, all_states shape={all_states.shape}")

    old_df = pd.read_csv(OLD_INDEX)
    # Build layout groups with ALL episodes per slot (for classifier)
    # Primary episode = first occurrence, all_episodes = all for that slot+layout
    layout_ids = sorted(old_df["layout_id"].unique())

    layout_groups = {}
    for lid in layout_ids:
        sub = old_df[old_df["layout_id"] == lid].drop_duplicates("episode_index")
        group = {}
        ok = True
        for slot in SLOTS:
            rows = sub[sub["target_slot"] == slot]
            if len(rows) == 0:
                ok = False; break
            group[slot] = {
                "episode_index":      int(rows.iloc[0]["episode_index"]),
                "orig_episode_index": int(rows.iloc[0]["orig_episode_index"]),
                "target_color":       str(rows.iloc[0]["target_color"]),
                "layout_left_to_right": str(rows.iloc[0]["layout_left_to_right"]),
                "all_episodes":       [int(r["episode_index"]) for _, r in rows.iterrows()],
            }
        if ok:
            layout_groups[lid] = group

    print(f"  {len(layout_groups)} layouts loaded\n")

    print("="*W)
    print("  BRANCH FRAME AUDIT v2 — full-state ambiguity sweep")
    print("="*W)
    print(f"\n  Strict criteria:")
    print(f"    ✓ full-6D pairwise distance < {FULL_STATE_GOOD}° = GOOD")
    print(f"    ~ full-6D pairwise distance < {FULL_STATE_MARGINAL}° = marginal")
    print(f"    ✗ full-6D pairwise distance > {FULL_STATE_MARGINAL}° = diverged")
    print(f"    State-only LOO k-NN accuracy <= 50% = ambiguous frame")

    all_candidates = []
    layout_best = {}   # layout_id -> best (lf, max_full_dist, classif_acc, df)

    for lid, group in layout_groups.items():
        df = sweep_layout(all_states, ep_to_frames, group)

        # Print sweep summary
        print_sweep_summary(lid, df)

        # Find global minimum of max_full_dist
        best_idx = df["max_full_dist"].idxmin()
        best_lf  = int(df.loc[best_idx, "lf"])
        best_d   = float(df.loc[best_idx, "max_full_dist"])

        # Also find best frame WITH grip < threshold (banana grasped)
        grip_ok_df = df[df["grip_max_val"] < GRIP_CLOSED_THRESH]
        best_grip_lf  = None
        best_grip_d   = None
        if len(grip_ok_df) > 0:
            gi = grip_ok_df["max_full_dist"].idxmin()
            best_grip_lf = int(grip_ok_df.loc[gi, "lf"])
            best_grip_d  = float(grip_ok_df.loc[gi, "max_full_dist"])

        print(f"\n  Global best (no grip constraint): lf={best_lf}  max_full_dist={best_d:.2f}°")
        if best_grip_lf is not None:
            print(f"  Best WITH grip<{GRIP_CLOSED_THRESH}°:          lf={best_grip_lf}  max_full_dist={best_grip_d:.2f}°")
        else:
            print(f"  !! No frame with grip<{GRIP_CLOSED_THRESH}° found for {lid}")

        # Show top-5 by max_full_dist (no constraint)
        top5 = df.nsmallest(5, "max_full_dist")
        print(f"\n  Top-5 frames by max full-state distance:")
        print(f"  {'lf':>6}  {'max_full':>9}  {'max_pan':>8}  {'max_lift':>9}  {'max_grip':>9}  {'grip_max':>9}  classification_check")
        print(f"  {'─'*75}")

        for _, r in top5.iterrows():
            rlf = int(r["lf"])
            # Run state classifier at this frame
            acc, _ = state_only_classif(all_states, ep_to_frames, group, rlf)
            acc_str = f"{acc*100:.0f}%" if acc is not None else "N/A"
            flag = "⚠leak" if (acc is not None and acc > 0.5) else "✓ok"
            gflag = "grip?" if r["grip_max_val"] < GRIP_CLOSED_THRESH else ""
            print(f"  {rlf:>6}  {r['max_full_dist']:>9.2f}  {r['max_pan']:>8.2f}  "
                  f"{r['max_lift']:>9.2f}  {r['max_grip']:>9.2f}  {r['grip_max_val']:>9.2f}  "
                  f"acc={acc_str} {flag}  {gflag}")

        # Decide which candidate to use for detail + images
        # Prefer: best_grip_lf if available and dist < 20, else best_lf
        cand_lf = best_grip_lf if best_grip_lf is not None else best_lf
        cand_d  = best_grip_d  if best_grip_lf is not None else best_d

        acc, _ = state_only_classif(all_states, ep_to_frames, group, cand_lf)
        print_candidate_detail(lid, df, cand_lf, acc)

        # Save images
        print(f"  Saving frames at lf={cand_lf} ...")
        for slot in SLOTS:
            ep    = group[slot]["episode_index"]
            color = group[slot]["target_color"]
            fname = f"{lid}_{slot}_{color}_ep{ep:03d}_lf{cand_lf:03d}.jpg"
            saved = save_frame_image(ds, ep_to_frames, ep, cand_lf, IMG_DIR / fname)
            if saved:
                print(f"    {fname}")

        # Also save early frame (lf=10) for comparison reference
        early_lf = 10
        for slot in SLOTS:
            ep    = group[slot]["episode_index"]
            color = group[slot]["target_color"]
            fname = f"{lid}_{slot}_{color}_ep{ep:03d}_lf{early_lf:03d}_early.jpg"
            save_frame_image(ds, ep_to_frames, ep, early_lf, IMG_DIR / fname)

        layout_best[lid] = {
            "best_lf":   cand_lf,
            "max_full_dist": cand_d,
            "classif_acc": acc,
            "grip_ok": best_grip_lf is not None,
        }

        # Record candidate rows
        r_cand = df[df["lf"] == cand_lf].iloc[0]
        for slot in SLOTS:
            ep = group[slot]["episode_index"]
            all_candidates.append({
                "layout_id": lid,
                "layout_left_to_right": group[slot]["layout_left_to_right"],
                "target_slot": slot,
                "target_color": group[slot]["target_color"],
                "episode_index": ep,
                "local_frame": cand_lf,
                "global_index": ep_to_frames[ep][cand_lf],
                "pan":   all_states[ep_to_frames[ep][cand_lf]][0],
                "lift":  all_states[ep_to_frames[ep][cand_lf]][1],
                "elbow": all_states[ep_to_frames[ep][cand_lf]][2],
                "wrist": all_states[ep_to_frames[ep][cand_lf]][3],
                "roll":  all_states[ep_to_frames[ep][cand_lf]][4],
                "grip":  all_states[ep_to_frames[ep][cand_lf]][5],
                "max_full_dist_in_group": cand_d,
                "state_classif_acc": acc,
            })

    # ── Grand summary ────────────────────────────────────────────────────────
    print(f"\n\n{'='*W}")
    print("  SUMMARY — best achievable full-state ambiguity per layout")
    print(f"{'='*W}")
    print(f"\n  {'layout_id':>20}  {'best_lf':>8}  {'max_full_dist':>14}  {'grip_ok':>8}  "
          f"{'classif_acc':>12}  verdict")
    print(f"  {'─'*80}")
    for lid, info in layout_best.items():
        acc = info["classif_acc"]
        acc_str = f"{acc*100:.0f}%" if acc is not None else "N/A"
        verdict = ("✓ USE" if (info["max_full_dist"] < FULL_STATE_GOOD
                               and (acc is None or acc <= 0.5))
                   else ("~ MARGINAL" if info["max_full_dist"] < FULL_STATE_MARGINAL
                         else "✗ SKIP"))
        print(f"  {lid:>20}  {info['best_lf']:>8}  {info['max_full_dist']:>14.2f}°  "
              f"{'yes' if info['grip_ok'] else 'NO':>8}  {acc_str:>12}  {verdict}")

    # Save CSV
    if all_candidates:
        cdf = pd.DataFrame(all_candidates)
        csv_path = OUT_DIR / "audit_branch_candidates_v2.csv"
        cdf.to_csv(csv_path, index=False)
        print(f"\n  Saved: {csv_path}")

    print(f"\n  Frame images: {IMG_DIR}/")
    print(f"\n  Interpretation guide:")
    print(f"    max_full_dist < {FULL_STATE_GOOD}°    + classif_acc ≤ 50%  → truly ambiguous, safe to train")
    print(f"    max_full_dist < {FULL_STATE_MARGINAL}°   + classif_acc ≤ 50%  → marginal, may leak via lift/elbow")
    print(f"    max_full_dist > {FULL_STATE_MARGINAL}°   OR classif_acc > 50% → state leaks target, reject")
    print("\nDone.")


if __name__ == "__main__":
    main()
