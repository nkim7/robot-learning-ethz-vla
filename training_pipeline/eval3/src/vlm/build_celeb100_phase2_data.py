#!/usr/bin/env python3
"""
build_celeb100_phase2_data.py
==============================
Build robot-view spatial grounding dataset for ALL 100 celebrities.

Each scene: 3 celebrities from the 100-celeb pool placed in left/center/right
slots on one of 5 real robot background templates (v6 compositor).

Training objective: given a robot-view scene with 3 celebrity cards,
the model must say which slot (left / center / right) contains a named celebrity.

Dataset size: ~20 000 scenes
  - 100 celebrities × 6 positions (as L, C, R in all 3-slot combos)
    Actually: for each background, sample random triplets of celebrities
    with each celebrity appearing in each slot approximately equally.
  - n_scenes_per_celeb_per_slot × 100 celebrities × 3 slots × 5 backgrounds
    We'll target ~14 samples per (celebrity, slot, background) for ~21 000 total.

Prompt styles (mixed):
  - P0 diagnostic: "There are three celebrity cards... Which slot is {name} in?"
  - P1 action+grounding: "Put the coke can on the image of {name}. Which slot..."
  - VLA natural: "Put the coke can on the image of {name}."  → answer: left/center/right

Output:
  ./data/celeb100_phase2/manifest.jsonl
  ./data/celeb100_phase2/scenes/         (rendered PNG files)

Usage:
  python3 build_celeb100_phase2_data.py [--n_per_slot 14] [--seed 42]
"""

import argparse, json, os, random
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from datasets import load_dataset

parser = argparse.ArgumentParser()
parser.add_argument("--n_per_slot",  type=int, default=14,
                    help="scenes per (celebrity, slot, background) combo")
parser.add_argument("--seed",        type=int, default=42)
parser.add_argument("--out_dir",     default="./data/celeb100_phase2")
parser.add_argument("--scene_dir",   default="./celeb_face_samples/in_order")
args = parser.parse_args()

random.seed(args.seed)
np.random.seed(args.seed)

OUT_DIR   = Path(args.out_dir)
SCENE_OUT = OUT_DIR / "scenes"
SCENE_OUT.mkdir(parents=True, exist_ok=True)
MANIFEST  = OUT_DIR / "manifest.jsonl"

SCENE_DIR = Path(args.scene_dir)

# ── Prompt templates ──────────────────────────────────────────────────────────
P0_TEMPLATES = [
    ("There are three celebrity cards in this robot scene arranged in left, center, "
     "and right slots. Which slot is {name} in? Answer with exactly one word: left, center, or right."),
    ("The robot sees three celebrity portrait cards. {name} is in one of three positions: "
     "left, center, or right. Which position is {name} in? Answer: left, center, or right."),
    ("In this robot scene there are three cards side by side. Where is {name}? "
     "Answer with exactly one word: left, center, or right."),
    ("Look at the three celebrity cards on the table. In which slot is {name}? "
     "Answer left, center, or right."),
]

P1_TEMPLATES = [
    ("Put the coke can on the image of {name}. The cards are arranged left, center, "
     "and right. Which slot contains {name}? Answer with exactly one word: left, center, or right."),
    ("Place the coke can on {name}'s picture. The portrait cards occupy the left, center, "
     "and right slots. Which slot contains {name}? Answer with exactly one word: left, center, or right."),
    ("Move the coke can to the card showing {name}. In this robot view, is {name} on the "
     "left, center, or right? Answer with exactly one word."),
    ("Put the coke can on {name}'s card. Which slot is that — left, center, or right? "
     "Answer with exactly one word."),
]

VLA_TEMPLATES = [
    "Put the coke can on the image of {name}.",
    "Place the coke can on {name}'s picture.",
    "Move the coke can to the card showing {name}.",
    "Put the coke can on {name}.",
    "Place the can on the card with {name}.",
]

# ── v6 Compositor ─────────────────────────────────────────────────────────────
CARD_QUADS = {
    "scene_L-Swift_C-LeCun_R-Obama_1.jpg": {
        "LEFT":   [(31,43),(93,82),(55,119),(-5,77)],
        "CENTER": [(98,16),(104,62),(153,60),(143,16)],
        "RIGHT":  [(154,79),(192,118),(249,84),(213,49)],
        "CAN":    [(148,41),(136,44),(133,53),(130,66),(122,84),(115,99),(116,115),(121,116),(132,117),(138,104),(146,94),(161,71),(165,62),(167,54),(160,46),(158,42)],
    },
    "scene_L-Swift_C-LeCun_R-Obama_2.jpg": {
        "LEFT":   [(29,44),(93,80),(55,116),(-5,77)],
        "CENTER": [(97,16),(103,60),(153,60),(144,18)],
        "RIGHT":  [(215,52),(156,80),(193,119),(248,85)],
        "CAN":    [(146,41),(137,43),(134,49),(132,59),(129,72),(123,84),(115,96),(113,106),(122,116),(136,116),(144,94),(166,62),(166,53),(156,42)],
    },
    "scene_L-LeCun_C-Swift_R-Obama.jpg": {
        "LEFT":   [(-5,74),(36,48),(84,85),(41,118)],
        "CENTER": [(91,13),(90,60),(136,59),(134,14)],
        "RIGHT":  [(202,47),(154,82),(202,123),(251,82)],
        "CAN":    [(147,40),(137,44),(133,56),(128,69),(123,81),(115,97),(115,114),(121,116),(130,116),(138,116),(147,95),(152,83),(158,74),(164,65),(168,56),(160,46),(157,41)],
    },
    "scene_L-LeCun_C-Obama_R-Swift_1.jpg": {
        "LEFT":   [(-4,77),(30,46),(85,83),(49,119)],
        "CENTER": [(108,14),(102,62),(161,66),(156,18)],
        "RIGHT":  [(171,78),(206,124),(264,92),(232,54)],
        "CAN":    [(144,41),(137,43),(131,55),(129,66),(119,86),(113,107),(119,114),(130,115),(136,115),(140,100),(152,84),(167,61),(167,51),(157,45),(155,40)],
    },
    "scene_L-LeCun_C-Obama_R-Swift_2.jpg": {
        "LEFT":   [(-5,76),(31,46),(85,81),(46,117)],
        "CENTER": [(106,14),(101,60),(160,64),(157,17)],
        "RIGHT":  [(172,84),(217,125),(268,89),(230,53)],
        "CAN":    [(141,41),(133,45),(130,57),(129,69),(121,83),(114,96),(113,112),(121,118),(132,118),(138,116),(137,105),(144,100),(152,85),(156,76),(165,64),(166,54),(160,47),(157,43),(150,41)],
    },
}
CORNER_ORDER = {
    "scene_L-Swift_C-LeCun_R-Obama_1.jpg": {"LEFT":[3,0,1,2],"CENTER":[0,3,2,1],"RIGHT":[3,2,1,0]},
    "scene_L-Swift_C-LeCun_R-Obama_2.jpg": {"LEFT":[3,0,1,2],"CENTER":[0,3,2,1],"RIGHT":[0,3,2,1]},
    "scene_L-LeCun_C-Swift_R-Obama.jpg":   {"LEFT":[0,1,2,3],"CENTER":[0,3,2,1],"RIGHT":[0,3,2,1]},
    "scene_L-LeCun_C-Obama_R-Swift_1.jpg": {"LEFT":[0,1,2,3],"CENTER":[0,3,2,1],"RIGHT":[3,2,1,0]},
    "scene_L-LeCun_C-Obama_R-Swift_2.jpg": {"LEFT":[0,1,2,3],"CENTER":[0,3,2,1],"RIGHT":[3,2,1,0]},
}
POS_KEY  = {"left": "LEFT", "center": "CENTER", "right": "RIGHT"}
BG_NAMES = list(CARD_QUADS.keys())

def _get_quad(pts, order): return np.array([pts[i] for i in order], np.float32)
def _expand(q, f=0.12):
    cx, cy = q.mean(0)
    return (q + f * (q - np.array([[cx, cy]] * 4))).astype(int)
def _match_brightness(w, base, mask):
    ref = base[mask > 128].astype(np.float32)
    src = w[mask > 128].astype(np.float32)
    if ref.size == 0 or src.size == 0: return w
    rm, rs = ref.mean(0), ref.std(0) + 1e-6
    sm, ss = src.mean(0), src.std(0) + 1e-6
    out = w.copy().astype(np.float32)
    out[mask > 128] = np.clip((src - sm) / ss * rs + rm, 0, 255)
    return out.astype(np.uint8)

def paste_v6(base, face_pil, pts, order, can):
    H, W = base.shape[:2]
    pq   = _expand(_get_quad(pts, order))
    face = cv2.cvtColor(np.array(face_pil.convert("RGB")), cv2.COLOR_RGB2BGR)
    fH, fW = face.shape[:2]
    M = cv2.getPerspectiveTransform(
        np.float32([[0,0],[fW,0],[fW,fH],[0,fH]]), pq.astype(np.float32))
    warped = cv2.warpPerspective(face, M, (W, H))
    mask   = np.zeros((H, W), np.uint8)
    cv2.fillConvexPoly(mask, pq.astype(np.int32), 255)
    cp = np.array(can, np.int32)
    cp[:,0] = np.clip(cp[:,0], 0, W-1)
    cp[:,1] = np.clip(cp[:,1], 0, H-1)
    cv2.fillPoly(mask, [cp], 0)
    warped = _match_brightness(warped, base, mask)
    warped = cv2.GaussianBlur(warped, (3,3), 0)
    mb = cv2.GaussianBlur(mask, (5,5), 0)
    a  = mb.astype(np.float32)[:,:,None] / 255.0
    return np.clip(base * (1-a) + warped * a, 0, 255).astype(np.uint8)

def render_scene(bg_fname, imgs_by_pos):
    """imgs_by_pos: dict {left/center/right: PIL.Image}"""
    base = cv2.imread(str(SCENE_DIR / bg_fname))
    q, o = CARD_QUADS[bg_fname], CORNER_ORDER[bg_fname]
    can  = q["CAN"]
    for pos, face in imgs_by_pos.items():
        base = paste_v6(base, face, q[POS_KEY[pos]], o[POS_KEY[pos]], can)
    noise = np.random.normal(0, 1.5, base.shape).astype(np.int16)
    base  = np.clip(base.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    return Image.fromarray(cv2.cvtColor(
        cv2.resize(base, (256,256), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2RGB))

# ── Load 100-celeb image pool ─────────────────────────────────────────────────
hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")

print("Loading celebrities_faces_all_different…")
ds = load_dataset("robot-learning-group47/celebrities_faces_all_different",
                  split="train", token=hf_token)
print(f"  {len(ds)} images loaded")

# Group by celebrity name
celeb_imgs = defaultdict(list)
for row in ds:
    celeb_imgs[row["label"]].append(row["image"].convert("RGB"))

all_celebs = sorted(celeb_imgs.keys())
print(f"  {len(all_celebs)} unique celebrities")
print(f"  Avg images per celeb: {len(ds)/len(all_celebs):.1f}")

# ── Build dataset ─────────────────────────────────────────────────────────────
manifest = []
scene_idx = 0

POSITIONS = ["left", "center", "right"]

# Strategy: for each celebrity as the TARGET, generate n_per_slot scenes
# where they appear in each of the 3 slots. Two other celebrities are sampled
# randomly as distractors.
# n_per_slot × 3 slots × 5 backgrounds × 100 celebs
# = 14 × 3 × 5 × 100 = 21,000 scenes
# Each scene generates 3 QA pairs (one per prompt style: P0, P1, VLA)
# = 63,000 training examples

print(f"\nBuilding dataset…")
print(f"  Target: {args.n_per_slot} scenes × 3 slots × 5 bgs × {len(all_celebs)} celebs")
print(f"  = {args.n_per_slot * 3 * 5 * len(all_celebs):,} scenes")

for celeb_idx, target_name in enumerate(all_celebs):
    target_imgs = celeb_imgs[target_name]

    for target_pos in POSITIONS:
        other_positions = [p for p in POSITIONS if p != target_pos]

        for _ in range(args.n_per_slot):
            # Sample 2 distractor celebrities (different from target)
            distractors = random.sample(
                [c for c in all_celebs if c != target_name], 2)

            # Assign distractors to remaining positions
            random.shuffle(other_positions)
            pos_to_celeb = {
                target_pos: target_name,
                other_positions[0]: distractors[0],
                other_positions[1]: distractors[1],
            }
            celeb_to_pos = {v: k for k, v in pos_to_celeb.items()}

            # Sample 1 image per celebrity in scene
            imgs_by_pos = {}
            for pos, cname in pos_to_celeb.items():
                imgs_by_pos[pos] = random.choice(celeb_imgs[cname])

            # Sample background
            bg_fname = random.choice(BG_NAMES)

            # Render scene
            scene_img = render_scene(bg_fname, imgs_by_pos)
            scene_path = SCENE_OUT / f"scene_{scene_idx:06d}.jpg"
            scene_img.save(str(scene_path), quality=92)

            # Generate QA pairs for this scene
            # 1. P0 diagnostic (target celebrity)
            p0_q = random.choice(P0_TEMPLATES).format(name=target_name)
            manifest.append({
                "scene_idx": scene_idx,
                "image_path": str(scene_path),
                "question": p0_q,
                "answer": target_pos,
                "target": target_name,
                "target_pos": target_pos,
                "prompt_type": "P0",
                "pos_to_celeb": pos_to_celeb,
            })

            # 2. P1 action+grounding (target celebrity)
            p1_q = random.choice(P1_TEMPLATES).format(name=target_name)
            manifest.append({
                "scene_idx": scene_idx,
                "image_path": str(scene_path),
                "question": p1_q,
                "answer": target_pos,
                "target": target_name,
                "target_pos": target_pos,
                "prompt_type": "P1",
                "pos_to_celeb": pos_to_celeb,
            })

            # 3. VLA natural (target celebrity) — answer is position
            vla_q = random.choice(VLA_TEMPLATES).format(name=target_name)
            manifest.append({
                "scene_idx": scene_idx,
                "image_path": str(scene_path),
                "question": vla_q,
                "answer": target_pos,
                "target": target_name,
                "target_pos": target_pos,
                "prompt_type": "VLA",
                "pos_to_celeb": pos_to_celeb,
            })

            # 4. (bonus) P0 for one distractor — adds multi-query supervision
            distractor_name = distractors[0]
            distractor_pos  = celeb_to_pos[distractor_name]
            dist_q = random.choice(P0_TEMPLATES).format(name=distractor_name)
            manifest.append({
                "scene_idx": scene_idx,
                "image_path": str(scene_path),
                "question": dist_q,
                "answer": distractor_pos,
                "target": distractor_name,
                "target_pos": distractor_pos,
                "prompt_type": "P0_distractor",
                "pos_to_celeb": pos_to_celeb,
            })

            scene_idx += 1

        if (celeb_idx * 3 + POSITIONS.index(target_pos) + 1) % 30 == 0:
            print(f"  [{celeb_idx+1}/{len(all_celebs)}] {target_name} / {target_pos} "
                  f"→ {scene_idx} scenes, {len(manifest)} QA pairs", flush=True)

print(f"\nTotal: {scene_idx:,} scenes, {len(manifest):,} QA pairs")

# Save manifest
with open(MANIFEST, "w") as f:
    for entry in manifest:
        f.write(json.dumps(entry) + "\n")
print(f"Saved manifest → {MANIFEST}")

# Save summary
summary = {
    "n_scenes": scene_idx,
    "n_qa_pairs": len(manifest),
    "n_celebs": len(all_celebs),
    "n_per_slot": args.n_per_slot,
    "n_backgrounds": len(BG_NAMES),
    "prompt_types": {
        "P0": sum(1 for e in manifest if e["prompt_type"] == "P0"),
        "P1": sum(1 for e in manifest if e["prompt_type"] == "P1"),
        "VLA": sum(1 for e in manifest if e["prompt_type"] == "VLA"),
        "P0_distractor": sum(1 for e in manifest if e["prompt_type"] == "P0_distractor"),
    },
    "celebrities": all_celebs,
}
(OUT_DIR / "summary.json").write_text(json.dumps(summary, indent=2))
print(f"Saved summary → {OUT_DIR}/summary.json")
print(f"\nDone. Dataset ready at: {OUT_DIR}")
