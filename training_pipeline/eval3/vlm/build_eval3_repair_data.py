#!/usr/bin/env python3
"""
build_eval3_repair_data.py
===========================
Build synthetic repair dataset for Eval 3 VLA backbone spatial grounding.

The VLA backbone (SmolVLM2 + LoRA) must learn that given a robot-view image
with three celebrity cards in left/center/right slots, a natural identity-based
VLA instruction ("Put the coke can on the image of Yann LeCun.") should activate
the correct spatial slot.

Each training example:
  image  : robot-view v6 composite (three celebrity cards in scene)
  question: natural VLA instruction or diagnostic grounding question
  answer : exactly one of: left | center | right

Dataset design:
  - All 6 identity-position permutations (L/C/R assignments)
  - All 5 real background templates
  - Image pool: original training images + unseen HF images
  - Per (arrangement, background): sample K image variants
  - Oversample: Yann-left × 3, Yann-right × 2, Yann-center × 2
  - Counterfactual pairs: same Yann image, different position — same bg
  - Mixed prompt styles: VLA natural + diagnostic + slot-only auxiliary
  - Label comes from actual rendered coordinates, NEVER from filename

Output:
  ./data/eval3_repair/manifest.jsonl   (question/answer/image_path)
  ./data/eval3_repair/scenes/          (rendered PNG files)

Usage:
  python3 build_eval3_repair_data.py [--n_base 12] [--seed 42] [--scene_dir ./celeb_face_samples/in_order]
"""

import argparse, itertools, json, os, random
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from datasets import load_dataset

parser = argparse.ArgumentParser()
parser.add_argument("--n_base",    type=int, default=12, help="image variants per (arr, bg) combo")
parser.add_argument("--seed",      type=int, default=42)
parser.add_argument("--out_dir",   default="./data/eval3_repair")
parser.add_argument("--scene_dir", default="./celeb_face_samples/in_order")
parser.add_argument("--local_toy_dir", default="./celeb_face_samples",
                    help="fallback local copy of the 15 original Eval 3 celebrity images")
parser.add_argument("--yann_left_mult",   type=int, default=3)
parser.add_argument("--yann_center_mult", type=int, default=2)
parser.add_argument("--yann_right_mult",  type=int, default=2)
args = parser.parse_args()

random.seed(args.seed)
np.random.seed(args.seed)

OUT_DIR   = Path(args.out_dir)
SCENE_OUT = OUT_DIR / "scenes"
SCENE_OUT.mkdir(parents=True, exist_ok=True)
MANIFEST  = OUT_DIR / "manifest.jsonl"
SCENE_DIR = Path(args.scene_dir)
LOCAL_TOY_DIR = Path(args.local_toy_dir)

CLASSES    = ["taylor", "obama", "yann"]
FULL_NAMES = {"taylor": "Taylor Swift", "obama": "Barack Obama", "yann": "Yann LeCun"}
HF_LABELS  = {"Taylor Swift": "taylor", "Barack Obama": "obama", "Yann LeCun": "yann"}
POSITIONS  = ["left", "center", "right"]

def to_rgb_image(value):
    if isinstance(value, Image.Image):
        return value.convert("RGB")
    if isinstance(value, dict) and value.get("path"):
        return Image.open(value["path"]).convert("RGB")
    if isinstance(value, str):
        path = Path(value)
        if path.exists():
            return Image.open(path).convert("RGB")
        raise FileNotFoundError(value)
    raise TypeError(f"Unsupported image value: {type(value)}")

def load_local_toy_images():
    local = defaultdict(list)
    if not LOCAL_TOY_DIR.exists():
        return local
    for cls in CLASSES:
        paths = sorted(set(LOCAL_TOY_DIR.glob(f"{cls}_*.png")) | set(LOCAL_TOY_DIR.glob(f"{cls}*.png")))
        for path in paths:
            local[cls].append(Image.open(path).convert("RGB"))
    return local

# ── Prompt templates ──────────────────────────────────────────────────────────
# VLA natural action prompts (main Eval 3 format)
VLA_TEMPLATES = [
    "Put the coke can on the image of {name}.",
    "Place the coke can on {name}'s picture.",
    "Move the coke can to the card showing {name}.",
    "Put the coke can on {name}.",
    "Place the can on the card with {name}.",
]
# P1-style action framing plus a grounding tail. This is the failure mode
# exposed by the final backbone probe, especially for Yann-left.
P1_TEMPLATES = [
    ("Put the coke can on the image of {name}. The cards are arranged left, center, "
     "and right. Which slot contains {name}? Answer with exactly one word: left, center, or right."),
    ("Place the coke can on {name}'s picture. The portrait cards occupy the left, center, "
     "and right slots. Which slot contains {name}? Answer with exactly one word: left, center, or right."),
    ("Move the coke can to the card showing {name}. In this robot view, is {name} on the "
     "left, center, or right? Answer with exactly one word."),
]
# Diagnostic grounding prompts (probe spatial encoding)
DIAG_TEMPLATES = [
    ("There are three celebrity cards in this robot scene arranged in left, center, "
     "and right slots. Which slot is {name} in? Answer with exactly one word: left, center, or right."),
    ("The robot sees three celebrity portrait cards. {name} is in one of three positions: "
     "left, center, or right. Which position is {name} in? Answer: left, center, or right."),
    ("In this robot scene there are three cards side by side. Where is {name}? "
     "Answer with exactly one word: left, center, or right."),
]
# Slot-only auxiliary (position-supervision signal, not primary Eval 3 format)
AUX_TEMPLATES = [
    "Put the coke can on the {pos} card.",
    "Move the can to the {pos} position.",
    "Place the coke can in the {pos} slot.",
]

# ── v6 Composition ─────────────────────────────────────────────────────────────
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
    base = cv2.imread(str(SCENE_DIR / bg_fname))
    q, o = CARD_QUADS[bg_fname], CORNER_ORDER[bg_fname]
    can  = q["CAN"]
    for pos, face in imgs_by_pos.items():
        base = paste_v6(base, face, q[POS_KEY[pos]], o[POS_KEY[pos]], can)
    noise = np.random.normal(0, 1.5, base.shape).astype(np.int16)
    base  = np.clip(base.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    return Image.fromarray(cv2.cvtColor(
        cv2.resize(base, (256,256), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2RGB))

# ── Load image pools ──────────────────────────────────────────────────────────
hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")

print("Loading original training images (eval3_TOY_video_images)…")
toy_ds = load_dataset("robot-learning-group47/eval3_TOY_video_images", split="train", token=hf_token)
toy_imgs = defaultdict(list)
toy_load_errors = 0
for row in toy_ds:
    raw_label = row.get("label") or row.get("identity") or row.get("target_person")
    key = HF_LABELS.get(raw_label)
    if key is None:
        # try to infer from image filename if label column missing
        fields = [
            str(row.get("file_name","") or row.get("image_path","")),
            str(row.get("prompt","")),
            str(row.get("target_person","")),
        ]
        haystack = " ".join(fields).lower()
        for k, v in {
            "taylor": "taylor",
            "swift": "taylor",
            "obama": "obama",
            "barack": "obama",
            "yann": "yann",
            "lecun": "yann",
            "le cun": "yann",
        }.items():
            if k in haystack:
                key = v; break
    if key:
        try:
            toy_imgs[key].append(to_rgb_image(row["image"]))
        except Exception:
            toy_load_errors += 1
if toy_load_errors:
    print(f"  warning: {toy_load_errors} HF original image paths were not locally readable; "
          f"falling back to {LOCAL_TOY_DIR}")
local_toy = load_local_toy_images()
for cls in CLASSES:
    if not toy_imgs[cls] and local_toy[cls]:
        toy_imgs[cls].extend(local_toy[cls])
for cls in CLASSES:
    print(f"  toy {cls}: {len(toy_imgs[cls])} images")

print("Loading unseen images (celebrities_faces_all_different)…")
unseen_ds = load_dataset("robot-learning-group47/celebrities_faces_all_different",
                         split="train", token=hf_token)
unseen_imgs = defaultdict(list)
for row in unseen_ds:
    key = HF_LABELS.get(row["label"])
    if key:
        unseen_imgs[key].append(to_rgb_image(row["image"]))
for cls in CLASSES:
    print(f"  unseen {cls}: {len(unseen_imgs[cls])} images")

# Merge: original training first, then unseen
all_imgs = {}
for cls in CLASSES:
    all_imgs[cls] = toy_imgs[cls] + unseen_imgs[cls]
    print(f"  total {cls}: {len(all_imgs[cls])} images")

# ── Build dataset ─────────────────────────────────────────────────────────────
ARRS = list(itertools.permutations(CLASSES))  # 6 arrangements

def yann_pos_in(arr):
    """Return position name of yann in this arrangement."""
    return POSITIONS[arr.index("yann")]

def oversample_factor(arr):
    """How many image variants to generate for this (arr, bg) combo."""
    yp = yann_pos_in(arr)
    if yp == "left":    return args.n_base * args.yann_left_mult
    if yp == "right":   return args.n_base * args.yann_right_mult
    if yp == "center":  return args.n_base * args.yann_center_mult
    return args.n_base

all_records = []   # each record = one QA pair
scene_counter = 0
seen_scene_keys = set()

print(f"\nGenerating dataset (n_base={args.n_base}) …")
print(f"  6 arrangements × 5 backgrounds = 30 combos")
print(f"  Oversample: Yann-left × {args.yann_left_mult}, "
      f"Yann-center × {args.yann_center_mult}, Yann-right × {args.yann_right_mult}")

for arr in ARRS:
    yp = yann_pos_in(arr)
    for bg in BG_NAMES:
        n_variants = oversample_factor(arr)
        generated = 0
        attempts  = 0
        while generated < n_variants and attempts < 10_000:
            attempts += 1
            ti = random.randrange(len(all_imgs["taylor"]))
            oi = random.randrange(len(all_imgs["obama"]))
            yi = random.randrange(len(all_imgs["yann"]))
            key = (ti, oi, yi, bg, tuple(arr))
            if key in seen_scene_keys:
                continue
            seen_scene_keys.add(key)

            left, mid, right = arr[0], arr[1], arr[2]
            imgs_by_pos = {
                "left":   all_imgs[left][ti % len(all_imgs[left])],
                "center": all_imgs[mid][oi % len(all_imgs[mid])],
                "right":  all_imgs[right][yi % len(all_imgs[right])],
            }
            # Re-map: the index is relative to the identity, not the position
            # Correct mapping:
            imgs_by_pos = {
                "left":   all_imgs[arr[0]][ti if arr[0]=="taylor" else (oi if arr[0]=="obama" else yi)],
                "center": all_imgs[arr[1]][ti if arr[1]=="taylor" else (oi if arr[1]=="obama" else yi)],
                "right":  all_imgs[arr[2]][ti if arr[2]=="taylor" else (oi if arr[2]=="obama" else yi)],
            }

            try:
                scene_img = render_scene(bg, imgs_by_pos)
            except Exception as e:
                print(f"  [skip] render error: {e}")
                continue

            # Save scene image with a clean name (no position info in filename)
            scene_fname = f"scene_{scene_counter:05d}.png"
            scene_path  = str(SCENE_OUT / scene_fname)
            scene_img.save(scene_path)

            # Ground truth positions (from actual rendered coordinates)
            pos_of = {cls: POSITIONS[arr.index(cls)] for cls in CLASSES}

            # ── Generate QA pairs for this scene ──────────────────────────────
            # 1. VLA natural prompts (one per identity, random template)
            for target in CLASSES:
                true_pos = pos_of[target]
                name     = FULL_NAMES[target]
                tmpl = random.choice(VLA_TEMPLATES)
                all_records.append({
                    "image":    scene_path,
                    "question": tmpl.format(name=name),
                    "answer":   true_pos,
                    "scene_id": scene_counter,
                    "prompt_type": "vla",
                    "target":   target,
                    "true_pos": true_pos,
                    "bg":       bg,
                    "arr":      list(arr),
                    "ti": ti, "oi": oi, "yi": yi,
                })

            # 2. P1-style prompts (one per identity, with extra Yann-left pressure)
            for target in CLASSES:
                true_pos = pos_of[target]
                name     = FULL_NAMES[target]
                n_p1 = 4 if target == "yann" and true_pos == "left" else 1
                for _ in range(n_p1):
                    tmpl = random.choice(P1_TEMPLATES)
                    all_records.append({
                        "image":    scene_path,
                        "question": tmpl.format(name=name),
                        "answer":   true_pos,
                        "scene_id": scene_counter,
                        "prompt_type": "p1_tail",
                        "target":   target,
                        "true_pos": true_pos,
                        "bg":       bg,
                        "arr":      list(arr),
                        "ti": ti, "oi": oi, "yi": yi,
                    })

            # 3. Diagnostic grounding prompts (one per identity, random template)
            for target in CLASSES:
                true_pos = pos_of[target]
                name     = FULL_NAMES[target]
                tmpl = random.choice(DIAG_TEMPLATES)
                all_records.append({
                    "image":    scene_path,
                    "question": tmpl.format(name=name),
                    "answer":   true_pos,
                    "scene_id": scene_counter,
                    "prompt_type": "diag",
                    "target":   target,
                    "true_pos": true_pos,
                    "bg":       bg,
                    "arr":      list(arr),
                    "ti": ti, "oi": oi, "yi": yi,
                })

            # 4. Slot-only auxiliary: keep this sparse so it cannot dominate.
            target = random.choice(CLASSES)
            true_pos = pos_of[target]
            tmpl = random.choice(AUX_TEMPLATES)
            all_records.append({
                "image":    scene_path,
                "question": tmpl.format(pos=true_pos),
                "answer":   true_pos,
                "scene_id": scene_counter,
                "prompt_type": "aux_slot",
                "target":   target,
                "true_pos": true_pos,
                "bg":       bg,
                "arr":      list(arr),
                "ti": ti, "oi": oi, "yi": yi,
            })

            scene_counter += 1
            generated     += 1

# ── Counterfactual pairs ──────────────────────────────────────────────────────
# Same Yann image in different positions: for each Yann image index, generate
# one scene with Yann-left, one Yann-center, one Yann-right on same background.
print(f"\nGenerating counterfactual pairs (same Yann image, all 3 positions)…")
cf_bgs    = BG_NAMES[:2]   # use 2 backgrounds for counterfactuals
n_yann    = min(10, len(all_imgs["yann"]))
n_taylor  = len(all_imgs["taylor"])
n_obama   = len(all_imgs["obama"])

for yi in range(n_yann):
    for bg in cf_bgs:
        ti = random.randrange(n_taylor)
        oi = random.randrange(n_obama)
        # Yann in each of the 3 positions
        for yann_pos in POSITIONS:
            other = [c for c in CLASSES if c != "yann"]
            # Build arrangement: put yann at yann_pos, others at remaining slots
            remaining = [p for p in POSITIONS if p != yann_pos]
            arr_dict  = {yann_pos: "yann",
                         remaining[0]: other[0],
                         remaining[1]: other[1]}
            imgs_by_pos = {
                "left":   all_imgs[arr_dict["left"]][ti if arr_dict["left"]=="taylor" else
                           (oi if arr_dict["left"]=="obama" else yi)],
                "center": all_imgs[arr_dict["center"]][ti if arr_dict["center"]=="taylor" else
                           (oi if arr_dict["center"]=="obama" else yi)],
                "right":  all_imgs[arr_dict["right"]][ti if arr_dict["right"]=="taylor" else
                           (oi if arr_dict["right"]=="obama" else yi)],
            }
            try:
                scene_img = render_scene(bg, imgs_by_pos)
            except Exception as e:
                continue

            scene_fname = f"scene_{scene_counter:05d}_cf.png"
            scene_path  = str(SCENE_OUT / scene_fname)
            scene_img.save(scene_path)

            # QA pairs for Yann (the counterfactual target), with P1 included.
            for prompt_type, tmpl in [
                ("counterfactual_vla", random.choice(VLA_TEMPLATES)),
                ("counterfactual_p1_tail", random.choice(P1_TEMPLATES)),
                ("counterfactual_diag", random.choice(DIAG_TEMPLATES)),
            ]:
                all_records.append({
                    "image":    scene_path,
                    "question": tmpl.format(name="Yann LeCun"),
                    "answer":   yann_pos,
                    "scene_id": scene_counter,
                    "prompt_type": prompt_type,
                    "target":   "yann",
                    "true_pos": yann_pos,
                    "bg":       bg,
                    "arr":      [arr_dict["left"], arr_dict["center"], arr_dict["right"]],
                    "ti": ti, "oi": oi, "yi": yi,
                })
            scene_counter += 1

# ── Stats ─────────────────────────────────────────────────────────────────────
print(f"\nDataset summary:")
print(f"  Total scenes     : {scene_counter}")
print(f"  Total QA pairs   : {len(all_records)}")

by_type = defaultdict(int)
by_target = defaultdict(int)
by_pos    = defaultdict(int)
for r in all_records:
    by_type[r["prompt_type"]] += 1
    by_target[r["target"]]    += 1
    by_pos[r["true_pos"]]     += 1

print(f"  By prompt type   : {dict(by_type)}")
print(f"  By identity      : {dict(by_target)}")
print(f"  By true position : {dict(by_pos)}")

yann_right = sum(1 for r in all_records if r["target"]=="yann" and r["true_pos"]=="right")
yann_ctr   = sum(1 for r in all_records if r["target"]=="yann" and r["true_pos"]=="center")
yann_left  = sum(1 for r in all_records if r["target"]=="yann" and r["true_pos"]=="left")
print(f"  Yann-left pairs  : {yann_left}")
print(f"  Yann-right pairs : {yann_right}")
print(f"  Yann-center pairs: {yann_ctr}")

# ── Shuffle and write manifest ─────────────────────────────────────────────────
random.shuffle(all_records)
with open(MANIFEST, "w") as f:
    for r in all_records:
        f.write(json.dumps(r) + "\n")
print(f"\nManifest saved -> {MANIFEST}")
print(f"Scenes saved   -> {SCENE_OUT}/ ({scene_counter} files)")
