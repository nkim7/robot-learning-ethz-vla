#!/usr/bin/env python3
"""
eval3_step3_step4.py
=====================
Step 3: Compare SP0 vs SP1 for ALL identities on the saved 200 scenes.
        Confirm SP1 doesn't hurt Taylor/Obama; see if it brings Yann >= 95%.
Step 4: Background bias audit on Yann-right SP0 failures:
        (a) neutral grey background — same card positions, no real bg texture
        (b) shuffled background — different bg template, same arrangement

Reads:
    ~/eval3_audit/eval3_g4_200scenes_results.json
    ~/eval3_audit/data/scene_setup/*.jpg
    HF dataset: robot-learning-group47/celebrities_faces_all_different
"""

import itertools, json, os, random
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from datasets import load_dataset
from peft import PeftModel
from transformers import AutoProcessor, SmolVLMForConditionalGeneration

# ── Config ────────────────────────────────────────────────────────────────────
MODEL_ID   = "robot-learning-group47/eval3_lora_fixed_Chengming"
BASE_ID    = "HuggingFaceTB/SmolVLM2-2.2B-Instruct"
DATASET_ID = "robot-learning-group47/celebrities_faces_all_different"
SCENE_DIR  = Path("./data/scene_setup")
RESULTS_IN = Path("./eval3_g4_200scenes_results.json")
OUTPUT     = Path("./eval3_step3_step4_results.json")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE  = torch.bfloat16 if DEVICE == "cuda" else torch.float32
print(f"Device: {DEVICE}")

CLASSES    = ["taylor", "obama", "yann"]
FULL_NAMES = {"taylor": "Taylor Swift", "obama": "Barack Obama", "yann": "Yann LeCun"}
HF_LABELS  = {"Taylor Swift": "taylor", "Barack Obama": "obama", "Yann LeCun": "yann"}

def parse_position(raw):
    t = raw.strip().lower()
    for w in ["left", "center", "middle", "right"]:
        if w in t:
            return "center" if w == "middle" else w
    return None

# ── v6 composition ─────────────────────────────────────────────────────────────
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
POS_KEY = {"left": "LEFT", "center": "CENTER", "right": "RIGHT"}
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
    pq = _expand(_get_quad(pts, order))
    face = cv2.cvtColor(np.array(face_pil.convert("RGB")), cv2.COLOR_RGB2BGR)
    fH, fW = face.shape[:2]
    M = cv2.getPerspectiveTransform(
        np.float32([[0,0],[fW,0],[fW,fH],[0,fH]]), pq.astype(np.float32))
    warped = cv2.warpPerspective(face, M, (W, H))
    mask = np.zeros((H, W), np.uint8)
    cv2.fillConvexPoly(mask, pq.astype(np.int32), 255)
    cp = np.array(can, np.int32)
    cp[:,0] = np.clip(cp[:,0], 0, W-1)
    cp[:,1] = np.clip(cp[:,1], 0, H-1)
    cv2.fillPoly(mask, [cp], 0)
    warped = _match_brightness(warped, base, mask)
    warped = cv2.GaussianBlur(warped, (3, 3), 0)
    mb = cv2.GaussianBlur(mask, (5, 5), 0)
    a = mb.astype(np.float32)[:,:,None] / 255.
    return np.clip(base * (1 - a) + warped * a, 0, 255).astype(np.uint8)

def make_v6_scene(bg_fname, imgs_by_pos):
    """Standard scene on real background."""
    base = cv2.imread(str(SCENE_DIR / bg_fname))
    q, o = CARD_QUADS[bg_fname], CORNER_ORDER[bg_fname]
    can = q["CAN"]
    for pos, face in imgs_by_pos.items():
        base = paste_v6(base, face, q[POS_KEY[pos]], o[POS_KEY[pos]], can)
    noise = np.random.normal(0, 1.5, base.shape).astype(np.int16)
    base = np.clip(base.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    return Image.fromarray(cv2.cvtColor(
        cv2.resize(base, (256, 256), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2RGB))

def make_grey_scene(bg_fname, imgs_by_pos):
    """Neutral grey background — same card coordinates, no real background texture."""
    base_real = cv2.imread(str(SCENE_DIR / bg_fname))
    H, W = base_real.shape[:2]
    # Mid-grey canvas matching approximate scene luminance
    base = np.full((H, W, 3), 128, dtype=np.uint8)
    q, o = CARD_QUADS[bg_fname], CORNER_ORDER[bg_fname]
    can = q["CAN"]
    for pos, face in imgs_by_pos.items():
        base = paste_v6(base, face, q[POS_KEY[pos]], o[POS_KEY[pos]], can)
    return Image.fromarray(cv2.cvtColor(
        cv2.resize(base, (256, 256), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2RGB))

def make_shuffled_scene(orig_bg, imgs_by_pos, rng):
    """Shuffled background — pick a different bg template, keep same arrangement."""
    other_bgs = [b for b in BG_NAMES if b != orig_bg]
    new_bg = rng.choice(other_bgs)
    return make_v6_scene(new_bg, imgs_by_pos), new_bg

# ── Prompts ───────────────────────────────────────────────────────────────────
SP0 = ("{name} is one of the three celebrities shown on cards in the robot scene. "
       "The cards are placed at the LEFT, CENTER, and RIGHT positions. "
       "Which position is {name} in? Answer with exactly one word: left, center, or right.")
SP1 = ("This image shows three celebrity portraits on cards arranged left, center, and right. "
       "Where is {name}? Answer with exactly one word: left, center, or right.")

# ── Load existing 200-scene results ──────────────────────────────────────────
print(f"\nLoading prior results from {RESULTS_IN} …")
with open(RESULTS_IN) as f:
    prior = json.load(f)

# Rebuild scene specs list indexed by scene_idx
scene_specs_by_idx = {}
for r in prior["results"]:
    idx = r["scene"]
    if idx not in scene_specs_by_idx:
        scene_specs_by_idx[idx] = {
            "ti": r["ti"], "oi": r["oi"], "yi": r["yi"],
            "bg": r["bg"], "arrangement": r["arrangement"]
        }

all_scene_idxs = sorted(scene_specs_by_idx.keys())
print(f"  Loaded {len(all_scene_idxs)} scene specs")

# ── Load HF dataset images ────────────────────────────────────────────────────
print("\nLoading HF dataset…")
hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")
ds = load_dataset(DATASET_ID, split="train", token=hf_token)
unseen = defaultdict(list)
for row in ds:
    key = HF_LABELS.get(row["label"])
    if key:
        unseen[key].append(row["image"].convert("RGB"))
for cls in CLASSES:
    print(f"  {cls}: {len(unseen[cls])} images")

# ── Load model ────────────────────────────────────────────────────────────────
print(f"\nLoading model {MODEL_ID} …")
processor = AutoProcessor.from_pretrained(MODEL_ID, token=hf_token)
base_model = SmolVLMForConditionalGeneration.from_pretrained(
    BASE_ID, torch_dtype=DTYPE, device_map=DEVICE, token=hf_token)
model = PeftModel.from_pretrained(base_model, MODEL_ID, token=hf_token)
model.eval()
print("Model ready.\n")

def ask(image, question, max_new_tokens=16):
    msgs = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": question}]}]
    prompt = processor.apply_chat_template(msgs, add_generation_prompt=True)
    inputs = processor(text=prompt, images=[image], return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        ids = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    return processor.decode(ids[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()

# ─────────────────────────────────────────────────────────────────────────────
# STEP 3 — SP1 for ALL identities on all 200 scenes
# ─────────────────────────────────────────────────────────────────────────────
print("=" * 70)
print("STEP 3 — SP1 for all identities on 200 scenes")
print(f"         {len(all_scene_idxs)} scenes × 3 targets = {len(all_scene_idxs)*3} queries")
print("=" * 70)

step3_results = []
rng = random.Random(42)

for scene_idx in all_scene_idxs:
    spec = scene_specs_by_idx[scene_idx]
    ti, oi, yi = spec["ti"], spec["oi"], spec["yi"]
    bg = spec["bg"]
    arrangement = spec["arrangement"]

    face_imgs = {
        "taylor": unseen["taylor"][ti],
        "obama":  unseen["obama"][oi],
        "yann":   unseen["yann"][yi],
    }
    left, mid, right = arrangement[0], arrangement[1], arrangement[2]
    imgs_by_pos = {"left": face_imgs[left], "center": face_imgs[mid], "right": face_imgs[right]}

    try:
        scene_img = make_v6_scene(bg, imgs_by_pos)
    except Exception as e:
        print(f"  [SKIP {scene_idx}] render failed: {e}")
        continue

    pos_of = {cls: ["left", "center", "right"][arrangement.index(cls)] for cls in CLASSES}

    for target in CLASSES:
        expected = pos_of[target]
        raw  = ask(scene_img, SP1.format(name=FULL_NAMES[target]))
        pred = parse_position(raw)
        ok   = (pred == expected) or (expected == "center" and pred == "middle")
        tag  = "OK" if ok else "FAIL"
        step3_results.append({
            "scene": scene_idx, "ti": ti, "oi": oi, "yi": yi,
            "bg": bg, "arrangement": arrangement,
            "prompt": "SP1", "target": target,
            "expected": expected, "raw": raw, "pred": pred, "ok": ok,
        })
        if not ok:
            print(f"  [{tag}] #{scene_idx:03d} SP1 {target:<6s} exp={expected:6s} got='{raw}'")

    if (scene_idx + 1) % 50 == 0:
        done = len(step3_results)
        correct = sum(r["ok"] for r in step3_results)
        print(f"  checkpoint {scene_idx+1}/200: {correct}/{done} = {correct/done:.1%}")

# Step 3 report
print("\n--- STEP 3 REPORT ---")
for cls in CLASSES:
    rows = [r for r in step3_results if r["target"] == cls]
    c = sum(r["ok"] for r in rows)
    print(f"  {FULL_NAMES[cls]:<15s} SP1: {c}/{len(rows)} = {c/len(rows):.1%}")

print()
for cls in CLASSES:
    for pos in ["left", "center", "right"]:
        rows = [r for r in step3_results if r["target"] == cls and r["expected"] == pos]
        if rows:
            c = sum(r["ok"] for r in rows)
            print(f"  {FULL_NAMES[cls]:<15s} SP1 {pos:6s}: {c}/{len(rows)} = {c/len(rows):.1%}")

# Compare SP0 vs SP1 for each identity using prior results
print("\n--- SP0 vs SP1 comparison ---")
for cls in CLASSES:
    sp0_rows = [r for r in prior["results"] if r["target"] == cls and r["prompt"] == "SP0"]
    sp1_rows = [r for r in step3_results if r["target"] == cls]
    sp0_c = sum(r["ok"] for r in sp0_rows)
    sp1_c = sum(r["ok"] for r in sp1_rows)
    delta = sp1_c - sp0_c
    sign = "+" if delta >= 0 else ""
    print(f"  {FULL_NAMES[cls]:<15s}: SP0={sp0_c}/{len(sp0_rows)}={sp0_c/len(sp0_rows):.1%}  "
          f"SP1={sp1_c}/{len(sp1_rows)}={sp1_c/len(sp1_rows):.1%}  "
          f"delta={sign}{delta}")

print()
for cls in CLASSES:
    for pos in ["left", "center", "right"]:
        sp0_rows = [r for r in prior["results"]
                    if r["target"] == cls and r["prompt"] == "SP0" and r["expected"] == pos]
        sp1_rows = [r for r in step3_results if r["target"] == cls and r["expected"] == pos]
        if sp0_rows and sp1_rows:
            sp0_c = sum(r["ok"] for r in sp0_rows)
            sp1_c = sum(r["ok"] for r in sp1_rows)
            delta = sp1_c - sp0_c
            sign = "+" if delta >= 0 else ""
            print(f"  {FULL_NAMES[cls]:<15s} {pos:6s}: SP0={sp0_c}/{len(sp0_rows)}={sp0_c/len(sp0_rows):.1%}  "
                  f"SP1={sp1_c}/{len(sp1_rows)}={sp1_c/len(sp1_rows):.1%}  delta={sign}{delta}")

# ─────────────────────────────────────────────────────────────────────────────
# STEP 4 — Background bias audit on Yann-right SP0 failures
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("STEP 4 — Background bias audit: Yann-right SP0 failures")
print("=" * 70)

# Identify Yann-right SP0 failures from prior results
yann_right_sp0_fails = [
    r for r in prior["results"]
    if r["target"] == "yann" and r["expected"] == "right"
    and r["prompt"] == "SP0" and not r["ok"]
]
yann_right_sp0_all = [
    r for r in prior["results"]
    if r["target"] == "yann" and r["expected"] == "right" and r["prompt"] == "SP0"
]
print(f"\n  Yann-right SP0 overall: {sum(r['ok'] for r in yann_right_sp0_all)}/{len(yann_right_sp0_all)}")
print(f"  Failures to re-test: {len(yann_right_sp0_fails)}\n")

step4_results = []
rng4 = random.Random(123)

for i, fail_r in enumerate(yann_right_sp0_fails):
    ti, oi, yi = fail_r["ti"], fail_r["oi"], fail_r["yi"]
    bg = fail_r["bg"]
    arrangement = fail_r["arrangement"]

    face_imgs = {
        "taylor": unseen["taylor"][ti],
        "obama":  unseen["obama"][oi],
        "yann":   unseen["yann"][yi],
    }
    left, mid, right = arrangement[0], arrangement[1], arrangement[2]
    imgs_by_pos = {"left": face_imgs[left], "center": face_imgs[mid], "right": face_imgs[right]}
    expected = "right"

    # (a) Neutral grey background
    try:
        grey_img = make_grey_scene(bg, imgs_by_pos)
        raw_grey = ask(grey_img, SP0.format(name="Yann LeCun"))
        pred_grey = parse_position(raw_grey)
        ok_grey = (pred_grey == expected)
    except Exception as e:
        raw_grey, pred_grey, ok_grey = f"ERROR:{e}", None, False

    # (b) Shuffled background
    try:
        shuffled_img, new_bg = make_shuffled_scene(bg, imgs_by_pos, rng4)
        raw_shuf = ask(shuffled_img, SP0.format(name="Yann LeCun"))
        pred_shuf = parse_position(raw_shuf)
        ok_shuf = (pred_shuf == expected)
    except Exception as e:
        raw_shuf, pred_shuf, ok_shuf, new_bg = f"ERROR:{e}", None, False, "?"

    # (c) SP1 on original background (from step3 results if available)
    sp1_match = next(
        (r for r in step3_results
         if r["scene"] == fail_r["scene"] and r["target"] == "yann"),
        None
    )
    ok_sp1 = sp1_match["ok"] if sp1_match else None
    raw_sp1 = sp1_match["raw"] if sp1_match else "N/A"

    g_tag = "OK" if ok_grey else "X"
    s_tag = "OK" if ok_shuf else "X"
    p_tag = ("OK" if ok_sp1 else "X") if ok_sp1 is not None else "?"
    print(f"  #{fail_r['scene']:03d} orig_bg={bg[6:12]}  "
          f"grey=[{g_tag}]'{raw_grey}'  shuf=[{s_tag}]'{raw_shuf}'(bg={new_bg[6:12]})  "
          f"SP1=[{p_tag}]'{raw_sp1}'")

    step4_results.append({
        "scene": fail_r["scene"],
        "ti": ti, "oi": oi, "yi": yi,
        "orig_bg": bg, "shuffled_bg": new_bg,
        "arrangement": arrangement,
        "grey_raw": raw_grey, "grey_pred": pred_grey, "grey_ok": ok_grey,
        "shuf_raw": raw_shuf, "shuf_pred": pred_shuf, "shuf_ok": ok_shuf,
        "sp1_raw": raw_sp1, "sp1_ok": ok_sp1,
    })

n_grey_ok = sum(r["grey_ok"] for r in step4_results)
n_shuf_ok = sum(r["shuf_ok"] for r in step4_results)
n_sp1_ok  = sum(r["sp1_ok"]  for r in step4_results if r["sp1_ok"] is not None)
n_sp1_avail = sum(1 for r in step4_results if r["sp1_ok"] is not None)
total_f = len(step4_results)

print(f"\n--- STEP 4 REPORT (on {total_f} Yann-right SP0 failure scenes) ---")
print(f"  Grey bg rescues   : {n_grey_ok}/{total_f} = {n_grey_ok/total_f:.1%} "
      f"(orig=0%, bg-neutral improves by {n_grey_ok/total_f:.1%})")
print(f"  Shuffled bg rescues: {n_shuf_ok}/{total_f} = {n_shuf_ok/total_f:.1%}")
if n_sp1_avail > 0:
    print(f"  SP1 rescues       : {n_sp1_ok}/{n_sp1_avail} = {n_sp1_ok/n_sp1_avail:.1%}")

# Breakdown: what does grey bg predict?
grey_preds = [r["grey_pred"] for r in step4_results if r["grey_pred"]]
grey_pred_counts = defaultdict(int)
for p in grey_preds:
    grey_pred_counts[p] += 1
print(f"\n  Grey bg prediction distribution: {dict(grey_pred_counts)}")

shuf_preds = [r["shuf_pred"] for r in step4_results if r["shuf_pred"]]
shuf_pred_counts = defaultdict(int)
for p in shuf_preds:
    shuf_pred_counts[p] += 1
print(f"  Shuffled bg prediction distribution: {dict(shuf_pred_counts)}")

# ── Also: Yann-right accuracy on grey/shuffled for the FULL set (not just failures) ──
print("\n--- STEP 4 EXTRA: Yann-right SP0 accuracy on all scenes by bg variant ---")
yann_right_sp0_scenes = [
    r for r in prior["results"]
    if r["target"] == "yann" and r["expected"] == "right" and r["prompt"] == "SP0"
]
print(f"  Testing all {len(yann_right_sp0_scenes)} Yann-right SP0 scenes with grey bg…")

step4_full_results = []
for fail_r in yann_right_sp0_scenes:
    ti, oi, yi = fail_r["ti"], fail_r["oi"], fail_r["yi"]
    bg = fail_r["bg"]
    arrangement = fail_r["arrangement"]

    face_imgs = {
        "taylor": unseen["taylor"][ti],
        "obama":  unseen["obama"][oi],
        "yann":   unseen["yann"][yi],
    }
    left, mid, right = arrangement[0], arrangement[1], arrangement[2]
    imgs_by_pos = {"left": face_imgs[left], "center": face_imgs[mid], "right": face_imgs[right]}
    expected = "right"

    try:
        grey_img = make_grey_scene(bg, imgs_by_pos)
        raw_grey = ask(grey_img, SP0.format(name="Yann LeCun"))
        pred_grey = parse_position(raw_grey)
        ok_grey = (pred_grey == expected)
    except Exception as e:
        raw_grey, pred_grey, ok_grey = f"ERROR:{e}", None, False

    step4_full_results.append({
        "scene": fail_r["scene"], "orig_ok": fail_r["ok"],
        "grey_raw": raw_grey, "grey_pred": pred_grey, "grey_ok": ok_grey,
    })

n_orig_ok_yr = sum(r["orig_ok"] for r in step4_full_results)
n_grey_ok_yr = sum(r["grey_ok"] for r in step4_full_results)
n_yr = len(step4_full_results)
print(f"  Yann-right SP0 orig accuracy: {n_orig_ok_yr}/{n_yr} = {n_orig_ok_yr/n_yr:.1%}")
print(f"  Yann-right SP0 grey accuracy: {n_grey_ok_yr}/{n_yr} = {n_grey_ok_yr/n_yr:.1%}")
if n_yr > 0:
    delta_yr = n_grey_ok_yr - n_orig_ok_yr
    sign = "+" if delta_yr >= 0 else ""
    print(f"  Delta: {sign}{delta_yr} ({'background hurts' if delta_yr > 0 else 'background helps' if delta_yr < 0 else 'no effect'})")

# ── Final recommendation ──────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("FINAL RECOMMENDATION")
print("=" * 70)

# Get SP1 Yann numbers from step3
yann_sp1_all = [r for r in step3_results if r["target"] == "yann"]
yann_sp1_right = [r for r in step3_results if r["target"] == "yann" and r["expected"] == "right"]
yann_sp1_c_all = sum(r["ok"] for r in yann_sp1_all)
yann_sp1_c_right = sum(r["ok"] for r in yann_sp1_right)

taylor_sp1 = [r for r in step3_results if r["target"] == "taylor"]
obama_sp1  = [r for r in step3_results if r["target"] == "obama"]
t_c = sum(r["ok"] for r in taylor_sp1)
o_c = sum(r["ok"] for r in obama_sp1)

print(f"\n  SP1 summary:")
print(f"    Taylor: {t_c}/{len(taylor_sp1)} = {t_c/len(taylor_sp1):.1%}")
print(f"    Obama:  {o_c}/{len(obama_sp1)} = {o_c/len(obama_sp1):.1%}")
print(f"    Yann (all): {yann_sp1_c_all}/{len(yann_sp1_all)} = {yann_sp1_c_all/len(yann_sp1_all):.1%}")
print(f"    Yann-right: {yann_sp1_c_right}/{len(yann_sp1_right)} = {yann_sp1_c_right/len(yann_sp1_right):.1%}")

sp1_total_c = t_c + o_c + yann_sp1_c_all
sp1_total_n = len(taylor_sp1) + len(obama_sp1) + len(yann_sp1_all)
print(f"\n  SP1 overall: {sp1_total_c}/{sp1_total_n} = {sp1_total_c/sp1_total_n:.1%}")

targets_met = []
if t_c / len(taylor_sp1) >= 0.97:
    targets_met.append("Taylor OK (>=97%)")
if o_c / len(obama_sp1) >= 0.97:
    targets_met.append("Obama OK (>=97%)")
if yann_sp1_c_all / len(yann_sp1_all) >= 0.95:
    targets_met.append("Yann SP1 OK (>=95%)")
if sp1_total_c / sp1_total_n >= 0.97:
    targets_met.append("Overall SP1 OK (>=97%)")

print(f"\n  Targets met: {targets_met}")

if len(targets_met) >= 3 and "Yann SP1 OK (>=95%)" in targets_met:
    print("\n  --> RECOMMENDATION: Switch to SP1 for deployment. No fine-tuning needed.")
elif yann_sp1_c_right / len(yann_sp1_right) >= 0.90:
    print("\n  --> RECOMMENDATION: SP1 helps significantly. Consider SP1 for deployment.")
    print("      If Yann-right is still below 95%, run Step 5 (targeted repair dataset).")
else:
    print("\n  --> RECOMMENDATION: SP1 is insufficient. Proceed to Step 5 (repair fine-tuning).")
    bg_bias = n_grey_ok_yr > n_orig_ok_yr
    if bg_bias:
        print("      Background bias confirmed — include background augmentation in repair data.")

# ── Save ──────────────────────────────────────────────────────────────────────
out = {
    "step3": {
        "description": "SP1 for all identities on 200 scenes",
        "n_scenes": len(all_scene_idxs),
        "results": step3_results,
    },
    "step4": {
        "description": "Background bias audit on Yann-right SP0 failures",
        "failure_retest": step4_results,
        "full_yann_right_grey": step4_full_results,
    },
}
OUTPUT.write_text(json.dumps(out, indent=2, default=str))
print(f"\nSaved → {OUTPUT}")
