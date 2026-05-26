#!/usr/bin/env python3
"""
eval3_spatial_grounding.py
===========================
Spatial grounding evaluation for Eval 3.

Goal: can the VLM correctly answer LEFT / CENTER / RIGHT for each identity
      in robot-view composite scenes built from *unseen* celebrity images?

For each of N_SCENES unique scenes:
  - Compose a v6 perspective-warped robot-view image
  - Ask "Where is Taylor Swift?" × 3 prompt styles
  - Ask "Where is Barack Obama?" × 3 prompt styles
  - Ask "Where is Yann LeCun?"  × 3 prompt styles
  = 9 queries per scene

Three prompt styles (SP0 / SP1 / SP2):
  SP0  robot-task framing (current deployment prompt)
  SP1  scene-description framing
  SP2  minimal direct question

Reports:
  - Overall accuracy per prompt style
  - Accuracy by identity (per prompt)
  - Accuracy by true position (per prompt)
  - Per-identity × per-position accuracy table
  - Position confusion matrices (pred vs true)
  - Center over-prediction rate
  - Yann-right → center confusion rate
  - Per-background template accuracy and bias
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
OUTPUT     = Path("./eval3_spatial_grounding_results.json")
N_SCENES   = 200
SEED       = 42
random.seed(SEED)
np.random.seed(SEED)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE  = torch.bfloat16 if DEVICE == "cuda" else torch.float32
print(f"Device: {DEVICE}  |  N_SCENES={N_SCENES}  |  Queries={N_SCENES*9}")

CLASSES    = ["taylor", "obama", "yann"]
FULL_NAMES = {"taylor": "Taylor Swift", "obama": "Barack Obama", "yann": "Yann LeCun"}
HF_LABELS  = {"Taylor Swift": "taylor", "Barack Obama": "obama", "Yann LeCun": "yann"}
POSITIONS  = ["left", "center", "right"]

# ── Prompts ───────────────────────────────────────────────────────────────────
SP0 = (
    "{name} is one of the three celebrities shown on cards in the robot scene. "
    "The cards are placed at the LEFT, CENTER, and RIGHT positions. "
    "Which position is {name} in? Answer with exactly one word: left, center, or right."
)
SP1 = (
    "This image shows three celebrity portraits on cards arranged left, center, and right. "
    "Where is {name}? Answer with exactly one word: left, center, or right."
)
SP2 = (
    "Where is {name}? Answer with exactly one word: left, center, or right."
)
PROMPTS = {"SP0": SP0, "SP1": SP1, "SP2": SP2}

# ── Parse response ────────────────────────────────────────────────────────────
def parse_position(raw):
    t = raw.strip().lower()
    for w in ["left", "center", "middle", "right"]:
        if w in t:
            return "center" if w == "middle" else w
    return "invalid"

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

def _get_quad(pts, order):
    return np.array([pts[i] for i in order], np.float32)

def _expand(q, f=0.12):
    cx, cy = q.mean(0)
    return (q + f * (q - np.array([[cx, cy]] * 4))).astype(int)

def _match_brightness(w, base, mask):
    ref = base[mask > 128].astype(np.float32)
    src = w[mask > 128].astype(np.float32)
    if ref.size == 0 or src.size == 0:
        return w
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
        np.float32([[0, 0], [fW, 0], [fW, fH], [0, fH]]),
        pq.astype(np.float32))
    warped = cv2.warpPerspective(face, M, (W, H))
    mask   = np.zeros((H, W), np.uint8)
    cv2.fillConvexPoly(mask, pq.astype(np.int32), 255)
    cp = np.array(can, np.int32)
    cp[:, 0] = np.clip(cp[:, 0], 0, W - 1)
    cp[:, 1] = np.clip(cp[:, 1], 0, H - 1)
    cv2.fillPoly(mask, [cp], 0)
    warped = _match_brightness(warped, base, mask)
    warped = cv2.GaussianBlur(warped, (3, 3), 0)
    mb = cv2.GaussianBlur(mask, (5, 5), 0)
    a  = mb.astype(np.float32)[:, :, None] / 255.0
    return np.clip(base * (1 - a) + warped * a, 0, 255).astype(np.uint8)

def make_scene(bg_fname, imgs_by_pos):
    base = cv2.imread(str(SCENE_DIR / bg_fname))
    q, o = CARD_QUADS[bg_fname], CORNER_ORDER[bg_fname]
    can  = q["CAN"]
    for pos, face in imgs_by_pos.items():
        base = paste_v6(base, face, q[POS_KEY[pos]], o[POS_KEY[pos]], can)
    noise = np.random.normal(0, 1.5, base.shape).astype(np.int16)
    base  = np.clip(base.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    return Image.fromarray(cv2.cvtColor(
        cv2.resize(base, (256, 256), interpolation=cv2.INTER_AREA),
        cv2.COLOR_BGR2RGB))

# ── Load HF dataset ───────────────────────────────────────────────────────────
print("\nLoading dataset…")
hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")
ds = load_dataset(DATASET_ID, split="train", token=hf_token)
unseen = defaultdict(list)
for row in ds:
    key = HF_LABELS.get(row["label"])
    if key:
        unseen[key].append(row["image"].convert("RGB"))
for cls in CLASSES:
    print(f"  {cls}: {len(unseen[cls])} images")

# ── Sample unique scenes ──────────────────────────────────────────────────────
ARRS = list(itertools.permutations(CLASSES))
seen_keys = set()
scene_specs = []
attempts = 0
while len(scene_specs) < N_SCENES and attempts < 100_000:
    attempts += 1
    ti = random.randrange(len(unseen["taylor"]))
    oi = random.randrange(len(unseen["obama"]))
    yi = random.randrange(len(unseen["yann"]))
    bg = random.choice(BG_NAMES)
    ar = random.randrange(len(ARRS))
    key = (ti, oi, yi, bg, ar)
    if key not in seen_keys:
        seen_keys.add(key)
        scene_specs.append((ti, oi, yi, bg, ARRS[ar]))
print(f"Sampled {len(scene_specs)} unique scenes ({attempts} attempts)\n")

# ── Load model ────────────────────────────────────────────────────────────────
print(f"Loading {MODEL_ID} …")
processor  = AutoProcessor.from_pretrained(MODEL_ID, token=hf_token)
base_model = SmolVLMForConditionalGeneration.from_pretrained(
    BASE_ID, torch_dtype=DTYPE, device_map=DEVICE, token=hf_token)
model = PeftModel.from_pretrained(base_model, MODEL_ID, token=hf_token)
model.eval()
print("Model ready.\n")

def ask(image, question):
    msgs   = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": question}]}]
    prompt = processor.apply_chat_template(msgs, add_generation_prompt=True)
    inputs = processor(text=prompt, images=[image], return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        ids = model.generate(**inputs, max_new_tokens=12, do_sample=False)
    return processor.decode(ids[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()

# ── Run evaluation ────────────────────────────────────────────────────────────
print("=" * 70)
print(f"SPATIAL GROUNDING EVAL — {N_SCENES} scenes × 3 identities × 3 prompts = {N_SCENES*9} queries")
print("=" * 70 + "\n")

all_results = []   # list of dicts with scene/prompt/target/expected/pred/ok
n_done = 0
n_ok   = 0

for scene_idx, (ti, oi, yi, bg, arrangement) in enumerate(scene_specs):
    face_imgs = {
        "taylor": unseen["taylor"][ti],
        "obama":  unseen["obama"][oi],
        "yann":   unseen["yann"][yi],
    }
    left, mid, right = arrangement[0], arrangement[1], arrangement[2]
    imgs_by_pos = {
        "left":   face_imgs[left],
        "center": face_imgs[mid],
        "right":  face_imgs[right],
    }
    pos_of = {cls: ["left","center","right"][arrangement.index(cls)] for cls in CLASSES}

    try:
        scene_img = make_scene(bg, imgs_by_pos)
    except Exception as e:
        print(f"[SKIP #{scene_idx:03d}] render error: {e}")
        continue

    for target in CLASSES:
        expected = pos_of[target]
        name     = FULL_NAMES[target]
        for prompt_key, prompt_tmpl in PROMPTS.items():
            raw  = ask(scene_img, prompt_tmpl.format(name=name))
            pred = parse_position(raw)
            ok   = (pred == expected)
            n_done += 1
            n_ok   += ok
            tag    = "OK" if ok else "FAIL"
            all_results.append({
                "scene":       scene_idx,
                "ti": ti, "oi": oi, "yi": yi,
                "bg":          bg,
                "arrangement": list(arrangement),
                "prompt":      prompt_key,
                "target":      target,
                "expected":    expected,
                "pred":        pred,
                "raw":         raw,
                "ok":          ok,
            })
            if not ok:
                print(f"  [{tag}] #{scene_idx:03d} {prompt_key} {target:<6s} "
                      f"true={expected:6s} pred={pred:6s} raw='{raw}'")

    if (scene_idx + 1) % 25 == 0:
        print(f"\n  ── {scene_idx+1}/{N_SCENES} scenes | "
              f"{n_ok}/{n_done} = {n_ok/n_done:.1%} overall ──\n")

# ── Reporting ─────────────────────────────────────────────────────────────────
SEP = "=" * 70
sep = "-" * 70

print(f"\n{SEP}")
print("RESULTS")
print(SEP)

# ── 1. Overall accuracy per prompt ────────────────────────────────────────────
print("\n[1] Overall accuracy per prompt style")
print(sep)
for pk in ["SP0", "SP1", "SP2"]:
    rows = [r for r in all_results if r["prompt"] == pk]
    c = sum(r["ok"] for r in rows)
    n = len(rows)
    print(f"  {pk}: {c}/{n} = {c/n:.1%}")

# ── 2. Accuracy by identity per prompt ───────────────────────────────────────
print("\n[2] Accuracy by identity")
print(sep)
header = f"  {'Identity':<16}"
for pk in ["SP0","SP1","SP2"]:
    header += f"  {pk:<12}"
print(header)
for cls in CLASSES:
    line = f"  {FULL_NAMES[cls]:<16}"
    for pk in ["SP0","SP1","SP2"]:
        rows = [r for r in all_results if r["target"]==cls and r["prompt"]==pk]
        c = sum(r["ok"] for r in rows)
        n = len(rows)
        line += f"  {c}/{n}={c/n:.1%}  "
    print(line)

# ── 3. Accuracy by true position per prompt ───────────────────────────────────
print("\n[3] Accuracy by true position")
print(sep)
header = f"  {'Position':<10}"
for pk in ["SP0","SP1","SP2"]:
    header += f"  {pk:<12}"
print(header)
for pos in POSITIONS:
    line = f"  {pos:<10}"
    for pk in ["SP0","SP1","SP2"]:
        rows = [r for r in all_results if r["expected"]==pos and r["prompt"]==pk]
        c = sum(r["ok"] for r in rows)
        n = len(rows)
        line += f"  {c}/{n}={c/n:.1%}  "
    print(line)

# ── 4. Per-identity × per-position accuracy (best prompt = SP0) ───────────────
for pk in ["SP0","SP1","SP2"]:
    print(f"\n[4] Per-identity x per-position — {pk}")
    print(sep)
    col_w = 12
    hdr = f"  {'identity':15s}"
    for pos in POSITIONS:
        hdr += f"  {pos:^{col_w}}"
    print(hdr)
    for cls in CLASSES:
        line = f"  {FULL_NAMES[cls]:<15}"
        for pos in POSITIONS:
            rows = [r for r in all_results
                    if r["target"]==cls and r["expected"]==pos and r["prompt"]==pk]
            if rows:
                c = sum(r["ok"] for r in rows)
                n = len(rows)
                cell = f"{c}/{n}={c/n:.0%}"
            else:
                cell = "—"
            line += f"  {cell:^{col_w}}"
        print(line)

# ── 5. Confusion matrices per prompt ──────────────────────────────────────────
for pk in ["SP0","SP1","SP2"]:
    print(f"\n[5] Confusion matrix — {pk}  (rows=true, cols=predicted)")
    print(sep)
    rows_data = [r for r in all_results if r["prompt"] == pk]
    pred_labels = POSITIONS + ["invalid"]
    col_w = 10
    _hdr_label = "true / pred"
    hdr_line = f"  {_hdr_label:12s}"
    for pl in pred_labels:
        hdr_line += f"  {pl:^{col_w}}"
    print(hdr_line)
    for true_pos in POSITIONS:
        line = f"  {true_pos:<12}"
        rows_pos = [r for r in rows_data if r["expected"] == true_pos]
        for pl in pred_labels:
            count = sum(1 for r in rows_pos if r["pred"] == pl)
            total = len(rows_pos)
            pct = count/total if total else 0
            cell = f"{count}({pct:.0%})"
            line += f"  {cell:^{col_w}}"
        print(line)

# ── 6. Center over-prediction ─────────────────────────────────────────────────
print("\n[6] Center over-prediction analysis")
print(sep)
for pk in ["SP0","SP1","SP2"]:
    rows = [r for r in all_results if r["prompt"] == pk]
    n_pred_center   = sum(1 for r in rows if r["pred"] == "center")
    n_true_center   = sum(1 for r in rows if r["expected"] == "center")
    n_wrong_as_ctr  = sum(1 for r in rows if r["pred"] == "center" and r["expected"] != "center")
    total = len(rows)
    print(f"  {pk}: predicted center {n_pred_center}/{total} ({n_pred_center/total:.1%})  "
          f"|  true center {n_true_center}/{total} ({n_true_center/total:.1%})  "
          f"|  false-center (wrong pred=center) {n_wrong_as_ctr}")

# ── 7. Yann-right → center confusion ─────────────────────────────────────────
print("\n[7] Yann-right confusion detail")
print(sep)
for pk in ["SP0","SP1","SP2"]:
    yr = [r for r in all_results if r["target"]=="yann" and r["expected"]=="right" and r["prompt"]==pk]
    c_ok       = sum(r["ok"] for r in yr)
    c_center   = sum(1 for r in yr if r["pred"] == "center")
    c_left     = sum(1 for r in yr if r["pred"] == "left")
    c_invalid  = sum(1 for r in yr if r["pred"] == "invalid")
    n = len(yr)
    print(f"  {pk} Yann-right: {c_ok}/{n}={c_ok/n:.1%} correct  "
          f"| pred=center {c_center}  pred=left {c_left}  pred=invalid {c_invalid}")

# ── 8. Per-background accuracy and bias ──────────────────────────────────────
print("\n[8] Per-background accuracy (SP0) — all identities")
print(sep)
for bg in BG_NAMES:
    short = bg[6:35]
    for pk in ["SP0"]:
        bg_rows = [r for r in all_results if r["bg"] == bg and r["prompt"] == pk]
        if not bg_rows:
            continue
        c = sum(r["ok"] for r in bg_rows)
        n = len(bg_rows)
        # position breakdown
        pos_str = ""
        for pos in POSITIONS:
            pr = [r for r in bg_rows if r["expected"] == pos]
            if pr:
                pc = sum(r["ok"] for r in pr)
                pos_str += f"  {pos}:{pc}/{len(pr)}"
        print(f"  {short:<30s} {c}/{n}={c/n:.1%}  [{pos_str.strip()}]")

print("\n[8b] Per-background Yann-right accuracy (SP0)")
print(sep)
for bg in BG_NAMES:
    short = bg[6:35]
    yr = [r for r in all_results
          if r["bg"]==bg and r["target"]=="yann" and r["expected"]=="right" and r["prompt"]=="SP0"]
    if yr:
        c = sum(r["ok"] for r in yr)
        n = len(yr)
        c_ctr = sum(1 for r in yr if r["pred"]=="center")
        print(f"  {short:<30s} {c}/{n}={c/n:.1%}  "
              f"(pred=center: {c_ctr}/{n}={c_ctr/n:.1%})")

# ── 9. Best prompt recommendation ────────────────────────────────────────────
print("\n[9] Prompt recommendation")
print(sep)
for pk in ["SP0","SP1","SP2"]:
    rows = [r for r in all_results if r["prompt"] == pk]
    c_all = sum(r["ok"] for r in rows)
    n_all = len(rows)
    yr = [r for r in rows if r["target"]=="yann" and r["expected"]=="right"]
    c_yr = sum(r["ok"] for r in yr)
    n_yr = len(yr)
    print(f"  {pk}: overall={c_all/n_all:.1%}  Yann-right={c_yr/n_yr:.1%}")

# determine best
best_pk = max(["SP0","SP1","SP2"], key=lambda pk:
    sum(r["ok"] for r in all_results if r["prompt"]==pk) /
    len([r for r in all_results if r["prompt"]==pk]))

best_yr_pk = max(["SP0","SP1","SP2"], key=lambda pk:
    sum(r["ok"] for r in all_results if r["prompt"]==pk and r["target"]=="yann" and r["expected"]=="right") /
    max(1, len([r for r in all_results if r["prompt"]==pk and r["target"]=="yann" and r["expected"]=="right"])))

print(f"\n  Best overall prompt: {best_pk}")
print(f"  Best Yann-right prompt: {best_yr_pk}")

# ── 10. Deployment verdict ────────────────────────────────────────────────────
print("\n[10] Deployment readiness")
print(sep)
for pk in ["SP0","SP1","SP2"]:
    rows = [r for r in all_results if r["prompt"] == pk]
    overall = sum(r["ok"] for r in rows) / len(rows)
    targets = {}
    for cls in CLASSES:
        cr = [r for r in rows if r["target"] == cls]
        targets[cls] = sum(r["ok"] for r in cr) / len(cr)
    yr = [r for r in rows if r["target"]=="yann" and r["expected"]=="right"]
    yr_acc = sum(r["ok"] for r in yr) / len(yr) if yr else 0

    verdict = "PASS" if (
        overall >= 0.97 and
        targets["taylor"] >= 0.97 and
        targets["obama"] >= 0.97 and
        targets["yann"] >= 0.95 and
        yr_acc >= 0.90
    ) else "FAIL"

    print(f"  {pk}: overall={overall:.1%}  taylor={targets['taylor']:.1%}  "
          f"obama={targets['obama']:.1%}  yann={targets['yann']:.1%}  "
          f"yann-right={yr_acc:.1%}  --> {verdict}")

# ── Save ──────────────────────────────────────────────────────────────────────
OUTPUT.write_text(json.dumps({
    "config": {"n_scenes": N_SCENES, "n_queries": len(all_results), "seed": SEED},
    "results": all_results,
}, indent=2, default=str))
print(f"\nSaved → {OUTPUT}")
