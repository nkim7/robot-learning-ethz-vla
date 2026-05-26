#!/usr/bin/env python3
"""
eval3_backbone_probe.py
=======================
Diagnostic probe: does the VLA's VLM backbone correctly encode spatial position
for target identities in robot-view composite scenes?

This is NOT an external resolver. The model is SmolVLM2 + LoRA acting as the VLA's
VLM backbone. Left/center/right accuracy is a proxy for whether the backbone's
internal representation correctly maps identity → spatial slot, which the action
head then uses to drive placement.

Methodology: forced-choice scoring (token logits, not free-text generation).
Instead of sampling from the model, we compare the raw logit of the first token
for each of the three possible answers (left / center / right) and take the argmax.
This is more reliable for a clean 3-class classification diagnostic.

Tests 3 prompt styles:
  P0  diagnostic ("Where is X? Answer: left, center, or right.")
  P1  VLA natural + diagnostic tail ("Put the coke can on the image of X.
      Which slot contains X? Answer: left, center, or right.")
  P2  pure VLA natural prompt, forced-choice on logits only

Reports:
  [1]  Overall accuracy per prompt style
  [2]  Accuracy by identity (per prompt)
  [3]  Accuracy by true position (per prompt)
  [4]  Identity × position accuracy table
  [5]  Confusion matrix (rows=true, cols=predicted)
  [6]  Center over-prediction rate
  [7]  Yann-right → center confusion
  [8]  Per-background bias + Yann-right per background
  [9]  Best prompt recommendation
  [10] Deployment threshold verdict
"""

import argparse, itertools, json, os, random
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
parser = argparse.ArgumentParser()
parser.add_argument("--model", default="robot-learning-group47/eval3_lora_fixed_Chengming",
                    help="LoRA adapter to evaluate, or repair adapter when --base_lora is set")
parser.add_argument("--base_model", default="HuggingFaceTB/SmolVLM2-2.2B-Instruct")
parser.add_argument("--base_lora", default=None,
                    help="optional existing LoRA to merge before loading --model as a repair adapter")
parser.add_argument("--dataset", default="robot-learning-group47/celebrities_faces_all_different")
parser.add_argument("--scene_dir", default="./clelb_pic/in_order")
parser.add_argument("--output", default="./eval3_backbone_probe_results.json")
parser.add_argument("--n_scenes", type=int, default=200)
parser.add_argument("--seed", type=int, default=42)
args = parser.parse_args()

MODEL_ID   = args.model
BASE_ID    = args.base_model
DATASET_ID = args.dataset
SCENE_DIR  = Path(args.scene_dir)
OUTPUT     = Path(args.output)
N_SCENES   = args.n_scenes
SEED       = args.seed
random.seed(SEED)
np.random.seed(SEED)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE  = torch.bfloat16 if DEVICE == "cuda" else torch.float32
print(f"Device: {DEVICE}  |  N_SCENES={N_SCENES}  |  Queries={N_SCENES*9} (3 identities x 3 prompts)")

CLASSES    = ["taylor", "obama", "yann"]
FULL_NAMES = {"taylor": "Taylor Swift", "obama": "Barack Obama", "yann": "Yann LeCun"}
HF_LABELS  = {"Taylor Swift": "taylor", "Barack Obama": "obama", "Yann LeCun": "yann"}
POSITIONS  = ["left", "center", "right"]
CHOICES    = ["left", "center", "right"]

# ── Prompt styles ─────────────────────────────────────────────────────────────
# P0: diagnostic, mirrors what a VQA probe would ask
P0 = ("There are three celebrity cards in this robot scene, "
      "arranged in the left, center, and right slots. "
      "Which slot is {name} in? "
      "Answer with exactly one word: left, center, or right.")

# P1: framed as a VLA action instruction + grounding question
P1 = ("Put the coke can on the image of {name}. "
      "The cards are arranged left, center, and right. "
      "Which slot contains {name}? "
      "Answer with exactly one word: left, center, or right.")

# P2: pure natural VLA command — forced-choice on logits, no answer suffix
P2 = ("Put the coke can on the image of {name}.")

PROMPTS = {"P0": P0, "P1": P1, "P2": P2}

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

def make_scene(bg_fname, imgs_by_pos):
    base = cv2.imread(str(SCENE_DIR / bg_fname))
    q, o = CARD_QUADS[bg_fname], CORNER_ORDER[bg_fname]
    can  = q["CAN"]
    for pos, face in imgs_by_pos.items():
        base = paste_v6(base, face, q[POS_KEY[pos]], o[POS_KEY[pos]], can)
    noise = np.random.normal(0, 1.5, base.shape).astype(np.int16)
    base  = np.clip(base.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    return Image.fromarray(cv2.cvtColor(
        cv2.resize(base, (256,256), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2RGB))

# ── Load dataset ──────────────────────────────────────────────────────────────
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

# ── Sample scenes ─────────────────────────────────────────────────────────────
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
print(f"Sampled {len(scene_specs)} unique scenes\n")

# ── Load model ────────────────────────────────────────────────────────────────
print(f"Loading model {MODEL_ID} …")
processor_source = MODEL_ID if args.base_lora is None else BASE_ID
processor  = AutoProcessor.from_pretrained(processor_source, token=hf_token)
base_model = SmolVLMForConditionalGeneration.from_pretrained(
    BASE_ID, torch_dtype=DTYPE, device_map=DEVICE, token=hf_token)
if args.base_lora:
    print(f"Merging base LoRA first: {args.base_lora}")
    base_model = PeftModel.from_pretrained(base_model, args.base_lora, token=hf_token)
    base_model = base_model.merge_and_unload()
model = PeftModel.from_pretrained(base_model, MODEL_ID, token=hf_token)
model.eval()

# Precompute first-token IDs for each choice (with and without leading space)
def get_choice_token_ids():
    tids = {}
    for c in CHOICES:
        # try "left", " left", "Left" — take whichever gives a single token first
        for variant in [c, " " + c, c.capitalize(), " " + c.capitalize()]:
            ids = processor.tokenizer.encode(variant, add_special_tokens=False)
            if ids:
                tids[c] = ids[0]
                break
    return tids

CHOICE_TIDS = get_choice_token_ids()
print(f"Choice token IDs: {CHOICE_TIDS}")
print("Model ready.\n")

def forced_choice(image, question_text):
    """Return (predicted_position, score_dict).
    Scores are raw logits at the first generated token position.
    """
    msgs   = [{"role":"user","content":[{"type":"image"},{"type":"text","text":question_text}]}]
    prompt = processor.apply_chat_template(msgs, add_generation_prompt=True)
    inputs = processor(text=prompt, images=[image], return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        logits = model(**inputs).logits[0, -1, :]  # [vocab_size]
    scores = {c: logits[tid].item() for c, tid in CHOICE_TIDS.items()}
    pred   = max(scores, key=scores.get)
    return pred, scores

# ── Run evaluation ────────────────────────────────────────────────────────────
print("=" * 70)
print(f"BACKBONE SPATIAL GROUNDING PROBE")
print(f"{N_SCENES} scenes × 3 identities × 3 prompts = {N_SCENES*9} forced-choice queries")
print("=" * 70 + "\n")

all_results = []
n_done = 0
n_ok   = 0

for scene_idx, (ti, oi, yi, bg, arrangement) in enumerate(scene_specs):
    face_imgs = {
        "taylor": unseen["taylor"][ti],
        "obama":  unseen["obama"][oi],
        "yann":   unseen["yann"][yi],
    }
    left, mid, right = arrangement[0], arrangement[1], arrangement[2]
    imgs_by_pos = {"left": face_imgs[left], "center": face_imgs[mid], "right": face_imgs[right]}
    pos_of = {cls: POSITIONS[arrangement.index(cls)] for cls in CLASSES}

    try:
        scene_img = make_scene(bg, imgs_by_pos)
    except Exception as e:
        print(f"[SKIP #{scene_idx:03d}] {e}")
        continue

    for target in CLASSES:
        expected = pos_of[target]
        name     = FULL_NAMES[target]
        for pk, tmpl in PROMPTS.items():
            pred, scores = forced_choice(scene_img, tmpl.format(name=name))
            ok = (pred == expected)
            n_done += 1
            n_ok   += ok
            all_results.append({
                "scene": scene_idx,
                "ti": ti, "oi": oi, "yi": yi,
                "bg":          bg,
                "arrangement": list(arrangement),
                "prompt":      pk,
                "target":      target,
                "expected":    expected,
                "pred":        pred,
                "scores":      scores,
                "ok":          ok,
            })
            if not ok:
                sc = ", ".join(f"{c}:{scores[c]:.1f}" for c in CHOICES)
                print(f"  FAIL #{scene_idx:03d} {pk} {target:<6s} "
                      f"true={expected:6s} pred={pred:6s}  [{sc}]")

    if (scene_idx + 1) % 25 == 0:
        print(f"\n  -- {scene_idx+1}/{N_SCENES} scenes | "
              f"{n_ok}/{n_done} = {n_ok/n_done:.1%} overall --\n")

# ── Reporting ─────────────────────────────────────────────────────────────────
SEP = "=" * 70
sep = "-" * 70

print(f"\n{SEP}")
print("SPATIAL GROUNDING PROBE RESULTS")
print(SEP)

# [1] Overall per prompt
print("\n[1] Overall accuracy (forced-choice)")
print(sep)
for pk in ["P0","P1","P2"]:
    rows = [r for r in all_results if r["prompt"] == pk]
    c = sum(r["ok"] for r in rows)
    n = len(rows)
    print(f"  {pk}: {c}/{n} = {c/n:.1%}")

# [2] By identity
print("\n[2] Accuracy by identity")
print(sep)
id_hdr = f"  {'Identity':<16}"
for pk in ["P0","P1","P2"]:
    id_hdr += f"  {pk:<14}"
print(id_hdr)
for cls in CLASSES:
    line = f"  {FULL_NAMES[cls]:<16}"
    for pk in ["P0","P1","P2"]:
        rows = [r for r in all_results if r["target"]==cls and r["prompt"]==pk]
        c = sum(r["ok"] for r in rows)
        n = len(rows)
        line += f"  {c}/{n}={c/n:.1%}     "
    print(line)

# [3] By true position
print("\n[3] Accuracy by true position")
print(sep)
pos_hdr = f"  {'Position':<10}"
for pk in ["P0","P1","P2"]:
    pos_hdr += f"  {pk:<14}"
print(pos_hdr)
for pos in POSITIONS:
    line = f"  {pos:<10}"
    for pk in ["P0","P1","P2"]:
        rows = [r for r in all_results if r["expected"]==pos and r["prompt"]==pk]
        c = sum(r["ok"] for r in rows)
        n = len(rows)
        line += f"  {c}/{n}={c/n:.1%}     "
    print(line)

# [4] Identity x position (best prompt = P0, then P1)
for pk in ["P0","P1","P2"]:
    print(f"\n[4] Identity x position — {pk}")
    print(sep)
    col_w = 14
    hdr = f"  {'':16}"
    for pos in POSITIONS:
        hdr += f"  {pos:^{col_w}}"
    print(hdr)
    for cls in CLASSES:
        line = f"  {FULL_NAMES[cls]:<16}"
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

# [5] Confusion matrix
for pk in ["P0","P1","P2"]:
    print(f"\n[5] Confusion matrix — {pk}  (rows=true, cols=predicted)")
    print(sep)
    pred_labels = POSITIONS
    col_w = 12
    hdr_label = "true / pred"
    hdr_line  = f"  {hdr_label:14}"
    for pl in pred_labels:
        hdr_line += f"  {pl:^{col_w}}"
    print(hdr_line)
    pk_rows = [r for r in all_results if r["prompt"] == pk]
    for true_pos in POSITIONS:
        line = f"  {true_pos:<14}"
        pos_rows = [r for r in pk_rows if r["expected"] == true_pos]
        n = len(pos_rows)
        for pl in pred_labels:
            count = sum(1 for r in pos_rows if r["pred"] == pl)
            pct = count/n if n else 0
            cell = f"{count}({pct:.0%})"
            line += f"  {cell:^{col_w}}"
        print(line)

# [6] Center over-prediction
print("\n[6] Center over-prediction")
print(sep)
for pk in ["P0","P1","P2"]:
    rows = [r for r in all_results if r["prompt"] == pk]
    n_pred_ctr  = sum(1 for r in rows if r["pred"] == "center")
    n_true_ctr  = sum(1 for r in rows if r["expected"] == "center")
    n_false_ctr = sum(1 for r in rows if r["pred"]=="center" and r["expected"]!="center")
    n = len(rows)
    overpred_rate = n_pred_ctr / n
    expected_rate = n_true_ctr / n
    bias = overpred_rate - expected_rate
    sign = "+" if bias >= 0 else ""
    print(f"  {pk}: pred_center={n_pred_ctr}/{n}({overpred_rate:.1%})  "
          f"true_center={n_true_ctr}/{n}({expected_rate:.1%})  "
          f"bias={sign}{bias:.1%}  false_center_errs={n_false_ctr}")

# [7] Yann-right confusion
print("\n[7] Yann-right confusion detail")
print(sep)
for pk in ["P0","P1","P2"]:
    yr = [r for r in all_results if r["target"]=="yann" and r["expected"]=="right" and r["prompt"]==pk]
    c_ok  = sum(r["ok"] for r in yr)
    c_ctr = sum(1 for r in yr if r["pred"]=="center")
    c_lft = sum(1 for r in yr if r["pred"]=="left")
    n = len(yr)
    print(f"  {pk} Yann-right ({n} queries): correct={c_ok}/{n}={c_ok/n:.1%}  "
          f"→center={c_ctr}({c_ctr/n:.1%})  →left={c_lft}({c_lft/n:.1%})")
    # Print per-scene logit margins for failures
    fails = [r for r in yr if not r["ok"]]
    if fails:
        print(f"       Failure score margins (right vs center logit):")
        for r in fails[:6]:
            margin = r["scores"].get("right", 0) - r["scores"].get("center", 0)
            print(f"         #{r['scene']:03d} bg={r['bg'][6:12]} "
                  f"right={r['scores'].get('right',0):.1f} "
                  f"center={r['scores'].get('center',0):.1f} "
                  f"left={r['scores'].get('left',0):.1f} "
                  f"margin={margin:.1f}")

# [8] Per-background accuracy and Yann-right breakdown
print("\n[8] Per-background accuracy (P0)")
print(sep)
pk = "P0"
for bg in BG_NAMES:
    short = bg[6:36]
    bg_rows = [r for r in all_results if r["bg"]==bg and r["prompt"]==pk]
    c = sum(r["ok"] for r in bg_rows)
    n = len(bg_rows)
    # per-position
    pos_parts = []
    for pos in POSITIONS:
        pr = [r for r in bg_rows if r["expected"]==pos]
        if pr:
            pc = sum(r["ok"] for r in pr)
            pos_parts.append(f"{pos}:{pc}/{len(pr)}")
    print(f"  {short:<32} {c}/{n}={c/n:.1%}  [{' | '.join(pos_parts)}]")

print(f"\n[8b] Yann-right per background (P0)")
print(sep)
for bg in BG_NAMES:
    short = bg[6:36]
    yr = [r for r in all_results
          if r["bg"]==bg and r["target"]=="yann" and r["expected"]=="right" and r["prompt"]=="P0"]
    if yr:
        c = sum(r["ok"] for r in yr)
        n = len(yr)
        c_ctr = sum(1 for r in yr if r["pred"]=="center")
        print(f"  {short:<32} {c}/{n}={c/n:.1%}  (→center: {c_ctr}/{n}={c_ctr/n:.1%})")

# [9] Best prompt
print("\n[9] Prompt summary")
print(sep)
for pk in ["P0","P1","P2"]:
    rows = [r for r in all_results if r["prompt"]==pk]
    overall = sum(r["ok"] for r in rows) / len(rows)
    yr = [r for r in rows if r["target"]=="yann" and r["expected"]=="right"]
    yr_acc = sum(r["ok"] for r in yr) / len(yr) if yr else 0
    print(f"  {pk}: overall={overall:.1%}  Yann-right={yr_acc:.1%}")

best_overall = max(["P0","P1","P2"], key=lambda pk:
    sum(r["ok"] for r in all_results if r["prompt"]==pk) /
    len([r for r in all_results if r["prompt"]==pk]))
best_yr = max(["P0","P1","P2"], key=lambda pk:
    sum(r["ok"] for r in all_results if r["prompt"]==pk and r["target"]=="yann" and r["expected"]=="right") /
    max(1, len([r for r in all_results if r["prompt"]==pk and r["target"]=="yann" and r["expected"]=="right"])))
print(f"\n  Best overall: {best_overall}  |  Best Yann-right: {best_yr}")

# [10] Deployment verdict
print("\n[10] Deployment threshold verdict (backbone spatial grounding)")
print(sep)
print("  Thresholds: overall>=97%  Taylor>=97%  Obama>=97%  Yann>=95%  Yann-right>=95%")
print(sep)
for pk in ["P0","P1","P2"]:
    rows = [r for r in all_results if r["prompt"]==pk]
    overall = sum(r["ok"] for r in rows) / len(rows)
    by_id = {}
    for cls in CLASSES:
        cr = [r for r in rows if r["target"]==cls]
        by_id[cls] = sum(r["ok"] for r in cr) / len(cr)
    yr = [r for r in rows if r["target"]=="yann" and r["expected"]=="right"]
    yr_acc = sum(r["ok"] for r in yr) / len(yr) if yr else 0

    checks = {
        "overall>=97%":       overall >= 0.97,
        "Taylor>=97%":        by_id["taylor"] >= 0.97,
        "Obama>=97%":         by_id["obama"]  >= 0.97,
        "Yann>=95%":          by_id["yann"]   >= 0.95,
        "Yann-right>=95%":    yr_acc          >= 0.95,
    }
    passed = all(checks.values())
    verdict = "PASS -- backbone meets deployment threshold" if passed else "FAIL -- backbone needs repair"
    fails_str = ", ".join(k for k, v in checks.items() if not v) or "none"
    print(f"\n  {pk}:  overall={overall:.1%}  taylor={by_id['taylor']:.1%}  "
          f"obama={by_id['obama']:.1%}  yann={by_id['yann']:.1%}  yann-right={yr_acc:.1%}")
    print(f"        --> {verdict}")
    if not passed:
        print(f"        Failing checks: {fails_str}")

# ── Save ──────────────────────────────────────────────────────────────────────
OUTPUT.write_text(json.dumps({
    "config": {"model": MODEL_ID, "n_scenes": N_SCENES,
               "base_model": BASE_ID, "base_lora": args.base_lora,
               "n_queries": len(all_results), "seed": SEED,
               "choice_token_ids": CHOICE_TIDS},
    "results": all_results,
}, indent=2, default=str))
print(f"\nSaved -> {OUTPUT}")
