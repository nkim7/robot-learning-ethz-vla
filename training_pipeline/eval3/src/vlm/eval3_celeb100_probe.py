#!/usr/bin/env python3
"""
eval3_celeb100_probe.py
========================
Forced-choice spatial grounding probe for ALL 100 celebrities.
Target: 99% P0 and P1 accuracy overall.

For each scene: 3 random celebrities placed in left/center/right slots.
For each of the 3 targets: ask P0 and P1, score via logit forced-choice.

Reports:
  [1] Overall P0 / P1 accuracy
  [2] Per-celebrity accuracy (worst-first)
  [3] Per-position accuracy
  [4] Eval-3 targets detail (Taylor Swift, Barack Obama, Yann LeCun)
  [5] PASS / FAIL verdict at 99% threshold

Usage:
  python3 eval3_celeb100_probe.py \
      --model robot-learning-group47/eval3_phase2_celeb100 \
      --n_scenes 500
"""

import argparse, json, os, random
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from datasets import load_dataset
from peft import PeftModel
from transformers import AutoProcessor, SmolVLMForConditionalGeneration

parser = argparse.ArgumentParser()
parser.add_argument("--model",      default="robot-learning-group47/eval3_phase2_celeb100")
parser.add_argument("--base_model", default="HuggingFaceTB/SmolVLM2-2.2B-Instruct")
parser.add_argument("--base_lora",  default="robot-learning-group47/eval3_phase1_celeb",
                    help="phase1 LoRA to merge before loading phase2 adapter")
parser.add_argument("--dataset",    default="robot-learning-group47/celebrities_faces_all_different")
parser.add_argument("--scene_dir",  default="./clelb_pic/in_order")
parser.add_argument("--output",     default="./celeb100_probe_results.json")
parser.add_argument("--n_scenes",   type=int, default=500)
parser.add_argument("--seed",       type=int, default=42)
args = parser.parse_args()

random.seed(args.seed)
np.random.seed(args.seed)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE  = torch.bfloat16 if DEVICE == "cuda" else torch.float32
SCENE_DIR = Path(args.scene_dir)
EVAL3_TARGETS = {"Taylor Swift", "Barack Obama", "Yann LeCun"}
POSITIONS = ["left", "center", "right"]
THRESHOLD = 0.99

P0 = ("There are three celebrity cards in this robot scene arranged in left, center, "
      "and right slots. Which slot is {name} in? Answer with exactly one word: left, center, or right.")
P1 = ("Put the coke can on the image of {name}. The cards are arranged left, center, "
      "and right. Which slot contains {name}? Answer with exactly one word: left, center, or right.")

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
hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")
print("Loading celebrities_faces_all_different…")
ds = load_dataset(args.dataset, split="train", token=hf_token)
celeb_imgs = defaultdict(list)
for row in ds:
    celeb_imgs[row["label"]].append(row["image"].convert("RGB"))
all_celebs = sorted(celeb_imgs.keys())
print(f"  {len(all_celebs)} celebrities, {len(ds)} images")

# ── Load model ────────────────────────────────────────────────────────────────
print(f"\nLoading model…")
processor = AutoProcessor.from_pretrained(args.base_model, token=hf_token)
base = SmolVLMForConditionalGeneration.from_pretrained(
    args.base_model, dtype=DTYPE, device_map=DEVICE, token=hf_token)
print(f"  Merging phase1 LoRA: {args.base_lora}")
base = PeftModel.from_pretrained(base, args.base_lora, token=hf_token)
base = base.merge_and_unload()
print(f"  Loading phase2 adapter: {args.model}")
model = PeftModel.from_pretrained(base, args.model, token=hf_token)
model.eval()

CHOICES = ["left", "center", "right"]
choice_tids = {}
for c in CHOICES:
    for variant in [c, " "+c, c.capitalize(), " "+c.capitalize()]:
        ids = processor.tokenizer.encode(variant, add_special_tokens=False)
        if ids:
            choice_tids[c] = ids[0]; break
print(f"  Choice token IDs: {choice_tids}")
print("Model ready.\n")

def forced_choice(img, question):
    msgs   = [{"role":"user","content":[{"type":"image"},{"type":"text","text":question}]}]
    prompt = processor.apply_chat_template(msgs, add_generation_prompt=True)
    inputs = processor(text=prompt, images=[img], return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        logits = model(**inputs).logits[0, -1, :]
    scores = {c: logits[tid].item() for c, tid in choice_tids.items()}
    return max(scores, key=scores.get), scores

# ── Sample scenes ─────────────────────────────────────────────────────────────
print(f"Sampling {args.n_scenes} scenes…")
scenes = []
for _ in range(args.n_scenes):
    triplet = random.sample(all_celebs, 3)
    random.shuffle(POSITIONS)
    pos_to_celeb = dict(zip(POSITIONS, triplet))
    bg = random.choice(BG_NAMES)
    scenes.append({"pos_to_celeb": pos_to_celeb, "bg": bg,
                   "imgs": {p: random.choice(celeb_imgs[c]) for p, c in pos_to_celeb.items()}})

# ── Run probe ─────────────────────────────────────────────────────────────────
SEP = "=" * 70
print(SEP)
print(f"CELEB100 SPATIAL GROUNDING PROBE  ({args.n_scenes} scenes × 3 targets × 2 prompts = {args.n_scenes*6} queries)")
print(SEP + "\n")

all_results = []
for si, scene in enumerate(scenes):
    img = make_scene(scene["bg"], scene["imgs"])
    for target_name, target_pos in {v: k for k, v in scene["pos_to_celeb"].items()}.items():
        for pk, tmpl in [("P0", P0), ("P1", P1)]:
            pred, scores = forced_choice(img, tmpl.format(name=target_name))
            ok = pred == target_pos
            all_results.append({
                "scene": si, "bg": scene["bg"],
                "target": target_name, "expected": target_pos,
                "prompt": pk, "pred": pred, "scores": scores, "ok": ok,
            })
            if not ok:
                print(f"  FAIL {pk} scene={si:04d} {target_name:<28} true={target_pos:6s} pred={pred}")
    if (si + 1) % 50 == 0:
        n_done = len(all_results)
        n_ok   = sum(r["ok"] for r in all_results)
        print(f"  -- {si+1}/{args.n_scenes} scenes | {n_ok}/{n_done} = {n_ok/n_done:.1%} --", flush=True)

# ── Report ────────────────────────────────────────────────────────────────────
sep = "-" * 70
print(f"\n{SEP}\nRESULTS\n{SEP}")

# [1] Overall
print("\n[1] Overall accuracy")
print(sep)
for pk in ["P0", "P1"]:
    rows = [r for r in all_results if r["prompt"] == pk]
    c = sum(r["ok"] for r in rows); n = len(rows)
    verdict = "✓ PASS" if c/n >= THRESHOLD else "✗ FAIL"
    print(f"  {pk}: {c}/{n} = {c/n:.2%}  {verdict} (threshold {THRESHOLD:.0%})")

# [2] Per-celebrity (worst first)
print("\n[2] Per-celebrity accuracy — P0 (worst first)")
print(sep)
by_celeb = defaultdict(lambda: defaultdict(list))
for r in all_results:
    by_celeb[r["target"]][r["prompt"]].append(r["ok"])

celeb_p0 = {name: (sum(v["P0"]), len(v["P0"])) for name, v in by_celeb.items() if v["P0"]}
sorted_celebs = sorted(celeb_p0.items(), key=lambda x: x[1][0]/x[1][1])

weak = []
for name, (c, n) in sorted_celebs:
    acc  = c / n
    star = " ★" if name in EVAL3_TARGETS else ""
    flag = " ← WEAK" if acc < 0.80 else ""
    print(f"  {name:<32}{c:3d}/{n:<3d} = {acc:.0%}{star}{flag}")
    if acc < 0.80:
        weak.append(name)

# [3] Per-position
print("\n[3] Per-position accuracy")
print(sep)
for pk in ["P0", "P1"]:
    print(f"  {pk}:")
    for pos in ["left", "center", "right"]:
        rows = [r for r in all_results if r["prompt"]==pk and r["expected"]==pos]
        c = sum(r["ok"] for r in rows); n = len(rows)
        print(f"    {pos:8s}: {c}/{n} = {c/n:.1%}")

# [4] Eval-3 targets
print("\n[4] Eval-3 target celebrities")
print(sep)
for name in sorted(EVAL3_TARGETS):
    print(f"  {name}:")
    for pk in ["P0", "P1"]:
        rows = [r for r in all_results if r["target"]==name and r["prompt"]==pk]
        c = sum(r["ok"] for r in rows); n = len(rows)
        if n:
            print(f"    {pk}: {c}/{n} = {c/n:.1%}")
            for pos in ["left", "center", "right"]:
                pr = [r for r in rows if r["expected"]==pos]
                if pr:
                    pc = sum(r["ok"] for r in pr)
                    print(f"      {pos}: {pc}/{len(pr)} = {pc/len(pr):.0%}")

# [5] Final verdict
print(f"\n{SEP}")
print("FINAL VERDICT")
print(SEP)
p0_rows = [r for r in all_results if r["prompt"]=="P0"]
p1_rows = [r for r in all_results if r["prompt"]=="P1"]
p0_acc  = sum(r["ok"] for r in p0_rows) / len(p0_rows)
p1_acc  = sum(r["ok"] for r in p1_rows) / len(p1_rows)
passed  = p0_acc >= THRESHOLD and p1_acc >= THRESHOLD

print(f"  P0: {p0_acc:.2%}  P1: {p1_acc:.2%}  threshold: {THRESHOLD:.0%}")
print(f"  {'✓ PASS — ready for deployment' if passed else '✗ FAIL — needs more training'}")
if weak:
    print(f"  Weak celebrities (<80%): {', '.join(weak)}")
if not passed:
    gap_p0 = THRESHOLD - p0_acc
    gap_p1 = THRESHOLD - p1_acc
    print(f"  Gap to 99%: P0={gap_p0:+.2%}  P1={gap_p1:+.2%}")

# Save
Path(args.output).write_text(json.dumps({
    "config": vars(args),
    "summary": {"p0": p0_acc, "p1": p1_acc, "passed": passed,
                "n_scenes": args.n_scenes, "n_celebs": len(all_celebs)},
    "results": all_results,
}, indent=2, default=str))
print(f"\nSaved → {args.output}")
