#!/usr/bin/env python3
"""
eval3_g4_200scenes.py
======================
G4 only: 200 unique robot-view scenes (v6 composition) with unseen images.

Design:
  - 200 unique randomly sampled (taylor_img, obama_img, yann_img, background, arrangement)
  - Per scene: 3 SP0-task grounding queries (one per target identity) + 1 SP1-scene query
    for Yann (the tricky one) = ~3.5 avg → 200 × 3.5 ≈ 700 total queries
  - Uses composite_test6.py perspective-warp composition

Run:
    python eval3_g4_200scenes.py
"""

import itertools, json, os, random, re
from collections import defaultdict
from pathlib import Path

import cv2, numpy as np, torch
from PIL import Image
from datasets import load_dataset
from peft import PeftModel
from transformers import AutoProcessor, SmolVLMForConditionalGeneration

# ── Config ────────────────────────────────────────────────────────────────────
MODEL_ID    = "robot-learning-group47/eval3_lora_fixed_Chengming"
BASE_ID     = "HuggingFaceTB/SmolVLM2-2.2B-Instruct"
DATASET_ID  = "robot-learning-group47/celebrities_faces_all_different"
SCENE_DIR   = Path("./data/scene_setup")
OUTPUT      = Path("./eval3_g4_200scenes_results.json")
N_SCENES    = 200
SEED        = 42
random.seed(SEED)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE  = torch.bfloat16 if DEVICE == "cuda" else torch.float32
print(f"Device: {DEVICE}")

# ── Labels ────────────────────────────────────────────────────────────────────
CLASSES    = ["taylor", "obama", "yann"]
FULL_NAMES = {"taylor": "Taylor Swift", "obama": "Barack Obama", "yann": "Yann LeCun"}
HF_LABELS  = {"Taylor Swift": "taylor", "Barack Obama": "obama", "Yann LeCun": "yann"}

def parse_position(raw):
    t = raw.strip().lower()
    for w in ["left", "center", "middle", "right"]:
        if w in t:
            return "center" if w == "middle" else w
    return None

# ── v6 composition data (from composite_test6.py) ─────────────────────────────
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

def _get_quad(pts, order): return np.array([pts[i] for i in order], np.float32)
def _expand(q, f=0.12):
    cx, cy = q.mean(0); return (q + f*(q - np.array([[cx,cy]]*4))).astype(int)
def _match_brightness(w, base, mask):
    ref = base[mask>128].astype(np.float32); src = w[mask>128].astype(np.float32)
    if ref.size==0 or src.size==0: return w
    rm,rs = ref.mean(0), ref.std(0)+1e-6; sm,ss = src.mean(0), src.std(0)+1e-6
    out = w.copy().astype(np.float32)
    out[mask>128] = np.clip((src-sm)/ss*rs+rm,0,255); return out.astype(np.uint8)

def paste_v6(base, face_pil, pts, order, can):
    H,W = base.shape[:2]
    pq  = _expand(_get_quad(pts, order))
    face = cv2.cvtColor(np.array(face_pil.convert("RGB")), cv2.COLOR_RGB2BGR)
    fH,fW = face.shape[:2]
    M = cv2.getPerspectiveTransform(
        np.float32([[0,0],[fW,0],[fW,fH],[0,fH]]), pq.astype(np.float32))
    warped = cv2.warpPerspective(face, M, (W,H))
    mask = np.zeros((H,W), np.uint8)
    cv2.fillConvexPoly(mask, pq.astype(np.int32), 255)
    cp = np.array(can, np.int32)
    cp[:,0]=np.clip(cp[:,0],0,W-1); cp[:,1]=np.clip(cp[:,1],0,H-1)
    cv2.fillPoly(mask, [cp], 0)
    warped = _match_brightness(warped, base, mask)
    warped = cv2.GaussianBlur(warped,(3,3),0)
    mb = cv2.GaussianBlur(mask,(5,5),0); a=mb.astype(np.float32)[:,:,None]/255.
    return np.clip(base*(1-a)+warped*a,0,255).astype(np.uint8)

def make_v6_scene(bg_fname, imgs_by_pos):
    base = cv2.imread(str(SCENE_DIR / bg_fname))
    q, o = CARD_QUADS[bg_fname], CORNER_ORDER[bg_fname]
    can  = q["CAN"]
    for pos, face in imgs_by_pos.items():
        base = paste_v6(base, face, q[POS_KEY[pos]], o[POS_KEY[pos]], can)
    noise = np.random.normal(0,1.5,base.shape).astype(np.int16)
    base  = np.clip(base.astype(np.int16)+noise,0,255).astype(np.uint8)
    return Image.fromarray(cv2.cvtColor(
        cv2.resize(base,(256,256),interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2RGB))

# ── Prompts ───────────────────────────────────────────────────────────────────
SP0 = ("{name} is one of the three celebrities shown on cards in the robot scene. "
       "The cards are placed at the LEFT, CENTER, and RIGHT positions. "
       "Which position is {name} in? Answer with exactly one word: left, center, or right.")
SP1 = ("This image shows three celebrity portraits on cards arranged left, center, and right. "
       "Where is {name}? Answer with exactly one word: left, center, or right.")

# ── Load unseen images ────────────────────────────────────────────────────────
print("Loading dataset…")
ds = load_dataset(DATASET_ID, split="train")
unseen = defaultdict(list)
for row in ds:
    key = HF_LABELS.get(row["label"])
    if key:
        unseen[key].append(row["image"].convert("RGB"))

for cls in CLASSES:
    print(f"  {cls}: {len(unseen[cls])} images")

BG_NAMES   = list(CARD_QUADS.keys())
ARRS       = list(itertools.permutations(CLASSES))   # 6 arrangements

# ── Sample 200 unique scenes ──────────────────────────────────────────────────
seen_keys = set()
scene_specs = []
attempts = 0
while len(scene_specs) < N_SCENES and attempts < 50_000:
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

print(f"\nSampled {len(scene_specs)} unique scenes ({attempts} attempts)")

# ── Load model ────────────────────────────────────────────────────────────────
hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")
print(f"\nLoading model {MODEL_ID}…")
processor = AutoProcessor.from_pretrained(MODEL_ID, token=hf_token)
base_model = SmolVLMForConditionalGeneration.from_pretrained(
    BASE_ID, torch_dtype=DTYPE, device_map=DEVICE, token=hf_token)
model = PeftModel.from_pretrained(base_model, MODEL_ID, token=hf_token)
model.eval()
print("Model ready.")

def ask(image, question, max_new_tokens=16):
    msgs = [{"role":"user","content":[{"type":"image"},{"type":"text","text":question}]}]
    prompt = processor.apply_chat_template(msgs, add_generation_prompt=True)
    inputs = processor(text=prompt, images=[image], return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        ids = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    return processor.decode(ids[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()

# ── Run G4 ────────────────────────────────────────────────────────────────────
# Per scene: SP0 for all 3 targets (3 queries) + SP1 for Yann only (1 query) = 3.5 avg → 700 total
print(f"\n{'='*70}")
print(f"G4 — {N_SCENES} unique robot-view scenes")
print(f"     SP0 × 3 targets  +  SP1 × 1 (Yann)  =  ~{N_SCENES*3 + N_SCENES} queries")
print(f"{'='*70}\n")

results = []
n_correct = 0
n_total   = 0

for scene_idx, (ti, oi, yi, bg, arrangement) in enumerate(scene_specs):
    face_imgs = {"taylor": unseen["taylor"][ti],
                 "obama":  unseen["obama"][oi],
                 "yann":   unseen["yann"][yi]}
    left, mid, right = arrangement
    imgs_by_pos = {"left": face_imgs[left], "center": face_imgs[mid], "right": face_imgs[right]}

    try:
        scene_img = make_v6_scene(bg, imgs_by_pos)
    except Exception as e:
        print(f"[SKIP {scene_idx}] render failed: {e}")
        continue

    pos_of = {cls: ["left","center","right"][arrangement.index(cls)] for cls in CLASSES}

    # SP0 for all 3 targets
    for target in CLASSES:
        expected = pos_of[target]
        raw  = ask(scene_img, SP0.format(name=FULL_NAMES[target]))
        pred = parse_position(raw)
        ok   = pred == expected or (expected == "center" and pred == "middle")
        n_correct += ok; n_total += 1
        tag = "✓" if ok else "✗"
        print(f"[{tag}] #{scene_idx:03d} SP0 bg={bg[6:10]} L={left[:4]} M={mid[:4]} R={right[:4]} "
              f"→ {FULL_NAMES[target]:<15s} exp={expected:6s} got='{raw}'")
        results.append({"scene":scene_idx,"ti":ti,"oi":oi,"yi":yi,"bg":bg,
                        "arrangement":list(arrangement),"prompt":"SP0",
                        "target":target,"expected":expected,"raw":raw,"pred":pred,"ok":ok})

    # SP1 for Yann only (extra prompt diversity)
    raw  = ask(scene_img, SP1.format(name=FULL_NAMES["yann"]))
    pred = parse_position(raw)
    ok   = pred == pos_of["yann"] or (pos_of["yann"] == "center" and pred == "middle")
    n_correct += ok; n_total += 1
    tag = "✓" if ok else "✗"
    print(f"[{tag}] #{scene_idx:03d} SP1 bg={bg[6:10]} L={left[:4]} M={mid[:4]} R={right[:4]} "
          f"→ {FULL_NAMES['yann']:<15s} exp={pos_of['yann']:6s} got='{raw}'")
    results.append({"scene":scene_idx,"ti":ti,"oi":oi,"yi":yi,"bg":bg,
                    "arrangement":list(arrangement),"prompt":"SP1",
                    "target":"yann","expected":pos_of["yann"],"raw":raw,"pred":pred,"ok":ok})

    # running accuracy every 20 scenes
    if (scene_idx + 1) % 20 == 0:
        print(f"\n  ── checkpoint {scene_idx+1}/{N_SCENES}: "
              f"{n_correct}/{n_total} = {n_correct/n_total:.1%} ──\n")

# ── Summary ───────────────────────────────────────────────────────────────────
print(f"\n{'='*70}")
print("RESULTS")
print(f"{'='*70}")
print(f"\n  Total queries : {n_total}")
print(f"  Total correct : {n_correct}  ({n_correct/n_total:.1%})")

# per-identity
print("\n  Per-identity (SP0 only):")
for cls in CLASSES:
    rows = [r for r in results if r["target"]==cls and r["prompt"]=="SP0"]
    c = sum(r["ok"] for r in rows)
    bar = "█"*c + "░"*(len(rows)-c)
    print(f"    {FULL_NAMES[cls]:<15s}: {c:3d}/{len(rows)}  {c/len(rows):.1%}  {bar}")

# per-position
print("\n  Per-expected-position (SP0 only):")
for pos in ["left","center","right"]:
    rows = [r for r in results if r["expected"]==pos and r["prompt"]=="SP0"]
    c = sum(r["ok"] for r in rows)
    print(f"    {pos:6s}: {c}/{len(rows)} = {c/len(rows):.1%}")

# per-background
print("\n  Per-background (SP0 only):")
for bg in BG_NAMES:
    rows = [r for r in results if r["bg"]==bg and r["prompt"]=="SP0"]
    if rows:
        c = sum(r["ok"] for r in rows)
        print(f"    {bg[6:30]}: {c}/{len(rows)} = {c/len(rows):.1%}")

# Yann SP0 vs SP1
yann_sp0 = [r for r in results if r["target"]=="yann" and r["prompt"]=="SP0"]
yann_sp1 = [r for r in results if r["target"]=="yann" and r["prompt"]=="SP1"]
print(f"\n  Yann SP0: {sum(r['ok'] for r in yann_sp0)}/{len(yann_sp0)} = "
      f"{sum(r['ok'] for r in yann_sp0)/len(yann_sp0):.1%}")
print(f"  Yann SP1: {sum(r['ok'] for r in yann_sp1)}/{len(yann_sp1)} = "
      f"{sum(r['ok'] for r in yann_sp1)/len(yann_sp1):.1%}")

# failures
failures = [r for r in results if not r["ok"]]
print(f"\n  Failures: {len(failures)}/{n_total}")
if failures:
    fail_by_id  = defaultdict(int)
    fail_by_pos = defaultdict(int)
    fail_by_bg  = defaultdict(int)
    for r in failures:
        fail_by_id[r["target"]]  += 1
        fail_by_pos[r["expected"]] += 1
        fail_by_bg[r["bg"][6:15]] += 1
    print(f"    by identity : {dict(fail_by_id)}")
    print(f"    by position : {dict(fail_by_pos)}")
    print(f"    by background: {dict(fail_by_bg)}")
    print(f"\n  First 10 failures:")
    for r in failures[:10]:
        print(f"    #{r['scene']:03d} {r['prompt']} target={r['target']:6s} "
              f"exp={r['expected']:6s} pred={str(r['pred']):6s} raw='{r['raw']}'")

OUTPUT.write_text(json.dumps({"n_scenes":N_SCENES,"n_queries":n_total,
    "accuracy":n_correct/n_total,"results":results}, indent=2, default=str))
print(f"\nSaved → {OUTPUT}")
