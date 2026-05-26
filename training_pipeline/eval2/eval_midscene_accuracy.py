#!/usr/bin/env python3
"""
Evaluate smolvlm2-500m-vqa-lora-merged on real trajectory frames
from eval2_fixed_bowl_fixed_banana_108 at 4 phases:
  initial(5%) / post_grasp(40%) / bowl_visible(65%) / branch(80%)

108 episodes x 4 phases x 4 prompt types = 1728 VQA queries.
"""
import csv, json, subprocess
from pathlib import Path
import torch
from PIL import Image
from transformers import AutoProcessor, SmolVLMForConditionalGeneration
from huggingface_hub import hf_hub_download
import fastparquet as fp

MODEL_ID = "robot-learning-group47/smolvlm2-500m-vqa-lora-merged"
DATASET  = "robot-learning-group47/eval2_fixed_bowl_fixed_banana_108"
FRAME_DIR = Path("data/midscene_108/eval_frames")
FRAME_DIR.mkdir(parents=True, exist_ok=True)

FPS = 15
PHASE_FRAC = {"initial":0.05, "post_grasp":0.40, "bowl_visible":0.65, "branch":0.80}
COLOR_NAMES = {"b":"blue","g":"green","r":"red"}
ORDINAL = {1:"1st",2:"2nd",3:"3rd"}

print("Downloading dataset files...")
video_path = hf_hub_download(DATASET, "videos/observation.images.front/chunk-000/file-000.mp4", repo_type="dataset")
combo_csv  = hf_hub_download(DATASET, "combo_episode_map_eval2_fixed_bowl_fixed_banana_108.csv", repo_type="dataset")
data_pq    = hf_hub_download(DATASET, "data/chunk-000/file-000.parquet", repo_type="dataset")

df = fp.ParquetFile(data_pq).to_pandas(["episode_index","frame_index"])
ep_lengths = df.groupby("episode_index").size()
ep_starts  = ep_lengths.cumsum().shift(1,fill_value=0)
ep_info = {int(ep):{"start":int(ep_starts[ep]),"length":int(ep_lengths[ep])} for ep in ep_lengths.index}
print(f"Episodes: {len(ep_info)}")

with open(combo_csv) as f:
    combos = list(csv.DictReader(f))

def build_prompts(layout, target_color):
    c   = [COLOR_NAMES[x] for x in layout]
    idx = c.index(target_color)
    slot = ["left","middle","right"][idx]
    pos  = idx+1
    others = [c[i] for i in range(3) if i!=idx]
    ref_color = others[0]
    side = "left" if idx < c.index(ref_color) else "right"
    return [
        ("color",   f"Put the banana in the {target_color} colored bowl.", slot),
        ("ordinal", f"Put the banana into the {ORDINAL[pos]} bowl from the left from the robot perspective.", slot),
        ("relative",f"Put the banana into the bowl on the {side} of the {ref_color} bowl from the robot perspective.", slot),
        ("negation",f"Put the banana into the bowl that is not {others[0]} and not {others[1]}.", slot),
    ]

SYSTEM = (
    "You are looking at a robot arm camera view. "
    "There are three bowls: one on the left, one in the centre, one on the right. "
    "Which bowl does the instruction refer to? "
    "Reply ONLY with one of: 'left', 'centre', 'right'."
)

def prompt_to_question(p):
    q = p.replace("Put the banana in the ","Where should the robot put the banana — in the ")
    q = q.replace("Put the banana into the ","Where should the robot put the banana — into the ")
    return q.rstrip(".")+"? Answer with only 'left', 'centre', or 'right'."

def parse_answer(raw):
    r = raw.lower()
    if "left"  in r: return "left"
    if "centre" in r or "center" in r or "middle" in r: return "middle"
    if "right" in r: return "right"
    return None

def extract_frame(abs_frame, out_path):
    if out_path.exists(): return True
    t = abs_frame/FPS
    subprocess.run(["ffmpeg","-ss",f"{t:.3f}","-i",str(video_path),
                    "-frames:v","1","-q:v","2",str(out_path),"-y"],capture_output=True)
    return out_path.exists()

device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Device: {device}")
processor = AutoProcessor.from_pretrained(MODEL_ID)
model = SmolVLMForConditionalGeneration.from_pretrained(
    MODEL_ID, torch_dtype=torch.bfloat16, _attn_implementation="eager").to(device)
model.eval()
print("Model ready.\n")

results = []

for c in combos:
    combo  = c["combo"]; layout = c["bowl_order_left_to_right"]
    target = c["target_bowl_color"]
    slot   = ["left","middle","right"][layout.index(target[0])]
    prompts = build_prompts(layout, target)
    print(f"\n{combo}  layout={layout}  target={target}  slot={slot}")

    for ep_idx in range(int(c["episode_start"]), int(c["episode_end"])+1):
        info = ep_info[ep_idx]
        for phase, frac in PHASE_FRAC.items():
            abs_frame = info["start"] + int(info["length"]*frac)
            fpath = FRAME_DIR / f"{combo}_ep{ep_idx:03d}_{phase}.jpg"
            if not extract_frame(abs_frame, fpath):
                print(f"  WARN: failed {fpath.name}"); continue
            image = Image.open(fpath).convert("RGB")

            for ptype, prompt_text, gt_slot in prompts:
                msgs = [{"role":"user","content":[{"type":"image"},{"type":"text","text":f"{SYSTEM}\n\n{prompt_to_question(prompt_text)}"}]}]
                text   = processor.apply_chat_template(msgs, add_generation_prompt=True)
                inputs = processor(text=text, images=[image], return_tensors="pt")
                inputs = {k:v.to(device) for k,v in inputs.items()}
                with torch.inference_mode():
                    out = model.generate(**inputs, max_new_tokens=10, do_sample=False)
                n   = inputs["input_ids"].shape[1]
                raw = processor.decode(out[0][n:], skip_special_tokens=True).strip()
                pred    = parse_answer(raw)
                correct = (pred == gt_slot)
                results.append({"combo":combo,"layout":layout,"target":target,
                    "gt_slot":gt_slot,"phase":phase,"prompt_type":ptype,
                    "pred":pred,"raw":raw,"correct":correct,"ep_idx":ep_idx})
                mark = "v" if correct else "X"
                print(f"  [{mark}] ep{ep_idx:03d} {phase:13s} {ptype:8s} gt={gt_slot:6s} pred={str(pred):6s} | {raw}")

total   = len(results)
n_right = sum(r["correct"] for r in results)
print(f"\n{'='*70}")
print(f"  OVERALL: {n_right}/{total} = {n_right/total*100:.1f}%")
print(f"{'='*70}")

def acc(sub): return sum(r["correct"] for r in sub)/len(sub) if sub else float("nan")

print("\n  By PHASE:")
for ph in ["initial","post_grasp","bowl_visible","branch"]:
    sub=[r for r in results if r["phase"]==ph]
    print(f"    {ph:15s}: {sum(r['correct'] for r in sub):4d}/{len(sub):4d} = {acc(sub)*100:.1f}%")

print("\n  By TARGET SLOT:")
for sl in ["left","middle","right"]:
    sub=[r for r in results if r["gt_slot"]==sl]
    print(f"    {sl:8s}: {sum(r['correct'] for r in sub):4d}/{len(sub):4d} = {acc(sub)*100:.1f}%")

print("\n  By PROMPT TYPE:")
for pt in ["color","ordinal","relative","negation"]:
    sub=[r for r in results if r["prompt_type"]==pt]
    print(f"    {pt:10s}: {sum(r['correct'] for r in sub):4d}/{len(sub):4d} = {acc(sub)*100:.1f}%")

print("\n  PHASE x SLOT:")
print(f"    {'':15s} {'left':>8s} {'middle':>8s} {'right':>8s} {'ALL':>8s}")
for ph in ["initial","post_grasp","bowl_visible","branch"]:
    row=f"    {ph:15s}"
    for sl in ["left","middle","right","ALL"]:
        sub=[r for r in results if r["phase"]==ph and (sl=="ALL" or r["gt_slot"]==sl)]
        row+=f"  {acc(sub)*100:6.1f}%" if sub else "     N/A"
    print(row)

print("\n  PHASE x PROMPT TYPE:")
print(f"    {'':15s} {'color':>8s} {'ordinal':>8s} {'relative':>9s} {'negation':>9s}")
for ph in ["initial","post_grasp","bowl_visible","branch"]:
    row=f"    {ph:15s}"
    for pt in ["color","ordinal","relative","negation"]:
        sub=[r for r in results if r["phase"]==ph and r["prompt_type"]==pt]
        row+=f"  {acc(sub)*100:6.1f}%" if sub else "     N/A"
    print(row)

out_path = Path("data/midscene_108/accuracy_results.json")
out_path.parent.mkdir(parents=True,exist_ok=True)
with open(out_path,"w") as f: json.dump(results,f,indent=2)
print(f"\nResults -> {out_path}")
