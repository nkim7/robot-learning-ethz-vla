"""
Zero-shot audit: can the SmolVLM2 backbone recognize Taylor Swift,
Barack Obama, and Yann LeCun from single photos, and localize them
in a 3-portrait scene?

SmolVLM2-2.2B-Instruct is the direct successor of SmolVLM-Instruct
and the likely backbone for SmolVLA on transformers >=5.x.
"""

import os
import json
import itertools
from pathlib import Path
from PIL import Image
import torch
from transformers import AutoProcessor, SmolVLMForConditionalGeneration

MODEL_ID = os.environ.get("MODEL_ID", "HuggingFaceTB/SmolVLM2-2.2B-Instruct")
IMAGE_DIR = Path(os.environ.get("IMAGE_DIR", "./clelb_pic"))
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.bfloat16 if DEVICE == "cuda" else torch.float32

print(f"Device : {DEVICE}")
print(f"Model  : {MODEL_ID}")
print(f"Images : {IMAGE_DIR}")
print("Loading model (may download ~4GB)...")

processor = AutoProcessor.from_pretrained(MODEL_ID)
model = SmolVLMForConditionalGeneration.from_pretrained(
    MODEL_ID,
    torch_dtype=DTYPE,
    device_map=DEVICE,
)
model.eval()
print(f"Model loaded. Params: {sum(p.numel() for p in model.parameters())/1e9:.2f}B\n")


def ask(images: list, question: str, max_new_tokens: int = 64) -> str:
    messages = [
        {
            "role": "user",
            "content": [{"type": "image"} for _ in images] + [{"type": "text", "text": question}],
        }
    ]
    prompt = processor.apply_chat_template(messages, add_generation_prompt=True)
    inputs = processor(text=prompt, images=images, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        ids = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    generated = ids[0][inputs["input_ids"].shape[1]:]
    return processor.decode(generated, skip_special_tokens=True).strip()


# ── TEST 1: Single-portrait recognition ────────────────────────────────────────
print("=" * 60)
print("TEST 1: Single-portrait recognition")
print("=" * 60)

celebrities = {
    "Taylor Swift": sorted(IMAGE_DIR.glob("taylor_*.png")),
    "Barack Obama": sorted(IMAGE_DIR.glob("obama_*.png")),
    "Yann LeCun":   sorted(IMAGE_DIR.glob("yann_*.png")),
}

results_t1 = {}
for celeb, paths in celebrities.items():
    results_t1[celeb] = []
    for p in paths:
        img = Image.open(p).convert("RGB")
        answer = ask([img], "Who is the person in this image? Give the full name.")
        last_name = celeb.split()[-1].lower()
        correct = last_name in answer.lower()
        tag = "✓" if correct else "✗"
        print(f"  [{tag}] {p.name}: {answer}")
        results_t1[celeb].append({"file": p.name, "answer": answer, "correct": correct})
    acc = sum(r["correct"] for r in results_t1[celeb]) / len(results_t1[celeb])
    print(f"  → {celeb} accuracy: {acc:.0%}\n")


# ── TEST 2: All 3 celebrities together — can it tell them apart? ───────────────
print("=" * 60)
print("TEST 2: All 3 celebrities in one prompt (confusion test)")
print("=" * 60)

# Pick one image of each celebrity, ask in one call
imgs_all = [
    Image.open(celebrities["Taylor Swift"][0]).convert("RGB"),
    Image.open(celebrities["Barack Obama"][0]).convert("RGB"),
    Image.open(celebrities["Yann LeCun"][0]).convert("RGB"),
]
q_all = (
    "I am showing you three separate photos (image 1, image 2, image 3). "
    "For each one, tell me the person's full name. "
    "Format your answer as:\n"
    "Image 1: [name]\nImage 2: [name]\nImage 3: [name]"
)
answer_all = ask(imgs_all, q_all, max_new_tokens=80)
print(f"  {answer_all}\n")
print(f"  (Expected: Image 1=Taylor Swift, Image 2=Barack Obama, Image 3=Yann LeCun)\n")


# ── TEST 3: Scene-level grounding (composite 3-wide image) ────────────────────
print("=" * 60)
print("TEST 3: Scene-level grounding (3-portrait side-by-side composite)")
print("=" * 60)

THUMB = (224, 224)

def make_scene(p_left, p_mid, p_right):
    imgs = [Image.open(p).convert("RGB").resize(THUMB) for p in [p_left, p_mid, p_right]]
    w = THUMB[0] * 3
    scene = Image.new("RGB", (w, THUMB[1]))
    for i, im in enumerate(imgs):
        scene.paste(im, (i * THUMB[0], 0))
    return scene

celeb_keys = list(celebrities.keys())
arrangements = list(itertools.permutations(celeb_keys))

results_t3 = []
for arrangement in arrangements:
    left, mid, right = arrangement
    scene = make_scene(celebrities[left][0], celebrities[mid][0], celebrities[right][0])
    for target in celeb_keys:
        expected_pos = arrangement.index(target)
        expected_str = ["left", "middle", "right"][expected_pos]
        q = (
            f"This image shows three celebrity portraits placed side by side. "
            f"The left third contains one person, the middle third another, the right third another. "
            f"Which position is {target} in — left, middle, or right? "
            f"Answer with just one word."
        )
        answer = ask([scene], q, max_new_tokens=16)
        correct = expected_str in answer.lower()
        tag = "✓" if correct else "✗"
        print(f"  [{tag}] L={left[:6]} M={mid[:6]} R={right[:6]} | "
              f"target={target[:6]} expected={expected_str} got={answer}")
        results_t3.append({
            "arrangement": list(arrangement),
            "target": target,
            "expected": expected_str,
            "answer": answer,
            "correct": correct,
        })

acc3 = sum(r["correct"] for r in results_t3) / len(results_t3)
print(f"\n  → Scene grounding (composite) accuracy: {acc3:.0%}")


# ── TEST 4: Scene grounding with separate image tokens ────────────────────────
print("\n" + "=" * 60)
print("TEST 4: Scene grounding with 3 separate images as input")
print("=" * 60)

results_t4 = []
for arrangement in arrangements:
    left, mid, right = arrangement
    imgs = [
        Image.open(celebrities[left][0]).convert("RGB"),
        Image.open(celebrities[mid][0]).convert("RGB"),
        Image.open(celebrities[right][0]).convert("RGB"),
    ]
    for target in celeb_keys:
        expected_pos = arrangement.index(target)
        expected_str = ["left", "middle", "right"][expected_pos]
        q = (
            f"I am showing you three celebrity photos. "
            f"Photo 1 is on the LEFT, Photo 2 is in the MIDDLE, Photo 3 is on the RIGHT. "
            f"Which position is {target} in? Answer with just one word: left, middle, or right."
        )
        answer = ask(imgs, q, max_new_tokens=16)
        correct = expected_str in answer.lower()
        tag = "✓" if correct else "✗"
        print(f"  [{tag}] L={left[:6]} M={mid[:6]} R={right[:6]} | "
              f"target={target[:6]} expected={expected_str} got={answer}")
        results_t4.append({
            "arrangement": list(arrangement),
            "target": target,
            "expected": expected_str,
            "answer": answer,
            "correct": correct,
        })

acc4 = sum(r["correct"] for r in results_t4) / len(results_t4)
print(f"\n  → Scene grounding (multi-image) accuracy: {acc4:.0%}")


# ── Summary ───────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("SUMMARY")
print("=" * 60)
for celeb in celeb_keys:
    acc = sum(r["correct"] for r in results_t1[celeb]) / len(results_t1[celeb])
    print(f"  Single-portrait  {celeb:<22}: {acc:.0%}")
print(f"  Scene grounding composite       : {acc3:.0%}")
print(f"  Scene grounding multi-image     : {acc4:.0%}")
print()
print("Interpretation:")
print("  Single >80% → recognition is solid, Phase 1 training minimal")
print("  Single 40-80% → Phase 1 training needed for that celebrity")
print("  Single <40% → Phase 1 critical, scrape more diverse images")
print("  Scene  <60% → Phase 2 training essential regardless of Phase 1")

out = {
    "model": MODEL_ID,
    "single_portrait": results_t1,
    "confusion_test": answer_all,
    "scene_grounding_composite": results_t3,
    "scene_grounding_multiimage": results_t4,
}
with open("eval3_zeroshot_results.json", "w") as f:
    json.dump(out, f, indent=2, default=str)
print("\nResults saved to eval3_zeroshot_results.json")
