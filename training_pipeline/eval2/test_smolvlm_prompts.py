"""
Test whether HuggingFaceTB/SmolVLM2-500M-Video-Instruct can correctly
answer compositional bowl questions from robot camera images.
"""

import os
from pathlib import Path
from PIL import Image
import torch
from transformers import AutoProcessor, AutoModelForImageTextToText

MODEL = "HuggingFaceTB/SmolVLM2-500M-Video-Instruct"
REPO_ROOT = Path(__file__).resolve().parent.parent.parent
IMG_DIR = Path(os.environ.get("IMG_DIR", REPO_ROOT / "permutation_init_position"))

# Ground truth: arrangement name → (left, middle, right) colors
ARRANGEMENTS = {
    "bgr": ("blue", "green", "red"),
    "brg": ("blue", "red",   "green"),
    "gbr": ("green","blue",  "red"),
    "grb": ("green","red",   "blue"),
    "rbg": ("red",  "blue",  "green"),
    "rgb": ("red",  "green", "blue"),
}

# Test cases: (question, expected_color)
def make_tests(left, mid, right):
    others_of = {
        left:  [mid, right],
        mid:   [left, right],
        right: [left, mid],
    }
    return [
        # Direct color
        (f"Which bowl should I put the banana in if the instruction is: Put the banana in the {left} colored bowl? Answer with just the color.", left),
        (f"Which bowl should I put the banana in if the instruction is: Put the banana in the {mid} colored bowl? Answer with just the color.", mid),
        (f"Which bowl should I put the banana in if the instruction is: Put the banana in the {right} colored bowl? Answer with just the color.", right),
        # Ordinal left→right
        ("Which bowl is 1st from the left? Answer with just the color.", left),
        ("Which bowl is 2nd from the left? Answer with just the color.", mid),
        ("Which bowl is 3rd from the left? Answer with just the color.", right),
        # Ordinal right→left
        ("Which bowl is 1st from the right? Answer with just the color.", right),
        ("Which bowl is 2nd from the right? Answer with just the color.", mid),
        ("Which bowl is 3rd from the right? Answer with just the color.", left),
        # Middle
        ("Which bowl is in the middle? Answer with just the color.", mid),
        # Relative
        (f"Which bowl is to the right of the {left} bowl? Answer with just the color.", mid),
        (f"Which bowl is to the left of the {right} bowl? Answer with just the color.", mid),
        # Negation
        (f"Which bowl is not {others_of[right][0]} and not {others_of[right][1]}? Answer with just the color.", right),
        (f"Which bowl is not {others_of[left][0]} and not {others_of[left][1]}? Answer with just the color.", left),
    ]

print(f"Loading {MODEL}...")
processor = AutoProcessor.from_pretrained(MODEL)
model = AutoModelForImageTextToText.from_pretrained(
    MODEL,
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    device_map="auto",
)
model.eval()
print("Model loaded.\n")

def ask(image: Image.Image, question: str) -> str:
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text",  "text": question},
            ],
        }
    ]
    inputs = processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_tensors="pt",
        return_dict=True,
    ).to(model.device, dtype=torch.float16 if torch.cuda.is_available() else torch.float32)

    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=20, do_sample=False)
    # decode only new tokens
    generated = out[0][inputs["input_ids"].shape[1]:]
    return processor.decode(generated, skip_special_tokens=True).strip().lower()

results = []
for arr_name, (left, mid, right) in ARRANGEMENTS.items():
    img_path = IMG_DIR / f"{arr_name}.jpeg"
    if not img_path.exists():
        print(f"[SKIP] {arr_name}: no image found")
        continue

    image = Image.open(img_path).convert("RGB")
    tests = make_tests(left, mid, right)

    correct = 0
    print(f"\n{'='*60}")
    print(f"Arrangement: {arr_name}  (left={left}, mid={mid}, right={right})")
    print(f"{'='*60}")

    for question, expected in tests:
        answer = ask(image, question)
        ok = expected in answer
        correct += ok
        status = "✓" if ok else "✗"
        print(f"  {status} Q: {question[:70]}")
        if not ok:
            print(f"       Expected: {expected} | Got: {answer}")

    pct = 100 * correct / len(tests)
    print(f"  → {correct}/{len(tests)} correct ({pct:.0f}%)")
    results.append((arr_name, correct, len(tests)))

print("\n" + "="*60)
print("SUMMARY")
print("="*60)
total_c = sum(c for _, c, _ in results)
total_t = sum(t for _, _, t in results)
for arr, c, t in results:
    print(f"  {arr}: {c}/{t} ({100*c//t}%)")
print(f"\n  OVERALL: {total_c}/{total_t} ({100*total_c//total_t}%)")
