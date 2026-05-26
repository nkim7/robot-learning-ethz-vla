"""
Synthesize Phase 2 scene images by compositing celebrity portraits
into real robot-setup backgrounds.

Creates all 6 permutations × 5 backgrounds × N augmentations.
Output: data/scene_synthetic/
"""

import itertools
import random
from pathlib import Path

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter

SCENE_DIR  = Path("./data/scene_setup")
FACE_DIR   = Path("./data/celeb_faces")
OUT_DIR    = Path("./data/scene_synthetic")
OUT_DIR.mkdir(parents=True, exist_ok=True)

CELEBS = ["Taylor Swift", "Barack Obama", "Yann LeCun"]
N_AUG  = 6   # augmented copies per permutation × background

# Portrait bounding boxes in the 256×256 scene images (x1,y1,x2,y2)
# Derived from MTCNN face detection on the 5 reference images
PORTRAIT_BOXES = {
    "left":   (2,  22, 75,  118),
    "center": (88,  0, 150,  82),
    "right":  (178, 22, 248, 118),
}

ANNOTATIONS = [
    ("scene_L-Swift_C-LeCun_R-Obama_1.jpg", "Taylor Swift", "Yann LeCun",   "Barack Obama"),
    ("scene_L-Swift_C-LeCun_R-Obama_2.jpg", "Taylor Swift", "Yann LeCun",   "Barack Obama"),
    ("scene_L-LeCun_C-Swift_R-Obama.jpg",   "Yann LeCun",   "Taylor Swift",  "Barack Obama"),
    ("scene_L-LeCun_C-Obama_R-Swift_1.jpg", "Yann LeCun",   "Barack Obama",  "Taylor Swift"),
    ("scene_L-LeCun_C-Obama_R-Swift_2.jpg", "Yann LeCun",   "Barack Obama",  "Taylor Swift"),
]

# Pre-load face images per celeb
face_pool = {}
for c in CELEBS:
    imgs = list((FACE_DIR / c).glob("*.jpg")) + list((FACE_DIR / c).glob("*.png"))
    random.shuffle(imgs)
    face_pool[c] = imgs
    print(f"  {c}: {len(imgs)} face images")


def add_portrait_border(img: Image.Image, border_px: int = 4) -> Image.Image:
    """Add white border like a printed photo."""
    W, H = img.size
    bordered = Image.new("RGB", (W + 2*border_px, H + 2*border_px), (240, 238, 232))
    bordered.paste(img, (border_px, border_px))
    return bordered


def augment(img: Image.Image, rng: random.Random) -> Image.Image:
    """Random brightness / contrast / saturation / slight blur."""
    img = ImageEnhance.Brightness(img).enhance(rng.uniform(0.75, 1.25))
    img = ImageEnhance.Contrast(img).enhance(rng.uniform(0.85, 1.15))
    img = ImageEnhance.Color(img).enhance(rng.uniform(0.8, 1.2))
    if rng.random() < 0.3:
        img = img.filter(ImageFilter.GaussianBlur(radius=rng.uniform(0.3, 0.8)))
    # tiny noise
    arr = np.array(img).astype(np.int16)
    noise = rng.randint(-8, 8)
    arr = np.clip(arr + np.random.randint(-8, 9, arr.shape, dtype=np.int16), 0, 255).astype(np.uint8)
    return Image.fromarray(arr)


def paste_portrait(bg: Image.Image, celeb: str, box, face_idx: int) -> Image.Image:
    x1, y1, x2, y2 = box
    w, h = x2 - x1, y2 - y1
    pool = face_pool[celeb]
    face = Image.open(pool[face_idx % len(pool)]).convert("RGB")
    face = face.resize((w, h), Image.LANCZOS)
    face = add_portrait_border(face, border_px=max(2, w // 20))
    face = face.resize((w, h), Image.LANCZOS)  # resize back after border
    result = bg.copy()
    result.paste(face, (x1, y1))
    return result


rng = random.Random(42)
saved = 0

# Load background images
backgrounds = []
for fname, *_ in ANNOTATIONS:
    p = SCENE_DIR / fname
    if p.exists():
        backgrounds.append(Image.open(p).convert("RGB"))

print(f"\nGenerating synthetic scenes...")
print(f"  {len(backgrounds)} backgrounds × 6 permutations × {N_AUG} augmentations")

# All 6 permutations of (left, center, right)
for perm in itertools.permutations(CELEBS):
    left, center, right = perm
    perm_name = f"L-{left.split()[0]}_C-{center.split()[0]}_R-{right.split()[0]}"

    face_counters = {c: rng.randint(0, 50) for c in CELEBS}

    for bg_idx, bg in enumerate(backgrounds):
        for aug_idx in range(N_AUG):
            img = bg.copy()
            img = paste_portrait(img, left,   PORTRAIT_BOXES["left"],   face_counters[left]   + aug_idx)
            img = paste_portrait(img, center, PORTRAIT_BOXES["center"], face_counters[center] + aug_idx)
            img = paste_portrait(img, right,  PORTRAIT_BOXES["right"],  face_counters[right]  + aug_idx)
            img = augment(img, rng)

            out_name = f"{perm_name}_bg{bg_idx}_aug{aug_idx}.jpg"
            img.save(OUT_DIR / out_name, "JPEG", quality=88)
            saved += 1

print(f"\nDone — {saved} synthetic scenes saved to {OUT_DIR}")
print(f"Permutations covered: all 6")
for perm in itertools.permutations(CELEBS):
    l, c, r = perm
    print(f"  L={l}, C={c}, R={r}")
