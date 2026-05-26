"""
Obama  → LFW via HuggingFace datasets (1783 labeled images)
Swift  → DuckDuckGo with long delays + face verification
"""

import hashlib
import io
import time
from pathlib import Path

import cv2
import numpy as np
import requests
from PIL import Image

OUT = Path("./data/celeb_faces")
OBAMA_DIR = OUT / "Barack Obama"
SWIFT_DIR = OUT / "Taylor Swift"
TARGET_OBAMA = 50
TARGET_SWIFT = 50

CASCADE = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)


def has_face(img: Image.Image) -> bool:
    arr = np.array(img.convert("RGB"))
    gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
    return len(CASCADE.detectMultiScale(gray, 1.1, 5, minSize=(50, 50))) > 0


def save_img(img: Image.Image, dest: Path, name: str) -> bool:
    try:
        img = img.convert("RGB")
        if img.size[0] < 80 or img.size[1] < 80:
            return False
        dest.mkdir(parents=True, exist_ok=True)
        img.save(dest / name, "JPEG", quality=90)
        return True
    except Exception:
        return False


def fetch(url: str) -> Image.Image | None:
    try:
        r = requests.get(url, timeout=8,
                         headers={"User-Agent": "Mozilla/5.0"})
        return Image.open(io.BytesIO(r.content)) if r.status_code == 200 else None
    except Exception:
        return None


def count(d: Path) -> int:
    return len(list(d.glob("*.jpg")) + list(d.glob("*.png"))) if d.exists() else 0


# ═══════════════════════════════════════════════════════════════
# OBAMA — LFW via HuggingFace
# ═══════════════════════════════════════════════════════════════
def download_obama_hf():
    print("\n=== Barack Obama via LFW (HuggingFace) ===")
    existing = count(OBAMA_DIR)
    print(f"  Currently: {existing} images")

    # Try multiple known HF mirrors of LFW
    from datasets import load_dataset

    print("  Loading logasja/lfw from HuggingFace...")
    ds = load_dataset("logasja/lfw", split="train")
    label_col, img_col = "label", "image"
    print(f"  Loaded: {len(ds)} samples")

    if ds is None:
        print("  All HF LFW mirrors failed.")
        return

    # Find Obama indices
    obama_indices = []
    for i, sample in enumerate(ds):
        raw = str(sample.get(label_col, "")).replace("_", " ").strip()
        if "obama" in raw.lower():
            obama_indices.append(i)

    print(f"  Found {len(obama_indices)} Obama images in LFW")

    saved = 0
    for idx in obama_indices:
        if existing + saved >= TARGET_OBAMA:
            break
        try:
            img = ds[idx][img_col]
            if isinstance(img, Image.Image):
                if save_img(img, OBAMA_DIR, f"lfw_{idx:06d}.jpg"):
                    saved += 1
        except Exception:
            continue

    print(f"  Saved {saved} Obama images. Total: {count(OBAMA_DIR)}")


# ═══════════════════════════════════════════════════════════════
# SWIFT — DuckDuckGo with heavy delays
# ═══════════════════════════════════════════════════════════════
SWIFT_QUERIES = [
    "Taylor Swift face portrait photo",
    "Taylor Swift headshot singer",
    "Taylor Swift close up face 2023",
    "Taylor Swift Grammy Awards face",
    "Taylor Swift Eras Tour portrait",
    "Taylor Swift interview close up",
    "Taylor Swift music video face",
    "Taylor Swift blonde hair portrait",
    "Taylor Swift Met Gala face",
    "Taylor Swift official photo shoot",
    "Taylor Swift VMA face close",
    "Taylor Swift Midnights album portrait",
]


def download_swift_ddg():
    print("\n=== Taylor Swift via DuckDuckGo ===")
    existing = count(SWIFT_DIR)
    print(f"  Currently: {existing}, target: {TARGET_SWIFT}")

    if existing >= TARGET_SWIFT:
        print("  Already at target.")
        return

    from ddgs import DDGS

    saved = 0
    with DDGS() as ddgs:
        for i, query in enumerate(SWIFT_QUERIES):
            if existing + saved >= TARGET_SWIFT:
                break
            print(f"  [{i+1}/{len(SWIFT_QUERIES)}] {query}")

            # Long sleep before each query to avoid rate limit
            if i > 0:
                print(f"    sleeping 20s...")
                time.sleep(20)

            try:
                results = list(ddgs.images(query, max_results=25, type_image="photo"))
            except Exception as e:
                print(f"    Rate limited ({e}), sleeping 60s")
                time.sleep(60)
                try:
                    results = list(ddgs.images(query, max_results=25))
                except Exception:
                    continue

            batch = 0
            for r in results:
                if existing + saved >= TARGET_SWIFT:
                    break
                url = r.get("image", "")
                if not url:
                    continue
                img = fetch(url)
                if img is None:
                    continue
                if not has_face(img):
                    continue
                h = hashlib.md5(url.encode()).hexdigest()[:10]
                if save_img(img, SWIFT_DIR, f"ddg_{saved:04d}_{h}.jpg"):
                    saved += 1
                    batch += 1
                time.sleep(0.3)
            print(f"    +{batch} images (total saved: {saved})")

    print(f"  Taylor Swift total: {count(SWIFT_DIR)}")


# ═══════════════════════════════════════════════════════════════
if __name__ == "__main__":
    download_obama_hf()
    download_swift_ddg()

    print("\n=== FINAL ===")
    for name, d in [("Barack Obama", OBAMA_DIR), ("Taylor Swift", SWIFT_DIR)]:
        n = count(d)
        print(f"  {'OK' if n >= 50 else 'LOW'} {name}: {n} images")
    print("ALL_DONE")
