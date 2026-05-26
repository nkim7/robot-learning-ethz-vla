"""
Download Taylor Swift images from Wikimedia Commons categories:
  - Category:Taylor_Swift_in_2024
  - Category:Taylor_Swift_in_2023
Uses the Commons API — no scraping, properly licensed images.
Filters: face must be detectable and cover >= 10% of image area.
"""

import io
import time
from pathlib import Path

import requests
from PIL import Image
import torch
from facenet_pytorch import MTCNN
import numpy as np

OUT_DIR = Path("./data/celeb_faces/Taylor Swift")
REJECT_DIR = Path("./data/rejected/Taylor Swift_wikimedia")
OUT_DIR.mkdir(parents=True, exist_ok=True)
REJECT_DIR.mkdir(parents=True, exist_ok=True)

MIN_FACE_FRAC = 0.08   # face must be >= 8% of image area
A5_W, A5_H = 448, 634

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
mtcnn = MTCNN(keep_all=False, device=DEVICE, post_process=False,
              min_face_size=40, thresholds=[0.6, 0.7, 0.8])

CATEGORIES = [
    "Taylor Swift in 2024",
    "Taylor Swift in 2023",
    "Taylor Swift in 2022",
    "Taylor Swift in 2021",
    "Taylor Swift in 2020",
    "Taylor Swift in 2019",
    "Taylor Swift in 2018",
    "Taylor Swift in 2017",
    "Photographs of Taylor Swift",
]

API = "https://commons.wikimedia.org/w/api.php"
HEADERS = {"User-Agent": "robot-learning-project/1.0 (educational)"}


def get_category_files(category: str) -> list[dict]:
    """Return list of {title, pageid} for all files in a Commons category."""
    files = []
    params = {
        "action": "query",
        "list": "categorymembers",
        "cmtitle": f"Category:{category}",
        "cmtype": "file",
        "cmlimit": "500",
        "format": "json",
    }
    while True:
        r = requests.get(API, params=params, headers=HEADERS, timeout=15)
        data = r.json()
        files.extend(data["query"]["categorymembers"])
        if "continue" not in data:
            break
        params["cmcontinue"] = data["continue"]["cmcontinue"]
    return files


def get_image_url(title: str) -> str | None:
    """Resolve a Commons file title to a direct image URL."""
    params = {
        "action": "query",
        "titles": title,
        "prop": "imageinfo",
        "iiprop": "url|mime",
        "format": "json",
    }
    for attempt in range(3):
        try:
            r = requests.get(API, params=params, headers=HEADERS, timeout=15)
            if not r.text.strip():
                time.sleep(2 ** attempt)
                continue
            pages = r.json()["query"]["pages"]
            for page in pages.values():
                ii = page.get("imageinfo", [{}])[0]
                mime = ii.get("mime", "")
                if mime.startswith("image/"):
                    return ii.get("url")
            return None
        except Exception:
            time.sleep(2 ** attempt)
    return None


def crop_a5(img: Image.Image, box) -> Image.Image:
    W, H = img.size
    x1, y1, x2, y2 = [float(v) for v in box]
    fw, fh = x2 - x1, y2 - y1
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    pad = max(fw, fh) * 0.5
    crop_w = fw + 2 * pad
    crop_h = crop_w * (A5_H / A5_W)
    if crop_h > H * 0.95:
        crop_h = H * 0.95
        crop_w = crop_h * (A5_W / A5_H)
    x0 = max(0, min(cx - crop_w / 2, W - crop_w))
    y0 = max(0, min(cy - crop_h / 2, H - crop_h))
    return img.crop((x0, y0, x0 + crop_w, y0 + crop_h)).resize((A5_W, A5_H), Image.LANCZOS)


saved = skipped = no_face = 0

for cat in CATEGORIES:
    print(f"\n=== {cat} ===")
    files = get_category_files(cat)
    print(f"  {len(files)} files in category")

    for f in files:
        title = f["title"]
        # Skip non-photo files
        low = title.lower()
        if not any(low.endswith(ext) for ext in (".jpg", ".jpeg", ".png")):
            continue

        # Derive safe local filename
        safe = title.replace("File:", "").replace(" ", "_").replace("/", "_")
        out_path = OUT_DIR / f"wiki_{safe}"
        if out_path.exists():
            skipped += 1
            continue

        url = get_image_url(title)
        if not url:
            continue

        try:
            r = requests.get(url, headers=HEADERS, timeout=20)
            img = Image.open(io.BytesIO(r.content)).convert("RGB")
        except Exception:
            continue

        W, H = img.size
        boxes, probs = mtcnn.detect(img)

        if boxes is None or len(boxes) == 0:
            print(f"  [no_face ] {safe}")
            no_face += 1
            continue

        best = int(np.argmax(probs))
        box = boxes[best]
        x1, y1, x2, y2 = [float(v) for v in box]
        face_frac = ((x2-x1) * (y2-y1)) / (W * H)

        if face_frac < MIN_FACE_FRAC:
            print(f"  [small   ] {safe}  face={face_frac:.1%}")
            img.save(REJECT_DIR / safe, "JPEG", quality=85)
            continue

        cropped = crop_a5(img, box)
        cropped.save(out_path, "JPEG", quality=92)
        print(f"  [keep    ] {safe}  face={face_frac:.1%}")
        saved += 1
        time.sleep(0.2)

print(f"\n=== Done ===")
print(f"  Saved:   {saved}")
print(f"  No face: {no_face}")
print(f"  Skipped (already exist): {skipped}")
existing = len(list(OUT_DIR.glob("*.jpg"))) + len(list(OUT_DIR.glob("*.png")))
print(f"  Total in dir: {existing}")
