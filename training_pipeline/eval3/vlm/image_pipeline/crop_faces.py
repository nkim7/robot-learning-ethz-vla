"""
Re-crop all LeCun images using MTCNN face detection.
- Rejects images where face is too small (person far away on stage etc.)
- Crops tightly around face with padding, then resizes to A5 ratio
- Moves rejects to ./data/rejected/
"""

import shutil
from pathlib import Path

import torch
import numpy as np
from facenet_pytorch import MTCNN
from PIL import Image

LECUN_DIR  = Path("./data/celeb_faces/Yann LeCun")
REJECT_DIR = Path("./data/rejected/Yann LeCun_crop")
REJECT_DIR.mkdir(parents=True, exist_ok=True)

A5_W, A5_H = 448, 634          # output size (A5 portrait ratio)
MIN_FACE_FRAC = 0.05            # face must be >= 5% of image area
PADDING = 0.5                   # padding around face as fraction of face size

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
mtcnn = MTCNN(keep_all=False, device=DEVICE, post_process=False,
              min_face_size=20, thresholds=[0.6, 0.7, 0.7])


def crop_a5(img: Image.Image, box) -> Image.Image:
    """Crop A5 portrait centered on detected face box with padding."""
    W, H = img.size
    x1, y1, x2, y2 = [float(v) for v in box]
    fw, fh = x2 - x1, y2 - y1
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2

    # Padding around face
    pad = max(fw, fh) * PADDING
    crop_w = fw + 2 * pad
    crop_h = crop_w * (A5_H / A5_W)

    # If crop_h too tall, expand width instead
    if crop_h > H * 0.95:
        crop_h = H * 0.95
        crop_w = crop_h * (A5_W / A5_H)

    # Center on face, clamp to image bounds
    x0 = max(0, min(cx - crop_w / 2, W - crop_w))
    y0 = max(0, min(cy - crop_h / 2, H - crop_h))
    x1c = min(W, x0 + crop_w)
    y1c = min(H, y0 + crop_h)

    cropped = img.crop((x0, y0, x1c, y1c))
    return cropped.resize((A5_W, A5_H), Image.LANCZOS)


all_imgs = sorted(LECUN_DIR.glob("*.jpg")) + sorted(LECUN_DIR.glob("*.png"))
print(f"Processing {len(all_imgs)} images...\n")

kept = small_face = no_face = 0

for p in all_imgs:
    try:
        img = Image.open(p).convert("RGB")
    except Exception:
        p.unlink()
        continue

    W, H = img.size
    img_area = W * H

    boxes, probs = mtcnn.detect(img)

    if boxes is None or len(boxes) == 0:
        print(f"  [no_face ] {p.name}")
        shutil.move(str(p), REJECT_DIR / p.name)
        no_face += 1
        continue

    # Pick highest-confidence face
    best = int(np.argmax(probs))
    box = boxes[best]
    x1, y1, x2, y2 = [float(v) for v in box]
    face_area = (x2 - x1) * (y2 - y1)
    face_frac = face_area / img_area

    if face_frac < MIN_FACE_FRAC:
        print(f"  [too_small] {p.name}  face={face_frac:.2%} of image")
        shutil.move(str(p), REJECT_DIR / p.name)
        small_face += 1
        continue

    cropped = crop_a5(img, box)
    cropped.save(p, "JPEG", quality=92)
    print(f"  [keep    ] {p.name}  face={face_frac:.2%}")
    kept += 1

print(f"\n=== Done ===")
print(f"  Kept:       {kept}")
print(f"  No face:    {no_face}")
print(f"  Face too small (stage/far away): {small_face}")
print(f"  Total remaining: {kept}")
print("CROP_DONE")
