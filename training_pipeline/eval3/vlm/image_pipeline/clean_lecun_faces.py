"""
1. Verify each LeCun image contains the correct person using face embeddings
   (compares against reference images in clelb_pic/yann_*.png)
2. Crop verified images to A5 ratio (1:√2) centered on the detected face
3. Saves cleaned images back in-place, moves rejects to ./data/rejected/

pip install facenet-pytorch opencv-python pillow torch
"""

import shutil
from pathlib import Path

import cv2
import numpy as np
import torch
from facenet_pytorch import MTCNN, InceptionResnetV1
from PIL import Image

LECUN_DIR  = Path("./data/celeb_faces/Yann LeCun")
REF_DIR    = Path("./clelb_pic")
REJECT_DIR = Path("./data/rejected/Yann LeCun")
A5_RATIO   = 210 / 148        # height / width  ≈ 1.418
OUT_SIZE   = (448, 634)        # width × height at A5 ratio, good for training
SIM_THRESH = 0.70              # cosine similarity threshold (lower = stricter)
PADDING    = 0.6               # padding around face box as fraction of face size

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

mtcnn   = MTCNN(keep_all=False, device=DEVICE, post_process=False)
resnet  = InceptionResnetV1(pretrained="vggface2").eval().to(DEVICE)


def get_embedding(img: Image.Image) -> torch.Tensor | None:
    """Detect face, return 512-d normalized embedding or None."""
    face = mtcnn(img)
    if face is None:
        return None
    face = face.unsqueeze(0).to(DEVICE).float() / 255.0
    with torch.no_grad():
        emb = resnet(face)
    return torch.nn.functional.normalize(emb, dim=1)[0]


def cosine_sim(a: torch.Tensor, b: torch.Tensor) -> float:
    return float((a * b).sum())


def crop_a5_around_face(img: Image.Image) -> Image.Image | None:
    """Detect face, crop A5 portrait centered on it, return resized image."""
    arr = np.array(img.convert("RGB"))
    gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
    cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )
    faces = cascade.detectMultiScale(gray, 1.1, 5, minSize=(40, 40))

    W, H = img.size

    if len(faces) == 0:
        # No face found — crop center A5 from whole image
        cx, cy = W // 2, H // 2
    else:
        # Use largest face
        x, y, fw, fh = max(faces, key=lambda f: f[2] * f[3])
        cx = x + fw // 2
        cy = y + fh // 2

        # Expand box with padding
        pad = int(max(fw, fh) * PADDING)
        x0 = max(0, x - pad)
        y0 = max(0, y - pad)
        x1 = min(W, x + fw + pad)
        y1 = min(H, y + fh + pad)

        # Use padded box center
        cx = (x0 + x1) // 2
        cy = (y0 + y1) // 2

    # Compute A5 crop around center
    # Try fitting width first
    crop_w = min(W, H)
    crop_h = int(crop_w * A5_RATIO)
    if crop_h > H:
        crop_h = H
        crop_w = int(crop_h / A5_RATIO)

    x0 = max(0, min(cx - crop_w // 2, W - crop_w))
    y0 = max(0, min(cy - crop_h // 2, H - crop_h))
    x1 = x0 + crop_w
    y1 = y0 + crop_h

    cropped = img.crop((x0, y0, x1, y1))
    return cropped.resize(OUT_SIZE, Image.LANCZOS)


# ── Build reference embedding from provided yann_*.png images ──────────────────
print("Building reference embedding from provided LeCun images...")
ref_imgs = sorted(REF_DIR.glob("yann_*.png"))
assert ref_imgs, f"No reference images found in {REF_DIR}"

ref_embs = []
for p in ref_imgs:
    img = Image.open(p).convert("RGB")
    emb = get_embedding(img)
    if emb is not None:
        ref_embs.append(emb)
        print(f"  [ref] {p.name} ✓")
    else:
        print(f"  [ref] {p.name} — no face detected, skipping")

assert ref_embs, "No faces found in reference images!"
ref_mean = torch.stack(ref_embs).mean(dim=0)
ref_mean = torch.nn.functional.normalize(ref_mean, dim=0)
print(f"  Reference embedding built from {len(ref_embs)} images.\n")

# ── Process downloaded images ─────────────────────────────────────────────────
REJECT_DIR.mkdir(parents=True, exist_ok=True)
all_imgs = sorted(LECUN_DIR.glob("*.jpg")) + sorted(LECUN_DIR.glob("*.png"))
# Skip the provided reference images (yann_*.jpg copies)
all_imgs = [p for p in all_imgs if not p.name.startswith("yann_")]

print(f"Processing {len(all_imgs)} downloaded images...\n")

kept = rejected = no_face = 0
for p in all_imgs:
    try:
        img = Image.open(p).convert("RGB")
    except Exception:
        p.unlink()
        continue

    emb = get_embedding(img)

    if emb is None:
        print(f"  [no_face ] {p.name}")
        shutil.move(str(p), REJECT_DIR / p.name)
        no_face += 1
        continue

    sim = cosine_sim(emb, ref_mean)

    if sim < SIM_THRESH:
        print(f"  [reject  ] {p.name}  sim={sim:.3f}")
        shutil.move(str(p), REJECT_DIR / p.name)
        rejected += 1
        continue

    # Verified — crop to A5 and save back
    cropped = crop_a5_around_face(img)
    if cropped is None:
        cropped = img.resize(OUT_SIZE, Image.LANCZOS)
    cropped.save(p, "JPEG", quality=92)
    print(f"  [keep    ] {p.name}  sim={sim:.3f}")
    kept += 1

# Also crop the provided reference images
for p in LECUN_DIR.glob("yann_*.jpg"):
    try:
        img = Image.open(p).convert("RGB")
        cropped = crop_a5_around_face(img)
        if cropped:
            cropped.save(p, "JPEG", quality=92)
    except Exception:
        pass

print(f"\n=== Done ===")
print(f"  Kept:     {kept}")
print(f"  Rejected: {rejected} (wrong person)")
print(f"  No face:  {no_face}")
print(f"  Total remaining: {kept + len(list(LECUN_DIR.glob('yann_*.jpg')))}")
print(f"  Rejects saved to: {REJECT_DIR}")
