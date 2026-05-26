"""
Deterministic phase-2 scene compositor.

For each base scene image × every 6 permutations of (Taylor Swift, Barack Obama, Yann LeCun):
  - Warp replacement portraits into the inner photo_quad of each card
  - Preserve card edges, borders, shadows (only inner region is touched)
  - Match brightness/contrast to original local card appearance
  - Slight Gaussian blur + mild noise for camera-style realism
  - Composite at original resolution, resize to 256x256 last

Output: data/scene_synthetic/
        <base_name>__L-<X>_C-<Y>_R-<Z>.jpg
"""

import itertools
import random
from pathlib import Path

import cv2
import numpy as np

FACE_DIR = Path("./data/celeb_faces")
OUT_DIR  = Path("./data/scene_synthetic")
OUT_DIR.mkdir(parents=True, exist_ok=True)

CELEBS = ["Taylor Swift", "Barack Obama", "Yann LeCun"]

# ── Manually-picked polygon data (from pick_card_corners.py) ──────────────────
RAW_QUADS = {
    "scene_L-Swift_C-LeCun_R-Obama_1.jpg": {
        "LEFT":   [(1,69),(29,44),(91,80),(54,117),(1,82)],
        "CENTER": [(97,17),(103,62),(128,61),(133,46),(138,41),(146,40),(142,15)],
        "RIGHT":  [(155,81),(193,118),(248,84),(209,51)],
    },
    "scene_L-Swift_C-LeCun_R-Obama_2.jpg": {
        "LEFT":   [(1,70),(1,82),(54,117),(92,79),(30,44)],
        "CENTER": [(97,16),(104,61),(129,61),(131,48),(135,43),(142,40),(147,41),(142,16)],
        "RIGHT":  [(154,81),(193,118),(249,83),(211,50)],
    },
    "scene_L-LeCun_C-Swift_R-Obama.jpg": {
        "LEFT":   [(1,69),(1,79),(41,117),(82,87),(36,48)],
        "CENTER": [(90,13),(90,60),(127,60),(131,51),(135,43),(138,42),(135,15)],
        "RIGHT":  [(152,83),(203,122),(250,81),(201,47)],
    },
    "scene_L-LeCun_C-Obama_R-Swift_1.jpg": {
        "LEFT":   [(1,69),(29,47),(86,83),(49,119),(1,80)],
        "CENTER": [(105,14),(100,61),(128,63),(131,54),(132,48),(136,43),(146,40),(151,41),(155,40),(157,40),(156,18)],
        "RIGHT":  [(170,80),(207,122),(254,101),(254,80),(229,54)],
    },
    "scene_L-LeCun_C-Obama_R-Swift_2.jpg": {
        "LEFT":   [(1,71),(1,82),(47,116),(85,81),(33,48)],
        "CENTER": [(106,14),(101,61),(128,63),(131,51),(136,43),(147,40),(152,40),(157,40),(156,18)],
        "RIGHT":  [(230,53),(173,83),(219,124),(254,100),(254,75)],
    },
}

# Base image → which celeb is LEFT/CENTER/RIGHT (ground truth layout)
BASE_LAYOUTS = {
    "scene_L-Swift_C-LeCun_R-Obama_1.jpg": ("Taylor Swift", "Yann LeCun",   "Barack Obama"),
    "scene_L-Swift_C-LeCun_R-Obama_2.jpg": ("Taylor Swift", "Yann LeCun",   "Barack Obama"),
    "scene_L-LeCun_C-Swift_R-Obama.jpg":   ("Yann LeCun",   "Taylor Swift",  "Barack Obama"),
    "scene_L-LeCun_C-Obama_R-Swift_1.jpg": ("Yann LeCun",   "Barack Obama",  "Taylor Swift"),
    "scene_L-LeCun_C-Obama_R-Swift_2.jpg": ("Yann LeCun",   "Barack Obama",  "Taylor Swift"),
}

SCENE_DIR = Path("./clelb_pic/in_order")


# ── Geometry helpers ──────────────────────────────────────────────────────────
def poly_to_quad(pts):
    """
    From N polygon points pick 4 corners (TL, TR, BR, BL)
    by angle from centroid — robust to rotated/irregular cards.
    """
    arr = np.array(pts, dtype=np.float32)
    hull = cv2.convexHull(arr).reshape(-1, 2)
    cx, cy = hull.mean(axis=0)
    angles = np.arctan2(hull[:, 1] - cy, hull[:, 0] - cx)
    # Target angles for TL(-135°), TR(-45°), BR(+45°), BL(+135°)
    targets = [-3*np.pi/4, -np.pi/4, np.pi/4, 3*np.pi/4]
    corners = []
    for t in targets:
        diffs = np.abs(((angles - t) + np.pi) % (2*np.pi) - np.pi)
        corners.append(hull[np.argmin(diffs)].astype(int).tolist())
    return corners


def inset_quad(quad, frac=0.10):
    """Shrink a 4-point quad toward its centroid by frac."""
    arr = np.array(quad, dtype=np.float32)
    cx, cy = arr.mean(axis=0)
    inner = arr + frac * (np.array([[cx, cy]] * 4) - arr)
    return inner.astype(int)


def order_clockwise(pts):
    """Re-order 4 points clockwise starting from top-left."""
    arr = np.array(pts, dtype=np.float32)
    cx, cy = arr.mean(axis=0)
    angles = np.arctan2(arr[:, 1] - cy, arr[:, 0] - cx)
    idx = np.argsort(angles)
    ordered = arr[idx]
    # Rotate so top-left is first (min x+y)
    sums = ordered[:, 0] + ordered[:, 1]
    start = np.argmin(sums)
    ordered = np.roll(ordered, -start, axis=0)
    return ordered.astype(int)


# ── Face crop helper ──────────────────────────────────────────────────────────
def face_center_crop(img_bgr):
    """
    Detect face in image; crop A5-ratio portrait centered on face.
    Falls back to center crop if no face found.
    """
    H, W = img_bgr.shape[:2]
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )
    faces = cascade.detectMultiScale(gray, 1.1, 5, minSize=(20, 20))
    A5 = 634 / 448  # height/width ratio

    if len(faces) > 0:
        x, y, fw, fh = max(faces, key=lambda f: f[2] * f[3])
        cx, cy = x + fw // 2, y + fh // 2
        pad = int(max(fw, fh) * 0.6)
        cw = fw + 2 * pad
        ch = int(cw * A5)
        if ch > H:
            ch, cw = H, int(H / A5)
    else:
        cw = min(W, int(H / A5))
        ch = int(cw * A5)
        cx, cy = W // 2, H // 2

    x0 = int(max(0, min(cx - cw // 2, W - cw)))
    y0 = int(max(0, min(cy - ch // 2, H - ch)))
    cropped = img_bgr[y0:y0+ch, x0:x0+cw]
    return cropped


# ── Style matching ────────────────────────────────────────────────────────────
def match_brightness(src, ref_region):
    """Adjust src mean/std to match ref_region (masked pixels only)."""
    if ref_region.size == 0:
        return src
    ref_mean = ref_region.mean(axis=(0, 1), keepdims=True).astype(np.float32)
    ref_std  = ref_region.std(axis=(0, 1), keepdims=True).astype(np.float32) + 1e-6
    src_f    = src.astype(np.float32)
    src_mean = src_f.mean(axis=(0, 1), keepdims=True)
    src_std  = src_f.std(axis=(0, 1), keepdims=True) + 1e-6
    adjusted = (src_f - src_mean) / src_std * ref_std + ref_mean
    return np.clip(adjusted, 0, 255).astype(np.uint8)


def add_camera_noise(img, sigma=3):
    noise = np.random.normal(0, sigma, img.shape).astype(np.int16)
    return np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)


# ── Core compositing ──────────────────────────────────────────────────────────
def paste_portrait(base, face_path, poly_pts, blur_ksize=3):
    """
    Warp face_path into the inner photo region of the card defined by poly_pts.
    Preserves card border/edges — only replaces the inner printed photo area.
    """
    H, W = base.shape[:2]

    # 1. Compute quads
    paper_quad = poly_to_quad(poly_pts)              # 4-corner card boundary
    photo_quad = inset_quad(paper_quad, frac=0.10)   # inner portrait region only
    photo_quad = order_clockwise(photo_quad)

    # 2. Load and crop portrait
    face = cv2.imread(str(face_path))
    if face is None:
        return base
    face = face_center_crop(face)
    fH, fW = face.shape[:2]

    # 3. Warp portrait into photo_quad
    src_pts = np.float32([[0, 0], [fW, 0], [fW, fH], [0, fH]])
    dst_pts = photo_quad.astype(np.float32)
    M = cv2.getPerspectiveTransform(src_pts, dst_pts)
    warped = cv2.warpPerspective(face, M, (W, H))

    # 4. Build mask from photo_quad polygon
    mask = np.zeros((H, W), dtype=np.uint8)
    cv2.fillConvexPoly(mask, photo_quad.astype(np.int32), 255)

    # 5. Sample original card appearance for brightness matching
    ref_pixels = base[mask > 0]
    if ref_pixels.size > 0:
        warped_pixels = warped[mask > 0].reshape(-1, 3)
        warped_matched = match_brightness(
            warped[mask > 0].reshape(1, -1, 3),
            ref_pixels.reshape(1, -1, 3)
        ).reshape(-1, 3)
        warped_out = base.copy()
        warped_out[mask > 0] = warped_matched
        warped = warped_out

    # 6. Blur to match webcam softness
    warped_blurred = cv2.GaussianBlur(warped, (blur_ksize, blur_ksize), 0)
    mask_blurred   = cv2.GaussianBlur(mask,   (5, 5), 0)

    # 7. Alpha blend only inside photo_quad
    alpha = mask_blurred.astype(np.float32) / 255.0
    alpha = alpha[:, :, None]
    result = base.astype(np.float32) * (1 - alpha) + warped_blurred.astype(np.float32) * alpha
    return result.astype(np.uint8)


# ── Face pool ─────────────────────────────────────────────────────────────────
rng = random.Random(42)
face_pool = {}
for c in CELEBS:
    imgs = list((FACE_DIR / c).glob("*.jpg")) + list((FACE_DIR / c).glob("*.png"))
    rng.shuffle(imgs)
    face_pool[c] = imgs
    print(f"  {c}: {len(imgs)} face images")

# ── Main loop ─────────────────────────────────────────────────────────────────
all_perms = list(itertools.permutations(CELEBS))
saved = 0

for base_fname, quads in RAW_QUADS.items():
    base_path = SCENE_DIR / base_fname
    if not base_path.exists():
        print(f"  WARNING: base not found: {base_path}")
        continue

    base_orig = cv2.imread(str(base_path))
    if base_orig is None:
        continue
    H_orig, W_orig = base_orig.shape[:2]

    for perm in all_perms:
        left_celeb, center_celeb, right_celeb = perm

        # Pick a face image for each celeb (deterministic, different per permutation)
        perm_seed = abs(hash(base_fname + "".join(perm))) % 10000
        left_face   = face_pool[left_celeb][perm_seed   % len(face_pool[left_celeb])]
        center_face = face_pool[center_celeb][(perm_seed+1) % len(face_pool[center_celeb])]
        right_face  = face_pool[right_celeb][(perm_seed+2) % len(face_pool[right_celeb])]

        # Composite at original resolution
        out = base_orig.copy()
        out = paste_portrait(out, left_face,   quads["LEFT"])
        out = paste_portrait(out, center_face, quads["CENTER"])
        out = paste_portrait(out, right_face,  quads["RIGHT"])

        # Mild noise
        out = add_camera_noise(out, sigma=2)

        # Resize to 256×256 last
        out_256 = cv2.resize(out, (256, 256), interpolation=cv2.INTER_AREA)

        stem   = Path(base_fname).stem
        L, C, R = left_celeb.split()[0], center_celeb.split()[0], right_celeb.split()[0]
        out_name = f"{stem}__L-{L}_C-{C}_R-{R}.jpg"
        cv2.imwrite(str(OUT_DIR / out_name), out_256, [cv2.IMWRITE_JPEG_QUALITY, 92])
        saved += 1

print(f"\nDone — {saved} synthetic scenes → {OUT_DIR}")
print(f"  5 bases × 6 permutations = {5*6} total")
