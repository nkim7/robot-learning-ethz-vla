"""
Push two datasets to HuggingFace:
  nkim7/eval3-celeb-faces   — filtered face pool (Taylor Swift / Barack Obama / Yann LeCun)
  nkim7/eval3-phase2-scenes — synthetic scene images with position labels
"""

import re
from pathlib import Path
from datasets import Dataset, DatasetDict, Features, Value, Image as HFImage
from PIL import Image

# ── 1. Celebrity face dataset ─────────────────────────────────────────────────
FACE_DIR = Path('./data/celeb_faces')
CELEBS   = ['Taylor Swift', 'Barack Obama', 'Yann LeCun']

face_rows = []
for celeb in CELEBS:
    for p in sorted((FACE_DIR/celeb).glob('*.jpg')) + sorted((FACE_DIR/celeb).glob('*.png')):
        try:
            img = Image.open(p).convert('RGB')
            face_rows.append({'image': img, 'celebrity': celeb, 'filename': p.name})
        except: pass

print(f'Face dataset: {len(face_rows)} images')
face_ds = Dataset.from_list(face_rows).cast_column('image', HFImage())
face_ds.push_to_hub('nkim7/eval3-celeb-faces', private=False)
print('  Pushed nkim7/eval3-celeb-faces')

# ── 2. Synthetic scene dataset ─────────────────────────────────────────────────
SCENE_DIR = Path('./data/phase2_scenes')
if not SCENE_DIR.exists():
    SCENE_DIR = Path('./data/composite_test')   # fallback to v6_scene images

scene_rows = []
# filename pattern: s00_scene_L-X_C-Y_R-Z__L-A_C-B_R-C.jpg
pat = re.compile(r'L-(\w+)_C-(\w+)_R-(\w+)\.jpg$')

NAME_MAP = {'Taylor': 'Taylor Swift', 'Barack': 'Barack Obama', 'Yann': 'Yann LeCun',
            'Swift': 'Taylor Swift', 'Obama': 'Barack Obama', 'LeCun': 'Yann LeCun'}

imgs = sorted(SCENE_DIR.glob('*.jpg'))
print(f'Scene dir: {SCENE_DIR} — {len(imgs)} images')

for p in imgs:
    m = pat.search(p.name)
    if not m: continue
    L, C, R = m.group(1), m.group(2), m.group(3)
    left   = NAME_MAP.get(L, L)
    center = NAME_MAP.get(C, C)
    right  = NAME_MAP.get(R, R)
    try:
        img = Image.open(p).convert('RGB')
        scene_rows.append({
            'image':    img,
            'left':     left,
            'center':   center,
            'right':    right,
            'filename': p.name,
        })
    except: pass

print(f'Scene dataset: {len(scene_rows)} images')
scene_ds = Dataset.from_list(scene_rows).cast_column('image', HFImage())
scene_ds.push_to_hub('nkim7/eval3-phase2-scenes', private=False)
print('  Pushed nkim7/eval3-phase2-scenes')

print('\nDone.')
