"""
For each celebrity, embed all face images with InceptionResnetV1 (VGGFace2),
compute the median embedding as the identity anchor, then reject any image
whose cosine distance to the anchor exceeds a threshold.
Rejected images are moved to data/rejected/<celeb>/.
"""
import shutil, torch, numpy as np
from pathlib import Path
from PIL import Image
from facenet_pytorch import MTCNN, InceptionResnetV1

FACE_DIR   = Path('./data/celeb_faces')
REJECT_DIR = Path('./data/rejected_identity')
CELEBS     = ['Taylor Swift', 'Barack Obama', 'Yann LeCun']
THRESHOLD  = 0.70   # cosine distance; lower = stricter
MIN_CONF   = 0.99
MIN_FRAC   = 0.15

device = 'cuda' if torch.cuda.is_available() else 'cpu'
mtcnn  = MTCNN(image_size=160, keep_all=False, device=device,
               min_face_size=20, thresholds=[0.6,0.7,0.8], post_process=True)
resnet = InceptionResnetV1(pretrained='vggface2').eval().to(device)

def get_embedding(img_pil):
    W, H = img_pil.size
    boxes, probs = mtcnn.detect(img_pil)
    if boxes is None: return None, None
    best = int(np.argmax(probs))
    if probs[best] < MIN_CONF: return None, None
    b = boxes[best]
    frac = ((b[2]-b[0])*(b[3]-b[1])) / (W*H)
    if frac < MIN_FRAC: return None, None
    face_tensor = mtcnn(img_pil)   # returns normalised 160x160 tensor
    if face_tensor is None: return None, None
    with torch.no_grad():
        emb = resnet(face_tensor.unsqueeze(0).to(device))
    return emb.cpu().numpy()[0], frac

for celeb in CELEBS:
    celeb_dir  = FACE_DIR / celeb
    reject_dir = REJECT_DIR / celeb
    reject_dir.mkdir(parents=True, exist_ok=True)

    imgs = list(celeb_dir.glob('*.jpg')) + list(celeb_dir.glob('*.png'))
    print(f'\n=== {celeb} ({len(imgs)} images) ===')

    # Pass 1: collect embeddings
    embs, paths = [], []
    for p in imgs:
        try:
            img = Image.open(p).convert('RGB')
            emb, _ = get_embedding(img)
            if emb is not None:
                embs.append(emb)
                paths.append(p)
        except: pass

    if len(embs) < 5:
        print(f'  Too few valid faces ({len(embs)}), skipping')
        continue

    embs_arr = np.stack(embs)   # (N, 512)

    # Anchor = geometric median (approximated by iterative weighted mean)
    anchor = embs_arr.mean(axis=0)
    for _ in range(10):
        dists = np.linalg.norm(embs_arr - anchor, axis=1)
        w = 1.0 / (dists + 1e-6)
        anchor = (embs_arr * w[:,None]).sum(0) / w.sum()
    anchor /= np.linalg.norm(anchor)

    # Pass 2: reject outliers
    kept = rejected = 0
    for p, emb in zip(paths, embs):
        e = emb / (np.linalg.norm(emb) + 1e-8)
        cos_dist = 1 - float(np.dot(anchor, e))
        if cos_dist > THRESHOLD:
            shutil.move(str(p), str(reject_dir / p.name))
            print(f'  [reject d={cos_dist:.3f}] {p.name}')
            rejected += 1
        else:
            kept += 1

    # Also move images that had no detectable face
    for p in imgs:
        if p not in paths and p.exists():
            shutil.move(str(p), str(reject_dir / p.name))
            rejected += 1

    print(f'  Kept: {kept}  Rejected: {rejected}')

print('\nDone.')
