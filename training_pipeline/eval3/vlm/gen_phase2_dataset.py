"""
Generate 150 synthetic phase-2 scene images (5 scenes × 6 perms × 5 seeds)
and write a JSONL manifest with QA pairs for SmolVLM2 training.

QA pairs per image (6 total):
  - "Where is [celeb]? Answer LEFT, CENTER, or RIGHT."  →  "LEFT/CENTER/RIGHT"
  - "Who is on the [pos] card? Give the full name."     →  "[Name]"
"""

import cv2, json, numpy as np, random, torch
from pathlib import Path
from PIL import Image
from facenet_pytorch import MTCNN

FACE_DIR = Path('./data/celeb_faces')
OUT_DIR  = Path('./data/phase2_scenes'); OUT_DIR.mkdir(exist_ok=True)
CELEBS   = ['Taylor Swift', 'Barack Obama', 'Yann LeCun']
N_SEEDS  = 5   # different face samples per (scene, perm) → 5×30 = 150 images

CARD_QUADS = {
    "scene_L-Swift_C-LeCun_R-Obama_1.jpg": {
        "LEFT":   [(31,43),(93,82),(55,119),(-5,77)],
        "CENTER": [(98,16),(104,62),(153,60),(143,16)],
        "RIGHT":  [(154,79),(192,118),(249,84),(213,49)],
        "CAN_OCCLUDE": [(148,41),(136,44),(133,53),(130,66),(122,84),(115,99),(116,115),(121,116),(132,117),(138,104),(146,94),(161,71),(165,62),(167,54),(160,46),(158,42)],
    },
    "scene_L-Swift_C-LeCun_R-Obama_2.jpg": {
        "LEFT":   [(29,44),(93,80),(55,116),(-5,77)],
        "CENTER": [(97,16),(103,60),(153,60),(144,18)],
        "RIGHT":  [(215,52),(156,80),(193,119),(248,85)],
        "CAN_OCCLUDE": [(146,41),(137,43),(134,49),(132,59),(129,72),(123,84),(115,96),(113,106),(122,116),(136,116),(144,94),(166,62),(166,53),(156,42)],
    },
    "scene_L-LeCun_C-Swift_R-Obama.jpg": {
        "LEFT":   [(-5,74),(36,48),(84,85),(41,118)],
        "CENTER": [(91,13),(90,60),(136,59),(134,14)],
        "RIGHT":  [(202,47),(154,82),(202,123),(251,82)],
        "CAN_OCCLUDE": [(147,40),(137,44),(133,56),(128,69),(123,81),(115,97),(115,114),(121,116),(130,116),(138,116),(147,95),(152,83),(158,74),(164,65),(168,56),(160,46),(157,41)],
    },
    "scene_L-LeCun_C-Obama_R-Swift_1.jpg": {
        "LEFT":   [(-4,77),(30,46),(85,83),(49,119)],
        "CENTER": [(108,14),(102,62),(161,66),(156,18)],
        "RIGHT":  [(171,78),(206,124),(264,92),(232,54)],
        "CAN_OCCLUDE": [(144,41),(137,43),(131,55),(129,66),(119,86),(113,107),(119,114),(130,115),(136,115),(140,100),(152,84),(167,61),(167,51),(157,45),(155,40)],
    },
    "scene_L-LeCun_C-Obama_R-Swift_2.jpg": {
        "LEFT":   [(-5,76),(31,46),(85,81),(46,117)],
        "CENTER": [(106,14),(101,60),(160,64),(157,17)],
        "RIGHT":  [(172,84),(217,125),(268,89),(230,53)],
        "CAN_OCCLUDE": [(141,41),(133,45),(130,57),(129,69),(121,83),(114,96),(113,112),(121,118),(132,118),(138,116),(137,105),(144,100),(152,85),(156,76),(165,64),(166,54),(160,47),(157,43),(150,41)],
    },
}

CORNER_ORDER = {
    "scene_L-Swift_C-LeCun_R-Obama_1.jpg": {"LEFT":[3,0,1,2],"CENTER":[0,3,2,1],"RIGHT":[3,2,1,0]},
    "scene_L-Swift_C-LeCun_R-Obama_2.jpg": {"LEFT":[3,0,1,2],"CENTER":[0,3,2,1],"RIGHT":[0,3,2,1]},
    "scene_L-LeCun_C-Swift_R-Obama.jpg":   {"LEFT":[0,1,2,3],"CENTER":[0,3,2,1],"RIGHT":[0,3,2,1]},
    "scene_L-LeCun_C-Obama_R-Swift_1.jpg": {"LEFT":[0,1,2,3],"CENTER":[0,3,2,1],"RIGHT":[3,2,1,0]},
    "scene_L-LeCun_C-Obama_R-Swift_2.jpg": {"LEFT":[0,1,2,3],"CENTER":[0,3,2,1],"RIGHT":[3,2,1,0]},
}

ALL_PERMS = [
    ('Taylor Swift', 'Barack Obama', 'Yann LeCun'),
    ('Taylor Swift', 'Yann LeCun',   'Barack Obama'),
    ('Barack Obama', 'Taylor Swift',  'Yann LeCun'),
    ('Barack Obama', 'Yann LeCun',    'Taylor Swift'),
    ('Yann LeCun',   'Taylor Swift',  'Barack Obama'),
    ('Yann LeCun',   'Barack Obama',  'Taylor Swift'),
]

# ── Face pool ─────────────────────────────────────────────────────────────────
device = 'cuda' if torch.cuda.is_available() else 'cpu'
mtcnn  = MTCNN(keep_all=False, device=device, min_face_size=20,
               thresholds=[0.6,0.7,0.8], post_process=False)

face_pool = {}
for c in CELEBS:
    imgs = list((FACE_DIR/c).glob('*.jpg')) + list((FACE_DIR/c).glob('*.png'))
    valid = []
    for p in imgs:
        try:
            img = Image.open(p).convert('RGB')
            W, H = img.size
            if min(W, H) < 120: continue
            hsv_s = np.array(img.convert('HSV'))[:,:,1].mean()
            if hsv_s < 30 or hsv_s > 140: continue
            boxes, probs = mtcnn.detect(img)
            if boxes is None: continue
            best = int(np.argmax(probs))
            if probs[best] < 0.99: continue
            b = boxes[best]
            if ((b[2]-b[0])*(b[3]-b[1]))/(W*H) < 0.15: continue
            x1,y1,x2,y2 = [int(np.clip(v,0,dim-1)) for v,dim in zip(b,[W,H,W,H])]
            arr = np.array(img)[y1:y2, x1:x2]
            if arr.size == 0 or arr.std() < 20: continue
            valid.append((str(p), b.tolist()))
        except: pass
    face_pool[c] = valid
    print(f'{c}: {len(valid)} valid faces')

# ── Helpers ───────────────────────────────────────────────────────────────────
def get_quad(poly_pts, order):
    return np.array([poly_pts[i] for i in order], dtype=np.float32)

def expand_quad(quad, frac=0.35):
    arr = np.array(quad, np.float32)
    cx, cy = arr.mean(0)
    return (arr + frac*(arr - [[cx,cy]]*4)).astype(int)

def crop_portrait(img_bgr, box):
    x1,y1,x2,y2 = [float(v) for v in box]
    H,W = img_bgr.shape[:2]
    fw,fh = x2-x1, y2-y1
    cx,cy = (x1+x2)/2, (y1+y2)/2
    pad = max(fw,fh)*4.0
    cw = fw+2*pad; ch = cw*(634/448)
    if ch > H: ch=H; cw=ch*(448/634)
    x0=int(max(0,min(cx-cw/2,W-cw))); y0=int(max(0,min(cy-ch/2,H-ch)))
    return img_bgr[y0:y0+int(ch), x0:x0+int(cw)]

def match_brightness(warped, base, mask):
    ref=base[mask>128].astype(np.float32); src=warped[mask>128].astype(np.float32)
    if ref.size==0 or src.size==0: return warped
    rm,rs=ref.mean(0),ref.std(0)+1e-6; sm,ss=src.mean(0),src.std(0)+1e-6
    out=warped.copy().astype(np.float32)
    out[mask>128]=np.clip((src-sm)/ss*rs+rm,0,255)
    return out.astype(np.uint8)

def paste(base, face_path, face_box, poly_pts, order, can_poly):
    H,W = base.shape[:2]
    pq = expand_quad(get_quad(poly_pts, order))
    face = cv2.imread(str(face_path))
    if face is None: return base
    face = crop_portrait(face, face_box)
    if face.size==0: return base
    fH,fW = face.shape[:2]
    src=np.float32([[0,0],[fW,0],[fW,fH],[0,fH]]); dst=pq.astype(np.float32)
    M=cv2.getPerspectiveTransform(src,dst)
    warped=cv2.warpPerspective(face,M,(W,H))
    mask=np.zeros((H,W),np.uint8); cv2.fillConvexPoly(mask,pq.astype(np.int32),255)
    can_pts=np.array(can_poly,np.int32)
    can_pts[:,0]=np.clip(can_pts[:,0],0,W-1); can_pts[:,1]=np.clip(can_pts[:,1],0,H-1)
    cv2.fillPoly(mask,[can_pts],0)
    warped=match_brightness(warped,base,mask)
    warped=cv2.GaussianBlur(warped,(3,3),0)
    mask_b=cv2.GaussianBlur(mask,(5,5),0)
    a=mask_b.astype(np.float32)[:,:,None]/255.
    return np.clip(base*(1-a)+warped*a,0,255).astype(np.uint8)

# ── Generate ──────────────────────────────────────────────────────────────────
manifest = []
saved = 0

for seed in range(N_SEEDS):
    rng = random.Random(seed * 1000 + 42)
    for base_fname, quads in CARD_QUADS.items():
        orders = CORNER_ORDER[base_fname]
        can    = quads['CAN_OCCLUDE']
        base   = cv2.imread(base_fname)
        if base is None: print(f'Missing {base_fname}'); continue

        for perm in ALL_PERMS:
            L, C, R = perm
            lp,lb = rng.choice(face_pool[L])
            cp,cb = rng.choice(face_pool[C])
            rp,rb = rng.choice(face_pool[R])

            out = base.copy()
            out = paste(out, lp, lb, quads['LEFT'],   orders['LEFT'],   can)
            out = paste(out, cp, cb, quads['CENTER'], orders['CENTER'], can)
            out = paste(out, rp, rb, quads['RIGHT'],  orders['RIGHT'],  can)

            noise = np.random.normal(0,1.5,out.shape).astype(np.int16)
            out   = np.clip(out.astype(np.int16)+noise,0,255).astype(np.uint8)
            out   = cv2.resize(out,(256,256),interpolation=cv2.INTER_AREA)

            stem = Path(base_fname).stem
            fname = f's{seed:02d}_{stem}__L-{L.split()[0]}_C-{C.split()[0]}_R-{R.split()[0]}.jpg'
            cv2.imwrite(str(OUT_DIR/fname), out, [cv2.IMWRITE_JPEG_QUALITY,92])

            pos_map = {L:'LEFT', C:'CENTER', R:'RIGHT'}
            # 6 QA pairs per image
            for celeb in [L, C, R]:
                pos = pos_map[celeb]
                manifest.append({
                    "image": str(OUT_DIR/fname),
                    "question": f"Where is {celeb}? Answer LEFT, CENTER, or RIGHT.",
                    "answer": pos,
                })
            for pos, celeb in [('LEFT',L),('CENTER',C),('RIGHT',R)]:
                manifest.append({
                    "image": str(OUT_DIR/fname),
                    "question": f"Who is on the {pos} card? Give the full name.",
                    "answer": celeb,
                })
            saved += 1

manifest_path = Path('./data/phase2_manifest.jsonl')
with open(manifest_path, 'w') as f:
    for row in manifest:
        f.write(json.dumps(row) + '\n')

print(f'\nDone — {saved} images, {len(manifest)} QA pairs → {manifest_path}')
