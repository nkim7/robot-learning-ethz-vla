"""
Interactive corner-order picker.

For each scene × card, shows the 4 picked polygon corners as numbered dots.
Click them in TL → TR → BR → BL order.
Press 'z' to undo the last click.
Press 's' to skip a card (keeps original order 0,1,2,3).
Press 'q' to quit and print results.

Output: a CORNER_ORDER dict you can paste into the compositor.
"""

import cv2
import numpy as np
from pathlib import Path

SCENE_DIR = Path("clelb_pic/in_order")
SCALE = 4

CARD_QUADS = {
    "scene_L-Swift_C-LeCun_R-Obama_1.jpg": {
        "LEFT":   [(31,43),(93,82),(55,119),(-5,77)],
        "CENTER": [(98,16),(104,62),(153,60),(143,16)],
        "RIGHT":  [(154,79),(192,118),(249,84),(213,49)],
    },
    "scene_L-Swift_C-LeCun_R-Obama_2.jpg": {
        "LEFT":   [(29,44),(93,80),(55,116),(-5,77)],
        "CENTER": [(97,16),(103,60),(153,60),(144,18)],
        "RIGHT":  [(215,52),(156,80),(193,119),(248,85)],
    },
    "scene_L-LeCun_C-Swift_R-Obama.jpg": {
        "LEFT":   [(-5,74),(36,48),(84,85),(41,118)],
        "CENTER": [(91,13),(90,60),(136,59),(134,14)],
        "RIGHT":  [(202,47),(154,82),(202,123),(251,82)],
    },
    "scene_L-LeCun_C-Obama_R-Swift_1.jpg": {
        "LEFT":   [(-4,77),(30,46),(85,83),(49,119)],
        "CENTER": [(108,14),(102,62),(161,66),(156,18)],
        "RIGHT":  [(171,78),(206,124),(264,92),(232,54)],
    },
    "scene_L-LeCun_C-Obama_R-Swift_2.jpg": {
        "LEFT":   [(-5,76),(31,46),(85,81),(46,117)],
        "CENTER": [(106,14),(101,60),(160,64),(157,17)],
        "RIGHT":  [(172,84),(217,125),(268,89),(230,53)],
    },
}

CARD_COLORS = {"LEFT": (0,220,0), "CENTER": (0,140,255), "RIGHT": (255,80,0)}
ROLE_LABELS = ["TL", "TR", "BR", "BL"]
ROLE_COLORS = [(255,255,0),(0,255,255),(0,100,255),(255,0,200)]

# Global state
img_base    = None
H = W       = 0
pts         = []        # current card's 4 poly corners (image coords)
clicked     = []        # indices into pts clicked so far (TL,TR,BR,BL)
current_fname = ""
current_card  = ""
results     = {}        # fname -> card -> [tl_idx, tr_idx, br_idx, bl_idx]

CLICK_RADIUS = 14 * SCALE // 4   # hit-test radius in display pixels


def to_disp(x, y):
    """Image coords → display coords (clamp off-screen pts to edge)."""
    return (int(np.clip(x * SCALE, 0, W - 1)), int(np.clip(y * SCALE, 0, H - 1)))


def nearest_pt(mx, my):
    """Return index of the closest corner to the mouse, or -1 if none close."""
    best_i, best_d = -1, CLICK_RADIUS * 2
    for i, (x, y) in enumerate(pts):
        dx, dy = to_disp(x, y)
        d = ((dx - mx)**2 + (dy - my)**2) ** 0.5
        if d < best_d:
            best_d, best_i = d, i
    return best_i


def draw():
    disp = img_base.copy()
    card_color = CARD_COLORS[current_card]

    # Draw all corners
    for i, (x, y) in enumerate(pts):
        dx, dy = to_disp(x, y)
        if i in clicked:
            role = clicked.index(i)
            color = ROLE_COLORS[role]
            label = f"{i}={ROLE_LABELS[role]}"
        else:
            color = card_color
            label = str(i)
        cv2.circle(disp, (dx, dy), 10, (0,0,0), -1)
        cv2.circle(disp, (dx, dy), 8,  color,   -1)
        cv2.putText(disp, label, (dx+10, dy-6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0,0,0), 3)
        cv2.putText(disp, label, (dx+10, dy-6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 1)

    # Draw assignment arrows so far
    for k in range(len(clicked)-1):
        a = clicked[k]; b = clicked[k+1]
        ax, ay = to_disp(*pts[a])
        bx, by = to_disp(*pts[b])
        cv2.arrowedLine(disp, (ax,ay), (bx,by), ROLE_COLORS[k], 2, tipLength=0.2)

    # Status bar
    next_role = ROLE_LABELS[len(clicked)] if len(clicked) < 4 else "done"
    fname = Path(current_fname).name
    bar = (f"[{fname}]  Card={current_card}  "
           f"Click → {next_role}  |  z=undo  s=skip  q=quit")
    cv2.putText(disp, bar, (4, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,0,0), 3)
    cv2.putText(disp, bar, (4, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,255,255), 1)
    cv2.imshow("Corner Order Picker", disp)


def on_click(event, mx, my, flags, param):
    global clicked
    if event != cv2.EVENT_LBUTTONDOWN:
        return
    if len(clicked) >= 4:
        return
    i = nearest_pt(mx, my)
    if i == -1:
        return
    if i in clicked:
        return          # don't allow same corner twice
    clicked.append(i)
    draw()


cv2.namedWindow("Corner Order Picker")
cv2.setMouseCallback("Corner Order Picker", on_click)

for fname, cards in CARD_QUADS.items():
    results[fname] = {}
    for card_name, poly_pts in cards.items():
        # Load / reload image for each card
        img = cv2.imread(str(SCENE_DIR / fname))
        if img is None:
            print(f"Cannot read {fname}, skipping")
            continue
        hh, ww = img.shape[:2]
        img_base = cv2.resize(img, (ww*SCALE, hh*SCALE), interpolation=cv2.INTER_LINEAR)
        H, W = hh*SCALE, ww*SCALE

        pts            = poly_pts
        clicked        = []
        current_fname  = fname
        current_card   = card_name

        draw()
        print(f"\n--- {Path(fname).name}  [{card_name}] ---")
        print(f"  Corners: { {i: p for i, p in enumerate(pts)} }")
        print(f"  Click TL → TR → BR → BL")

        done = False
        while not done:
            key = cv2.waitKey(0) & 0xFF
            if key == ord('z'):
                if clicked:
                    clicked.pop()
                    draw()
            elif key == ord('s'):
                print(f"  Skipped → keeping 0,1,2,3")
                results[fname][card_name] = [0, 1, 2, 3]
                done = True
            elif key == ord('q'):
                # print and exit
                results[fname][card_name] = clicked if len(clicked)==4 else [0,1,2,3]
                print("\n\nCORNER_ORDER = {")
                for f, cards_d in results.items():
                    print(f'    "{f}": {{')
                    for cn, order in cards_d.items():
                        print(f'        "{cn}": {order},')
                    print("    },")
                print("}")
                cv2.destroyAllWindows()
                raise SystemExit(0)
            elif len(clicked) == 4:
                # All 4 assigned — advance
                results[fname][card_name] = list(clicked)
                print(f"  Order: TL={clicked[0]} TR={clicked[1]} BR={clicked[2]} BL={clicked[3]}")
                done = True

cv2.destroyAllWindows()

print("\n\nCORNER_ORDER = {")
for f, cards_d in results.items():
    print(f'    "{f}": {{')
    for cn, order in cards_d.items():
        print(f'        "{cn}": {order},')
    print("    },")
print("}")
