"""
Interactive card corner picker — N-point polygon per card.

Click any number of points to outline each card boundary.
Handles occluded corners and odd angles.
Press Enter/Space to finish the current card and move to the next.
Press 'z' to undo the last point.
Press 's' to skip the current image.
Press 'q' to quit early.

Cards order: LEFT → CENTER → RIGHT (x3 per image)
The full polygon is used for blending mask.
4 warp corners are auto-extracted from the convex hull (TL/TR/BR/BL).

Usage:
    python3 pick_card_corners.py clelb_pic/in_order/scene_*.jpg
"""

import sys
import cv2
import numpy as np
from pathlib import Path

PATHS = sys.argv[1:] if len(sys.argv) > 1 else sorted(
    Path("clelb_pic/in_order").glob("scene_*.jpg")
)
SCALE   = 3
PADDING = 80   # extra canvas around image (in original pixels) for out-of-frame corners

CARDS  = ["LEFT", "CENTER", "RIGHT", "CAN_OCCLUDE"]
COLORS = [(0, 220, 0), (220, 100, 0), (0, 80, 255), (0, 220, 220)]

all_results  = {}
points       = []   # current card's clicked points
card_idx     = 0
cards_done   = []   # list of finished polygon point-lists for this image
img_orig     = None
img_display  = None
H = W        = 0
current_path = ""


def load_image(path):
    global img_orig, img_display, H, W, points, card_idx, cards_done, current_path
    img_orig = cv2.imread(str(path))
    if img_orig is None:
        return False
    H, W = img_orig.shape[:2]
    # Build padded canvas: gray border around the image
    canvas = np.full((H + 2*PADDING, W + 2*PADDING, 3), 60, dtype=np.uint8)
    canvas[PADDING:PADDING+H, PADDING:PADDING+W] = img_orig
    # Draw faint image boundary
    cv2.rectangle(canvas, (PADDING, PADDING), (PADDING+W-1, PADDING+H-1), (120,120,120), 1)
    img_display = cv2.resize(canvas,
                             ((W + 2*PADDING)*SCALE, (H + 2*PADDING)*SCALE),
                             interpolation=cv2.INTER_NEAREST)
    points     = []
    card_idx   = 0
    cards_done = []
    current_path = str(path)
    return True


def hull_corners(pts):
    """Return 4 corners (TL, TR, BR, BL) from convex hull of pts."""
    arr = np.array(pts, dtype=np.float32)
    hull = cv2.convexHull(arr).reshape(-1, 2)
    cx, cy = hull.mean(axis=0)
    def angle(p): return np.arctan2(p[1] - cy, p[0] - cx)
    hull_sorted = sorted(hull, key=angle)
    # top-left: min(x+y), top-right: max(x-y), bottom-right: max(x+y), bottom-left: min(x-y)
    tl = min(hull, key=lambda p: p[0] + p[1])
    tr = max(hull, key=lambda p: p[0] - p[1])
    br = max(hull, key=lambda p: p[0] + p[1])
    bl = min(hull, key=lambda p: p[0] - p[1])
    return [tl.tolist(), tr.tolist(), br.tolist(), bl.tolist()]


def inset_quad(quad, frac=0.10):
    """Inset a 4-point quad toward its centroid by frac."""
    arr = np.array(quad, dtype=np.float32)
    cx, cy = arr.mean(axis=0)
    inner = arr + frac * (np.array([[cx, cy]] * 4) - arr)
    return inner.astype(int).tolist()


def to_disp(x, y):
    """Convert image-space coords (can be negative / out of frame) to display pixels."""
    return ((x + PADDING) * SCALE, (y + PADDING) * SCALE)


def draw():
    disp  = img_display.copy()
    fname = Path(current_path).name

    # Draw finished cards
    for ci, poly_pts in enumerate(cards_done):
        poly = np.array([to_disp(x, y) for x, y in poly_pts], np.int32)
        cv2.polylines(disp, [poly], True, COLORS[ci], 2)
        for x, y in poly_pts:
            cv2.circle(disp, to_disp(x, y), 5, COLORS[ci], -1)
        cx = int(np.mean([p[0] for p in poly_pts]))
        cy = int(np.mean([p[1] for p in poly_pts]))
        cv2.putText(disp, CARDS[ci], to_disp(cx-18, cy+5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0,0,0), 3)
        cv2.putText(disp, CARDS[ci], to_disp(cx-18, cy+5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, COLORS[ci], 1)

    # Draw current card in progress
    for x, y in points:
        cv2.circle(disp, to_disp(x, y), 5, COLORS[card_idx], -1)
    for i in range(len(points) - 1):
        cv2.line(disp, to_disp(*points[i]), to_disp(*points[i+1]), COLORS[card_idx], 1)
    if len(points) > 2:
        cv2.line(disp, to_disp(*points[-1]), to_disp(*points[0]), COLORS[card_idx], 1)

    card_name = CARDS[card_idx] if card_idx < len(CARDS) else "done"
    n  = len(points)
    hint = "(cyan) outline the can/robot region to keep in front" if card_name == "CAN_OCCLUDE" else "(gray border = out-of-frame corners OK)"
    bar = (f"[{fname}]  Card={card_name}  {n} pts  |  "
           f"Enter/Space=done  z=undo  s=skip  q=quit  {hint}")
    cv2.putText(disp, bar, (4, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.36, (0,0,0), 3)
    cv2.putText(disp, bar, (4, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.36, (255,255,255), 1)
    cv2.imshow("Corner Picker", disp)


def on_click(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN and card_idx < len(CARDS):
        # Convert display coords back to image-space (allowing out-of-frame values)
        img_x = x // SCALE - PADDING
        img_y = y // SCALE - PADDING
        points.append((img_x, img_y))
        draw()


def finish_card():
    global card_idx, points
    if len(points) < 3:
        print(f"  Need at least 3 points for {CARDS[card_idx]}, got {len(points)} — skipping card")
        points = []
        return
    cards_done.append(list(points))
    print(f"  {CARDS[card_idx]}: {len(points)} points")
    points = []
    card_idx += 1
    if card_idx < len(CARDS):
        draw()


def finish_image():
    if len(cards_done) == 4:
        result = {}
        for ci, name in enumerate(["LEFT", "CENTER", "RIGHT"]):
            poly  = cards_done[ci]
            quad  = hull_corners(poly)
            photo = inset_quad(quad, 0.10)
            result[name] = {"poly": poly, "paper": quad, "photo": photo}
        result["CAN_OCCLUDE"] = {"poly": cards_done[3]}
        all_results[current_path] = result
        print(f"  Image done — {Path(current_path).name}")


def print_results():
    print("\n\n# ═══════════════ CARD_QUADS — paste into compositing script ═══════════════")
    print("CARD_QUADS = {")
    for path, cards in all_results.items():
        fname = Path(path).name
        print(f'    "{fname}": {{')
        for card_name, data in cards.items():
            if card_name == "CAN_OCCLUDE":
                print(f'        "CAN_OCCLUDE": {data["poly"]},')
            else:
                print(f'        "{card_name}": {{')
                print(f'            "poly":  {data["poly"]},')
                print(f'            "paper": {[[int(v) for v in p] for p in data["paper"]]},')
                print(f'            "photo": {[[int(v) for v in p] for p in data["photo"]]},')
                print(f'        }},')
        print("    },")
    print("}")
    print("# ═════════════════════════════════════════════════════════════════════════\n")


cv2.namedWindow("Corner Picker")
cv2.setMouseCallback("Corner Picker", on_click)

for path in PATHS:
    if not load_image(path):
        print(f"Skipping (can't read): {path}")
        continue
    print(f"\n--- {Path(path).name} ---")
    draw()

    done = False
    while not done:
        key = cv2.waitKey(0) & 0xFF
        if key == ord('z'):
            if points:
                points.pop()
                draw()
            elif cards_done:
                card_idx -= 1
                points = list(cards_done.pop())
                draw()
        elif key in (13, 32):   # Enter or Space → finish current card
            finish_card()
            if card_idx == len(CARDS):
                finish_image()
                done = True
        elif key == ord('s'):
            print(f"  Skipped.")
            done = True
        elif key == ord('q'):
            print_results()
            cv2.destroyAllWindows()
            sys.exit(0)

print_results()
cv2.destroyAllWindows()
