"""
Download celebrity face images using DuckDuckGo image search.
Filters images with OpenCV face detection to ensure only real face photos are saved.

Usage:
    pip install duckduckgo_search opencv-python pillow requests
    python3 download_celeb_images.py --stage toy
    python3 download_celeb_images.py --stage top50
    python3 download_celeb_images.py --stage rest
"""

import argparse
import hashlib
import io
import os
import time
import urllib.request
from pathlib import Path

import cv2
import numpy as np
import requests
from duckduckgo_search import DDGS
from PIL import Image

parser = argparse.ArgumentParser()
parser.add_argument("--out_dir", default="./data/celeb_faces")
parser.add_argument("--toy_img_dir", default="./clelb_pic")
parser.add_argument("--stage", default="toy",
                    choices=["toy", "top50", "rest", "all"])
parser.add_argument("--target", type=int, default=100)
args = parser.parse_args()

OUT = Path(args.out_dir)
OUT.mkdir(parents=True, exist_ok=True)

# Face detector
CASCADE = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

# ── Celebrity list ────────────────────────────────────────────────────────────
TOY = ["Taylor Swift", "Barack Obama", "Yann LeCun"]

TOP50 = [
    "Elon Musk", "Jeff Bezos", "Bill Gates", "Mark Zuckerberg",
    "Donald Trump", "Joe Biden", "Vladimir Putin", "Emmanuel Macron",
    "Xi Jinping", "Narendra Modi", "Justin Trudeau", "Kamala Harris",
    "Angela Merkel", "Boris Johnson", "Volodymyr Zelensky",
    "Leonardo DiCaprio", "Tom Hanks", "Brad Pitt", "Scarlett Johansson",
    "Jennifer Aniston", "Tom Cruise", "Will Smith", "Denzel Washington",
    "Morgan Freeman", "Robert Downey Jr", "Keanu Reeves", "Johnny Depp",
    "Beyonce", "Rihanna", "Adele", "Lady Gaga", "Billie Eilish",
    "Ariana Grande", "Justin Bieber", "Ed Sheeran", "Drake", "Eminem",
    "Cristiano Ronaldo", "Lionel Messi", "LeBron James", "Serena Williams",
    "Roger Federer", "Tiger Woods", "Michael Jordan", "Kobe Bryant",
    "Kylian Mbappe", "Lewis Hamilton", "Oprah Winfrey", "Kim Kardashian",
]

REST = [
    "Sundar Pichai", "Satya Nadella", "Tim Cook", "Jensen Huang",
    "Geoffrey Hinton", "Andrew Ng", "Demis Hassabis", "Sam Altman",
    "Larry Page", "Sergey Brin", "Warren Buffett", "Michelle Obama",
    "George W Bush", "Bill Clinton", "Nelson Mandela", "Pope Francis",
    "Angelina Jolie", "Meryl Streep", "Ryan Reynolds", "Chris Evans",
    "Natalie Portman", "Emma Stone", "Anne Hathaway", "Julia Roberts",
    "Margot Robbie", "Zendaya", "Timothee Chalamet", "Dwayne Johnson",
    "Harrison Ford", "Hugh Jackman", "Chris Hemsworth", "Emma Watson",
    "Charlize Theron", "Halle Berry", "Steven Spielberg",
    "Christopher Nolan", "Martin Scorsese",
    "Dua Lipa", "Selena Gomez", "The Weeknd", "Harry Styles",
    "Nicki Minaj", "Post Malone", "Kendrick Lamar", "Elton John",
    "Madonna", "Olivia Rodrigo", "Sabrina Carpenter",
    "Rafael Nadal", "Novak Djokovic", "Usain Bolt", "Simone Biles",
    "Stephen Curry", "Kevin Durant", "Patrick Mahomes", "Tom Brady",
    "Naomi Osaka", "Erling Haaland", "Luka Doncic", "Max Verstappen",
    "Muhammad Ali", "Mike Tyson", "Shaquille O'Neal",
    "Jimmy Fallon", "Stephen Colbert", "Trevor Noah", "Kevin Hart",
    "Dave Chappelle", "Malala Yousafzai", "Greta Thunberg",
    "Neil deGrasse Tyson", "Stephen Hawking",
    "Meghan Markle", "Prince William", "Kate Middleton", "King Charles",
    "Kylie Jenner",
]

STAGE_MAP = {
    "toy": TOY,
    "top50": TOP50,
    "rest": REST,
    "all": TOY + TOP50 + REST,
}
CELEBS = STAGE_MAP[args.stage]

# Multiple search queries per celebrity for diversity
QUERIES = {
    "Yann LeCun": [
        "Yann LeCun portrait",
        "Yann LeCun Meta AI researcher",
        "Yann LeCun deep learning professor NYU",
        "Yann LeCun interview",
        "Yann LeCun headshot",
    ],
    "Barack Obama": [
        "Barack Obama portrait",
        "Barack Obama president face",
        "Barack Obama official photo",
        "Barack Obama speech close up",
        "Barack Obama headshot",
    ],
    "Taylor Swift": [
        "Taylor Swift portrait",
        "Taylor Swift face close up",
        "Taylor Swift headshot",
        "Taylor Swift red carpet",
        "Taylor Swift interview",
    ],
    "Geoffrey Hinton": [
        "Geoffrey Hinton portrait",
        "Geoffrey Hinton AI researcher",
        "Geoffrey Hinton headshot",
    ],
    "Demis Hassabis": [
        "Demis Hassabis portrait",
        "Demis Hassabis DeepMind CEO",
        "Demis Hassabis headshot",
    ],
    "Jensen Huang": [
        "Jensen Huang portrait",
        "Jensen Huang Nvidia CEO",
        "Jensen Huang headshot",
    ],
}

def default_queries(name: str) -> list[str]:
    return [
        f"{name} portrait",
        f"{name} headshot",
        f"{name} face photo",
    ]


# ── Image utilities ────────────────────────────────────────────────────────────
def has_face(img: Image.Image) -> bool:
    arr = np.array(img.convert("RGB"))
    gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
    faces = CASCADE.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5,
                                     minSize=(60, 60))
    return len(faces) > 0


def fetch_and_save(url: str, dest: Path, prefix: str) -> bool:
    try:
        r = requests.get(url, timeout=8,
                         headers={"User-Agent": "Mozilla/5.0 (research)"})
        if r.status_code != 200:
            return False
        img = Image.open(io.BytesIO(r.content)).convert("RGB")
        if img.size[0] < 100 or img.size[1] < 100:
            return False
        if not has_face(img):
            return False
        h = hashlib.md5(url.encode()).hexdigest()[:10]
        dest.mkdir(parents=True, exist_ok=True)
        img.save(dest / f"{prefix}_{h}.jpg", "JPEG", quality=90)
        return True
    except Exception:
        return False


def count_images(celeb_dir: Path) -> int:
    if not celeb_dir.exists():
        return 0
    return len(list(celeb_dir.glob("*.jpg")) + list(celeb_dir.glob("*.png")))


# ── Copy provided TOY images ───────────────────────────────────────────────────
def copy_toy_images():
    toy_dir = Path(args.toy_img_dir)
    if not toy_dir.exists():
        return
    mapping = {
        "Taylor Swift": "taylor_*.png",
        "Barack Obama": "obama_*.png",
        "Yann LeCun":   "yann_*.png",
    }
    for name, pattern in mapping.items():
        dest = OUT / name
        dest.mkdir(parents=True, exist_ok=True)
        for src in toy_dir.glob(pattern):
            try:
                img = Image.open(src).convert("RGB")
                img.save(dest / src.name.replace(".png", ".jpg"), "JPEG", quality=90)
            except Exception:
                pass


# ── Main download ──────────────────────────────────────────────────────────────
def download_celeb(name: str, target: int):
    dest = OUT / name
    existing = count_images(dest)
    if existing >= target:
        print(f"  [skip] {name}: {existing} images already")
        return

    queries = QUERIES.get(name, default_queries(name))
    need = target - existing
    per_query = max(need // len(queries) + 10, 20)

    print(f"  {name}: need {need} more images ({len(queries)} queries)")

    saved = 0
    with DDGS() as ddgs:
        for query in queries:
            if existing + saved >= target:
                break
            try:
                results = list(ddgs.images(
                    query,
                    max_results=per_query,
                    type_image="photo",
                    size="medium",
                ))
            except Exception as e:
                print(f"    [warn] DDG query failed: {e}")
                time.sleep(2)
                continue

            for r in results:
                if existing + saved >= target:
                    break
                url = r.get("image", "")
                if not url:
                    continue
                if fetch_and_save(url, dest, f"ddg_{saved:04d}"):
                    saved += 1
                time.sleep(0.05)

            time.sleep(1)  # between queries

    total = existing + saved
    status = "OK" if total >= target // 2 else "LOW"
    print(f"  [{status}] {name}: {total} images (+{saved} new)")


# ── Run ────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print(f"Stage: {args.stage} | {len(CELEBS)} celebrities | target={args.target}\n")

    # Always copy provided TOY images first
    copy_toy_images()

    for i, celeb in enumerate(CELEBS):
        print(f"[{i+1}/{len(CELEBS)}] {celeb}")
        download_celeb(celeb, args.target)

    print("\n=== FINAL COUNTS ===")
    for celeb in CELEBS:
        n = count_images(OUT / celeb)
        print(f"  {'OK' if n >= args.target // 2 else 'LOW':3} {celeb}: {n}")
