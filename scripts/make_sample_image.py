"""Generate a synthetic test image for exercising the pattern pipeline.

Not decorative: it is built to contain the specific structures that break naive
implementations — smooth gradients (which fragment into banding), thin shapes
(which produce unlabelable slivers), nested holes (which fill in if contour
hierarchy is mishandled), and sensor-like noise (which explodes region counts if
pre-smoothing is wrong).

    python scripts/make_sample_image.py sample.png
"""

from __future__ import annotations

import sys

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

WIDTH, HEIGHT = 900, 1200


def build() -> Image.Image:
    img = Image.new("RGB", (WIDTH, HEIGHT), "#ffffff")
    draw = ImageDraw.Draw(img)

    # Sky: a vertical gradient, the classic quantization stress test.
    top = np.array([120, 185, 230], dtype=np.float64)
    bottom = np.array([226, 240, 248], dtype=np.float64)
    for y in range(HEIGHT):
        t = y / HEIGHT
        draw.line([(0, y), (WIDTH, y)], fill=tuple((top * (1 - t) + bottom * t).astype(int)))

    # Sun, and rolling hills in distinct greens.
    draw.ellipse([640, 90, 820, 270], fill="#ffd34d")
    draw.ellipse([-220, 760, 620, 1320], fill="#4e8a46")
    draw.ellipse([420, 850, 1200, 1400], fill="#3a6b38")

    # A bird: overlapping organic shapes, one with a genuine hole (the eye).
    draw.ellipse([300, 380, 620, 760], fill="#d94f3d")          # body
    draw.ellipse([250, 300, 470, 520], fill="#e8653f")          # head
    draw.polygon([(250, 400), (160, 440), (250, 475)], fill="#f2a63b")  # beak
    draw.ellipse([318, 358, 358, 398], fill="#ffffff")          # eye white
    draw.ellipse([330, 370, 346, 386], fill="#20201e")          # pupil (hole in the white)
    draw.polygon([(470, 470), (660, 560), (520, 700)], fill="#8e3fa8")  # wing
    draw.polygon([(560, 700), (700, 860), (600, 880)], fill="#2f6fb0")  # tail

    # Thin branch: narrow enough that most of it cannot hold a number.
    draw.line([(120, 900), (780, 820)], fill="#6b4a2f", width=14)
    for cx, cy in ((240, 870), (430, 848), (620, 830)):
        draw.ellipse([cx - 34, cy - 20, cx + 34, cy + 20], fill="#5f9e4a")

    # Slight blur, then film-grain noise, to imitate a real camera capture.
    img = img.filter(ImageFilter.GaussianBlur(0.6))
    arr = np.asarray(img, dtype=np.float64)
    rng = np.random.default_rng(7)
    arr = np.clip(arr + rng.normal(0.0, 4.5, arr.shape), 0, 255)
    return Image.fromarray(arr.astype(np.uint8), "RGB")


if __name__ == "__main__":
    out = sys.argv[1] if len(sys.argv) > 1 else "sample.png"
    build().save(out)
    print(f"wrote {out}")
