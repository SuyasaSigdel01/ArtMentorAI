"""Command-line entry point.

    python -m artmentor.cli photo.jpg -o out/ --colors 12

Exists so the core pipeline can be exercised end to end with no server, no
database and no frontend — which is the whole point of keeping `core` free of
framework dependencies. If this stops working, the boundary has been broken.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

from .core.imaging.loader import ImageTooLargeError, UnsupportedImageError, load_image
from .core.pipeline import PatternParams, generate_pattern
from .core.render.svg import render_pattern_svg


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="artmentor",
        description="Convert an image into a numbered paint-by-numbers pattern.",
    )
    parser.add_argument("image", type=Path, help="input image (JPEG, PNG, or WEBP)")
    parser.add_argument("-o", "--out", type=Path, default=Path("out"), help="output directory")
    parser.add_argument("-c", "--colors", type=int, default=12, help="palette size")
    parser.add_argument(
        "--max-dimension", type=int, default=900, help="longest side before processing"
    )
    parser.add_argument(
        "--min-area", type=int, default=90, help="smallest paintable region, in pixels"
    )
    parser.add_argument("--smoothing", type=int, default=5, help="majority filter kernel")
    parser.add_argument("--seed", type=int, default=0, help="k-means seed")
    parser.add_argument("--no-legend", action="store_true", help="omit the color key")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    try:
        loaded = load_image(args.image, max_dimension=args.max_dimension)
    except (UnsupportedImageError, ImageTooLargeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    params = PatternParams(
        n_colors=args.colors,
        max_dimension=args.max_dimension,
        min_region_area=args.min_area,
        label_smoothing=args.smoothing,
        seed=args.seed,
    )
    pattern = generate_pattern(loaded.rgb, params)

    out: Path = args.out
    out.mkdir(parents=True, exist_ok=True)
    legend = not args.no_legend

    (out / "pattern.svg").write_text(
        render_pattern_svg(pattern, filled=False, legend=legend), encoding="utf-8"
    )
    (out / "preview.svg").write_text(
        render_pattern_svg(pattern, filled=True, show_numbers=False, legend=legend),
        encoding="utf-8",
    )

    # Flat raster of the quantized result, handy for eyeballing palette quality.
    palette_rgb = np.array([[e.rgb.r, e.rgb.g, e.rgb.b] for e in pattern.palette], dtype=np.uint8)
    quantized_rgb = palette_rgb[pattern.label_map]
    cv2.imwrite(str(out / "quantized.png"), cv2.cvtColor(quantized_rgb, cv2.COLOR_RGB2BGR))

    (out / "palette.json").write_text(
        json.dumps(
            [
                {
                    "number": e.number,
                    "hex": e.rgb.hex,
                    "lab": [round(e.lab.L, 2), round(e.lab.a, 2), round(e.lab.b, 2)],
                    "pixels": e.pixel_count,
                    "share": round(e.share, 4),
                }
                for e in pattern.palette
            ],
            indent=2,
        ),
        encoding="utf-8",
    )
    (out / "summary.json").write_text(json.dumps(pattern.summary(), indent=2), encoding="utf-8")

    summary = pattern.summary()
    print(f"source      : {args.image} ({loaded.source_format}, {loaded.original_size[0]}x{loaded.original_size[1]})")
    print(f"pattern     : {summary['size'][0]}x{summary['size'][1]}")
    print(f"colors      : {summary['colors']}")
    print(f"regions     : {summary['regions']} ({summary['labelable_regions']} can hold a number)")
    print(f"written to  : {out}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
