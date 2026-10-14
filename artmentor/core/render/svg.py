"""SVG output for patterns.

SVG rather than PNG, deliberately. The format choice pays for itself three times:
it prints at any size without pixelation, the files are a fraction of the size of
a raster equivalent, and every region arrives in the browser as an addressable
DOM element — which is what makes an interactive "tap a region to see its color
mix" gallery possible later without re-architecting anything.
"""

from __future__ import annotations

from xml.sax.saxutils import escape

import numpy as np

from ..types import Pattern

__all__ = ["render_pattern_svg"]

_FONT = "Helvetica, Arial, sans-serif"


def _path_data(polygons: list[np.ndarray]) -> str:
    """Build one path covering a region's outer contour and any holes.

    After an initial moveto, bare coordinate pairs are implicit linetos, which
    keeps these files small — and they get large fast at a few thousand regions.
    """
    parts: list[str] = []
    for polygon in polygons:
        # One decimal place: enough for smoothed sub-pixel vertices to render
        # cleanly, while keeping the file roughly a third smaller than full floats.
        coords = " ".join(f"{x:.1f},{y:.1f}" for x, y in polygon)
        parts.append(f"M{coords}Z")
    return "".join(parts)


def _text_color_for(lightness: float) -> str:
    """Black on light swatches, white on dark ones."""
    return "#111111" if lightness > 55.0 else "#ffffff"


def _legend_svg(pattern: Pattern, width: int, *, swatch: int = 34, gap: int = 7) -> tuple[str, int]:
    """Render the color key. Returns (markup, height consumed)."""
    pad = 10
    cell = swatch + gap
    per_row = max(1, (width - 2 * pad + gap) // cell)
    rows = (len(pattern.palette) + per_row - 1) // per_row
    row_height = swatch + 16
    height = pad + rows * row_height + pad

    parts = [f'<rect x="0" y="0" width="{width}" height="{height}" fill="#ffffff"/>']
    for i, entry in enumerate(pattern.palette):
        row, col = divmod(i, per_row)
        x = pad + col * cell
        y = pad + row * row_height
        parts.append(
            f'<rect x="{x}" y="{y}" width="{swatch}" height="{swatch}" rx="4" '
            f'fill="{entry.rgb.hex}" stroke="#999999" stroke-width="0.7"/>'
        )
        parts.append(
            f'<text x="{x + swatch / 2:.1f}" y="{y + swatch / 2:.1f}" '
            f'font-family="{_FONT}" font-size="{swatch * 0.46:.1f}" font-weight="600" '
            f'fill="{_text_color_for(entry.lab.L)}" text-anchor="middle" '
            f'dominant-baseline="central">{entry.number}</text>'
        )
        parts.append(
            f'<text x="{x + swatch / 2:.1f}" y="{y + swatch + 11:.1f}" '
            f'font-family="{_FONT}" font-size="8" fill="#666666" '
            f'text-anchor="middle">{entry.share * 100:.0f}%</text>'
        )
    return "\n".join(parts), height


def render_pattern_svg(
    pattern: Pattern,
    *,
    filled: bool = False,
    show_numbers: bool = True,
    legend: bool = True,
    stroke: str = "#444444",
    stroke_width: float = 0.5,
    min_label_radius: float = 3.0,
    max_font_size: float = 14.0,
) -> str:
    """Render a pattern to an SVG document.

    Args:
        filled: fill each region with its color (a preview) rather than leaving
            it white (the printable pattern the learner actually paints on).
        show_numbers: print the palette number inside each region large enough
            to hold one.
        legend: include the color key strip at the top.
    """
    width, height = pattern.width, pattern.height

    legend_markup, legend_height = ("", 0)
    if legend and pattern.palette:
        legend_markup, legend_height = _legend_svg(pattern, width)

    total_height = height + legend_height

    palette_hex = [entry.rgb.hex for entry in pattern.palette]

    body: list[str] = []
    for region in pattern.regions:
        if not region.contours:
            continue
        fill = palette_hex[region.palette_index] if filled else "#ffffff"
        # data-n carries the palette number onto every path. The whole reason
        # this renderer emits SVG rather than a raster is that each region
        # arrives in the browser as an addressable element; without an
        # identifier on it, that advantage goes unused. A client can now
        # highlight every region of one colour, which is what makes a mixing
        # guide feel connected to the picture.
        body.append(
            f'<path data-n="{pattern.palette[region.palette_index].number}" '
            f'd="{_path_data(region.contours)}" fill="{fill}" '
            f'fill-rule="evenodd" stroke="{stroke}" stroke-width="{stroke_width}"/>'
        )

    numbers: list[str] = []
    if show_numbers:
        for region in pattern.regions:
            if region.label_point is None or region.label_radius < min_label_radius:
                continue
            x, y = region.label_point
            # Font size follows the room actually available inside the region, so
            # digits never spill across an outline.
            size = min(max(region.label_radius * 1.05, 3.5), max_font_size)
            entry = pattern.palette[region.palette_index]
            color = _text_color_for(entry.lab.L) if filled else "#555555"
            numbers.append(
                f'<text x="{x}" y="{y}" font-family="{_FONT}" font-size="{size:.1f}" '
                f'fill="{color}" text-anchor="middle" dominant-baseline="central">'
                f"{escape(entry.label)}</text>"
            )

    return "\n".join(
        [
            '<?xml version="1.0" encoding="UTF-8"?>',
            (
                f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" '
                f'height="{total_height}" viewBox="0 0 {width} {total_height}">'
            ),
            f'<rect width="{width}" height="{total_height}" fill="#ffffff"/>',
            legend_markup,
            f'<g transform="translate(0,{legend_height})">',
            "\n".join(body),
            "\n".join(numbers),
            "</g>",
            "</svg>",
        ]
    )
