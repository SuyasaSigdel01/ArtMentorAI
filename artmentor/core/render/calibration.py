"""A printable calibration card.

Photographing paint is the only way a learner can ask "is this the right
colour?", and a camera reports light rather than pigment. A known reference in
the frame is what makes the answer trustworthy — see `core/swatch.py` for the
correction itself. This module prints the reference.

Design decisions, each learned the hard way:

**The reference patch is grey, not white.** White sits near the sensor's
ceiling, so warm light pushes its red channel past 255 and it clips. The
measured reference is then wrong in exactly the way the correction exists to
fix. Measured on simulated tungsten: a white patch took a 5.3 dE error down to
only 4.6, where a mid-grey patch took the same error to 0.9.

**There is a step wedge, not a single patch.** Five greys make clipping visible
to the human holding the card: if the top two squares look identical in the
photo, the shot is overexposed and the reading cannot be trusted. One patch
alone fails silently.

**The card must be printed in greyscale.** A colour printer builds grey from
cyan, magenta and yellow inks, and that mixture is usually not neutral — which
destroys the one property the reference needs. Black ink alone is close enough
to neutral to work. The instruction is printed on the card itself, because a
user who reads it once in an app will not remember it at the sink.

**It is not a colorimetric standard.** Home printers, inks and papers vary, so
absolute reflectance is unknown. The correction only needs the patch to be
*neutral* — equal across the channels — not to be a specific lightness, which
is a far weaker requirement and one a greyscale print can meet.
"""

from __future__ import annotations

from dataclasses import dataclass

__all__ = ["CalibrationCard", "render_calibration_card_svg"]

_FONT = "Helvetica, Arial, sans-serif"

# Patch levels as 8-bit grey. The reference sits at 180 rather than brighter:
# bright enough to carry signal, dark enough that a warm cast (roughly x1.2 in
# red) still lands under 255 instead of clipping.
_REFERENCE_LEVEL = 180
_WEDGE_LEVELS = (230, 205, 180, 130, 70)


@dataclass(frozen=True)
class CalibrationCard:
    """Geometry of a rendered card, in millimetres from the top-left corner.

    Returned alongside the SVG so a caller can describe where things are without
    re-deriving the layout, and so tests can assert on it.
    """

    width_mm: float
    height_mm: float
    reference_box_mm: tuple[float, float, float, float]  # x, y, w, h
    swatch_box_mm: tuple[float, float, float, float]
    reference_level: int

    def reference_fraction(self) -> tuple[float, float, float, float]:
        """The reference patch as x0, y0, x1, y1 fractions of the card.

        Matches the box format the API's `/shade-check` endpoint accepts, for
        the case where a photo is cropped to the card.
        """
        x, y, w, h = self.reference_box_mm
        return (
            x / self.width_mm,
            y / self.height_mm,
            (x + w) / self.width_mm,
            (y + h) / self.height_mm,
        )


def _grey(level: int) -> str:
    return f"#{level:02x}{level:02x}{level:02x}"


def _text(x: float, y: float, content: str, size: float, *, fill: str = "#333333",
          anchor: str = "start", weight: str = "normal") -> str:
    return (
        f'<text x="{x:.1f}" y="{y:.1f}" font-family="{_FONT}" font-size="{size:.1f}" '
        f'fill="{fill}" text-anchor="{anchor}" font-weight="{weight}">{content}</text>'
    )


def render_calibration_card_svg(
    *,
    width_mm: float = 180.0,
    height_mm: float = 120.0,
) -> tuple[str, CalibrationCard]:
    """Render a printable calibration card.

    Returns the SVG and a `CalibrationCard` describing where the patches sit.

    Dimensions are in millimetres so the print comes out at a known physical
    size. The default fits comfortably on A4 or Letter with room to trim.
    """
    if width_mm < 100 or height_mm < 70:
        raise ValueError("card must be at least 100mm x 70mm to hold legible patches")

    margin = 10.0
    parts: list[str] = [
        f'<rect x="0" y="0" width="{width_mm}" height="{height_mm}" fill="#ffffff"/>',
        # Cut line, so the card can be trimmed to a known size.
        (
            f'<rect x="1" y="1" width="{width_mm - 2}" height="{height_mm - 2}" '
            f'fill="none" stroke="#bbbbbb" stroke-width="0.3" stroke-dasharray="2 2"/>'
        ),
    ]

    y = margin + 5
    parts.append(_text(margin, y, "ArtMentorAI colour reference", 6.5, fill="#111111", weight="600"))
    y += 5
    # Black, not red. The card is meant to be printed in greyscale, where red
    # becomes a mid-grey and the most important instruction on the page loses
    # its emphasis — on a card whose entire purpose is being printed that way.
    parts.append(
        _text(margin, y, "PRINT IN GREYSCALE / BLACK AND WHITE. Do not scale to fit.", 3.8,
              fill="#000000", weight="700")
    )
    y += 7

    # --- the step wedge -----------------------------------------------------
    wedge_h = 16.0
    gap = 3.0
    patch_w = (width_mm - 2 * margin - gap * (len(_WEDGE_LEVELS) - 1)) / len(_WEDGE_LEVELS)
    reference_box = (0.0, 0.0, 0.0, 0.0)

    for i, level in enumerate(_WEDGE_LEVELS):
        x = margin + i * (patch_w + gap)
        is_reference = level == _REFERENCE_LEVEL
        parts.append(
            f'<rect x="{x:.1f}" y="{y:.1f}" width="{patch_w:.1f}" height="{wedge_h:.1f}" '
            f'fill="{_grey(level)}" stroke="{"#222222" if is_reference else "#cccccc"}" '
            f'stroke-width="{0.8 if is_reference else 0.2}"/>'
        )
        if is_reference:
            reference_box = (x, y, patch_w, wedge_h)
            parts.append(
                _text(x + patch_w / 2, y + wedge_h + 4, "REFERENCE", 3.4,
                      fill="#111111", anchor="middle", weight="600")
            )

    y += wedge_h + 8
    parts.append(
        _text(margin, y, "If the two lightest squares look identical in your photo, "
                         "it is overexposed — move to softer light.", 3.2, fill="#777777")
    )

    # --- where the paint goes ----------------------------------------------
    y += 6
    swatch = min(34.0, height_mm - y - margin - 6)
    swatch_x = margin
    parts.append(
        f'<rect x="{swatch_x:.1f}" y="{y:.1f}" width="{swatch:.1f}" height="{swatch:.1f}" '
        f'fill="#ffffff" stroke="#333333" stroke-width="0.6" stroke-dasharray="1.5 1.5"/>'
    )
    parts.append(
        _text(swatch_x + swatch / 2, y + swatch / 2, "PAINT", 4.0,
              fill="#999999", anchor="middle", weight="600")
    )

    # --- instructions -------------------------------------------------------
    text_x = swatch_x + swatch + 8
    ty = y + 5
    for line in (
        "1. Print this card in greyscale on matte paper.",
        "2. Fill the dashed square with your mixed paint.",
        "3. Let it dry — wet paint is glossy and reads too light.",
        "4. Photograph the whole card flat, in even light.",
        "5. Avoid direct sun and your own shadow.",
    ):
        parts.append(_text(text_x, ty, line, 3.5, fill="#444444"))
        ty += 5.2

    # --- corner marks -------------------------------------------------------
    # Plain registration marks. Nothing reads them yet; they are here so that
    # automatic card detection can be added later without reprinting every card
    # already in users' hands.
    arm = 5.0
    for cx, cy in (
        (margin / 2, margin / 2),
        (width_mm - margin / 2, margin / 2),
        (margin / 2, height_mm - margin / 2),
        (width_mm - margin / 2, height_mm - margin / 2),
    ):
        parts.append(
            f'<path d="M{cx - arm / 2:.1f},{cy:.1f} L{cx + arm / 2:.1f},{cy:.1f} '
            f'M{cx:.1f},{cy - arm / 2:.1f} L{cx:.1f},{cy + arm / 2:.1f}" '
            f'stroke="#000000" stroke-width="0.4"/>'
        )

    svg = "\n".join(
        [
            '<?xml version="1.0" encoding="UTF-8"?>',
            (
                f'<svg xmlns="http://www.w3.org/2000/svg" '
                f'width="{width_mm}mm" height="{height_mm}mm" '
                f'viewBox="0 0 {width_mm} {height_mm}">'
            ),
            *parts,
            "</svg>",
        ]
    )

    return svg, CalibrationCard(
        width_mm=width_mm,
        height_mm=height_mm,
        reference_box_mm=reference_box,
        swatch_box_mm=(swatch_x, y, swatch, swatch),
        reference_level=_REFERENCE_LEVEL,
    )