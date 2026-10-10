"""A starter paint catalog.

READ THIS BEFORE TRUSTING THESE NUMBERS.

Every color below is an *approximation* eyeballed from typical student-grade
acrylics, not a measurement. They are good enough to develop and test the mixing
engine against, and not good enough to give a learner advice about their actual
tubes.

Replacing them is a real task with a real method, and it is on the roadmap:
photograph each tube's swatch under controlled lighting alongside a calibrated
reference card, solve for the color transform, and record the resulting Lab
value. Until that happens, treat every mixing result as directionally useful and
numerically provisional, and do not ship mixing advice to users as if it were
measured.

`tinting_strength` is how strongly a paint dominates a mixture — the painter's
own term, and the correction that keeps single-constant Kubelka-Munk honest (see
`color/mixing.py`). Values here are relative, with an ordinary opaque pigment at
1.0. Titanium white and the phthalos are the notable outliers.
"""

from __future__ import annotations

from .color.space import hex_to_lab
from .interfaces import Paint

__all__ = ["STARTER_ACRYLICS", "catalog_by_id", "make_paint"]


def make_paint(
    paint_id: str,
    name: str,
    hex_color: str,
    *,
    brand: str = "generic student acrylic",
    opacity: float = 1.0,
    tinting_strength: float = 1.0,
) -> Paint:
    """Build a Paint from a hex color, converting to Lab once at definition time."""
    return Paint(
        id=paint_id,
        name=name,
        brand=brand,
        lab=hex_to_lab(hex_color),
        opacity=opacity,
        tinting_strength=tinting_strength,
    )


# A twelve-tube starter set: the pigments a beginner's first box actually
# contains, spanning warm and cool versions of each primary so that mixing
# exercises can teach why that distinction matters.
STARTER_ACRYLICS: tuple[Paint, ...] = (
    # Titanium white scatters light far more than it absorbs, so it dominates
    # mixtures out of all proportion to its volume. The high strength here is
    # what stops the model from barely lightening a mix at all.
    make_paint("ti-white", "Titanium White", "#fdfdfa", tinting_strength=6.0),
    make_paint("ivory-black", "Ivory Black", "#1b1a17", tinting_strength=3.0),
    # Yellows: warm (cadmium) and cool (lemon).
    make_paint("cad-yellow", "Cadmium Yellow Medium", "#ffd800", tinting_strength=1.2),
    make_paint("lemon-yellow", "Lemon Yellow", "#f5f04a", tinting_strength=1.0),
    make_paint("yellow-ochre", "Yellow Ochre", "#c9a227", tinting_strength=1.0),
    # Reds: warm (cadmium) and cool (crimson).
    make_paint("cad-red", "Cadmium Red Medium", "#e03c31", tinting_strength=1.3),
    make_paint("crimson", "Alizarin Crimson", "#9b1b30", tinting_strength=1.5),
    # Blues: warm (ultramarine, violet-leaning) and cool (phthalo green-shade,
    # which leans cyan — that difference is the whole reason both are in the box,
    # and it is why ultramarine mixes dull olives while phthalo mixes vivid
    # greens). Phthalo is also notoriously overpowering: a speck of it colors a
    # whole pile of white.
    make_paint("ultramarine", "Ultramarine Blue", "#2e3192", tinting_strength=1.6),
    make_paint("phthalo-blue", "Phthalo Blue (green shade)", "#005b96", tinting_strength=4.0),
    make_paint("cerulean", "Cerulean Blue", "#2a6f97", tinting_strength=1.1),
    # Earths: the colors beginners reach for and then cannot mix back.
    make_paint("burnt-umber", "Burnt Umber", "#6b3f23", tinting_strength=1.4),
    make_paint("burnt-sienna", "Burnt Sienna", "#8a3324", tinting_strength=1.3),
)


def catalog_by_id(paints=STARTER_ACRYLICS) -> dict[str, Paint]:
    """Index a paint collection by id, for lookup from stored recipes."""
    return {paint.id: paint for paint in paints}
