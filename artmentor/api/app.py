"""FastAPI application.

Endpoints:
    GET  /health        liveness
    GET  /paints        the paint catalog a client can offer as "what do you own?"
    POST /patterns      image -> numbered pattern, SVG, and mixing recipes
    POST /shade-check   photo of a mix + target color -> verdict and correction

Two deliberate limits, both honest rather than hidden:

**Uploads are read with a hard cap before decoding.** `core.imaging.loader` is
the security boundary and does the real validation — magic bytes, pixel caps,
EXIF stripping — but it can only do that once the bytes are in memory, so the
cap here stops a large body from being buffered in the first place.

**Work runs in FastAPI's threadpool, not a job queue.** Endpoints are declared
`def` rather than `async def`, so Starlette runs them in a worker thread and the
event loop keeps serving. That is enough for development and light use, and it
is *not* enough for production: there is no timeout, no memory ceiling, and no
backpressure, so a flood of large images will exhaust the pool. The architecture
notes call for Celery or RQ before any public deployment. Treat this module as
correct but not yet hardened for load.
"""

from __future__ import annotations

from typing import Annotated

from fastapi import FastAPI, File, Form, HTTPException, Query, UploadFile
from pydantic import BaseModel, Field

from ..core.imaging.loader import (
    MAX_BYTES,
    ImageTooLargeError,
    UnsupportedImageError,
    load_image,
)
from ..core.paints import STARTER_ACRYLICS, catalog_by_id
from ..core.pipeline import PIPELINE_VERSION, PatternParams, generate_pattern
from ..core.recipes import recipes_for_pattern
from ..core.render.svg import render_pattern_svg
from ..core.swatch import check_shade
from ..core.types import RGBColor

app = FastAPI(
    title="ArtMentorAI",
    version=PIPELINE_VERSION,
    summary="Turn photographs into paintable patterns, and check mixed colors.",
)

CATALOG = catalog_by_id()


# ---------------------------------------------------------------------------
# Response models
# ---------------------------------------------------------------------------
class PaintOut(BaseModel):
    id: str
    name: str
    brand: str
    hex: str
    tinting_strength: float


class MixStep(BaseModel):
    paint: str
    paint_id: str
    parts: float


class PaletteColorOut(BaseModel):
    number: int
    hex: str
    share: float = Field(description="Fraction of the image this color covers, 0..1")
    mix: list[MixStep] | None = None
    instructions: str | None = None
    delta_e: float | None = Field(
        default=None, description="CIEDE2000 between the mixable recipe and the target"
    )
    achievable: bool | None = Field(
        default=None, description="False means this is the closest the palette reaches"
    )


class PatternOut(BaseModel):
    width: int
    height: int
    colors: int
    regions: int
    pipeline_version: str
    palette: list[PaletteColorOut]
    svg: str


class ShadeCheckOut(BaseModel):
    matched: bool
    delta_e: float
    band: str
    confidence: str
    hint: str
    message: str
    measured_hex: str
    target_hex: str
    uniform: bool


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _read_upload(upload: UploadFile) -> bytes:
    """Read an upload, refusing anything oversized before it is all in memory."""
    data = upload.file.read(MAX_BYTES + 1)
    if len(data) > MAX_BYTES:
        raise HTTPException(
            status_code=413, detail=f"Image exceeds the {MAX_BYTES // (1024 * 1024)}MB limit."
        )
    if not data:
        raise HTTPException(status_code=400, detail="Empty upload.")
    return data


def _decode(data: bytes, max_dimension: int):
    """Hand bytes to the hardened loader and translate its errors into HTTP."""
    try:
        return load_image(data, max_dimension=max_dimension)
    except ImageTooLargeError as exc:
        raise HTTPException(status_code=413, detail=str(exc)) from exc
    except UnsupportedImageError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


def _resolve_palette(paint_ids: str | None):
    """Which paints the learner owns. Defaults to the full starter set."""
    if not paint_ids:
        return list(STARTER_ACRYLICS)

    wanted = [p.strip() for p in paint_ids.split(",") if p.strip()]
    unknown = [p for p in wanted if p not in CATALOG]
    if unknown:
        raise HTTPException(status_code=400, detail=f"Unknown paint ids: {', '.join(unknown)}")
    if len(wanted) < 1:
        raise HTTPException(status_code=400, detail="At least one paint is required.")
    return [CATALOG[p] for p in wanted]


def _parse_box(spec: str | None, field: str) -> tuple[slice, slice] | None:
    """Parse 'x0,y0,x1,y1' as fractions of the image into array slices.

    Fractions rather than pixels, so a client can send the same coordinates
    regardless of what the image was downscaled to on the way in.
    """
    if not spec:
        return None
    parts = spec.split(",")
    if len(parts) != 4:
        raise HTTPException(
            status_code=400, detail=f"{field} must be 'x0,y0,x1,y1' as fractions of the image."
        )
    try:
        x0, y0, x1, y1 = (float(p) for p in parts)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=f"{field} values must be numbers.") from exc

    if not all(0.0 <= v <= 1.0 for v in (x0, y0, x1, y1)):
        raise HTTPException(status_code=400, detail=f"{field} values must be between 0 and 1.")
    if x1 <= x0 or y1 <= y0:
        raise HTTPException(status_code=400, detail=f"{field} must have x1 > x0 and y1 > y0.")

    # Stored as fractions; turned into concrete slices once the size is known.
    return (slice(y0, y1), slice(x0, x1))  # type: ignore[arg-type]


def _to_pixel_box(box, shape: tuple[int, int]) -> tuple[slice, slice] | None:
    if box is None:
        return None
    h, w = shape
    ys, xs = box
    return (
        slice(int(ys.start * h), max(int(ys.stop * h), int(ys.start * h) + 1)),
        slice(int(xs.start * w), max(int(xs.stop * w), int(xs.start * w) + 1)),
    )


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
@app.get("/health")
def health() -> dict:
    return {"status": "ok", "pipeline_version": PIPELINE_VERSION}


@app.get("/paints", response_model=list[PaintOut])
def list_paints() -> list[PaintOut]:
    """The starter catalog, for a client to render as "which do you own?".

    These colors are approximations pending measured paint data — see
    `core/paints.py`. A client should not present them as exact.
    """
    from ..core.color.space import lab_to_rgb

    return [
        PaintOut(
            id=p.id,
            name=p.name,
            brand=p.brand,
            hex=lab_to_rgb(p.lab).hex,
            tinting_strength=p.tinting_strength,
        )
        for p in STARTER_ACRYLICS
    ]


@app.post("/patterns", response_model=PatternOut)
def create_pattern(
    image: Annotated[UploadFile, File(description="JPEG, PNG, or WEBP")],
    colors: Annotated[int, Query(ge=2, le=24)] = 12,
    max_dimension: Annotated[int, Query(ge=200, le=2000)] = 900,
    min_region_area: Annotated[int, Query(ge=10, le=5000)] = 90,
    include_recipes: Annotated[bool, Query()] = True,
    max_components: Annotated[int, Query(ge=1, le=3)] = 3,
    paints: Annotated[str | None, Query(description="Comma-separated paint ids")] = None,
    filled: Annotated[bool, Query(description="Color the SVG instead of outlines")] = False,
) -> PatternOut:
    """Turn an uploaded photograph into a numbered, paintable pattern."""
    data = _read_upload(image)
    loaded = _decode(data, max_dimension)

    pattern = generate_pattern(
        loaded.rgb,
        PatternParams(
            n_colors=colors,
            max_dimension=max_dimension,
            min_region_area=min_region_area,
        ),
    )

    recipes = None
    if include_recipes:
        recipes = recipes_for_pattern(
            pattern, _resolve_palette(paints), max_components=max_components
        )

    palette: list[PaletteColorOut] = []
    for index, entry in enumerate(pattern.palette):
        item = PaletteColorOut(number=entry.number, hex=entry.rgb.hex, share=entry.share)
        if recipes is not None:
            recipe = recipes[index]
            item.mix = [
                MixStep(paint=c.paint.name, paint_id=c.paint.id, parts=c.parts)
                for c in recipe.recipe.components
            ]
            item.instructions = recipe.recipe.describe()
            item.delta_e = round(recipe.recipe.delta_e, 2)
            item.achievable = recipe.is_achievable
        palette.append(item)

    return PatternOut(
        width=pattern.width,
        height=pattern.height,
        colors=len(pattern.palette),
        regions=len(pattern.regions),
        pipeline_version=pattern.pipeline_version,
        palette=palette,
        svg=render_pattern_svg(pattern, filled=filled),
    )


@app.post("/shade-check", response_model=ShadeCheckOut)
def shade_check(
    image: Annotated[UploadFile, File(description="Photo of the mixed paint")],
    target: Annotated[str, Form(description="Target color as hex, e.g. #4a7a3a")],
    swatch: Annotated[str | None, Form(description="'x0,y0,x1,y1' fractions")] = None,
    reference: Annotated[
        str | None, Form(description="'x0,y0,x1,y1' fractions of a grey card")
    ] = None,
) -> ShadeCheckOut:
    """Judge a photographed paint mix and say which way to correct it.

    Include a mid-grey card in the frame and pass its box as `reference` for an
    accurate read. Without one the response still works, but `confidence` drops
    to "estimated" and the message says so — a client should not present an
    uncalibrated reading as authoritative.
    """
    try:
        target_rgb = RGBColor.from_hex(target)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=f"Invalid target color: {target!r}") from exc

    from ..core.color.space import rgb_to_lab

    data = _read_upload(image)
    loaded = _decode(data, 1200)

    shape = loaded.rgb.shape[:2]
    result = check_shade(
        loaded.rgb,
        rgb_to_lab(target_rgb),
        swatch_region=_to_pixel_box(_parse_box(swatch, "swatch"), shape),
        white_region=_to_pixel_box(_parse_box(reference, "reference"), shape),
    )

    return ShadeCheckOut(
        matched=result.is_match,
        delta_e=round(result.delta_e, 2),
        band=result.verdict.band,
        confidence=result.confidence,
        hint=result.hint,
        message=result.message(),
        measured_hex=result.measured.rgb.hex,
        target_hex=target_rgb.hex,
        uniform=result.measured.is_uniform,
    )