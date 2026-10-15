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

from pathlib import Path
from typing import Annotated

from fastapi import FastAPI, File, Form, HTTPException, Query, UploadFile
from fastapi.responses import FileResponse, Response
from pydantic import BaseModel, Field

from ..core.color.mixing import predict_mix_color
from ..core.critique import critique_attempt
from ..core.imaging.loader import (
    MAX_BYTES,
    ImageTooLargeError,
    UnsupportedImageError,
    load_image,
)
from ..core.paints import STARTER_ACRYLICS, catalog_by_id
from ..core.pipeline import PIPELINE_VERSION, PatternParams, generate_pattern
from ..core.recipes import recipes_for_pattern
from ..core.render.calibration import render_calibration_card_svg
from ..core.render.grid import render_grid_svg
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


class MixOut(BaseModel):
    hex: str
    lab: list[float]
    instructions: str


class GridOut(BaseModel):
    columns: int
    rows: int
    cells: int
    square_enough: bool = Field(
        description="False means the image's proportions do not divide into square cells"
    )
    cell_mm_a4: float = Field(description="Millimetres per cell when scaled to A4's long edge")
    svg: str


class MeasurementOut(BaseModel):
    key: str
    value: float
    unit: str
    explanation: str


class CritiqueOut(BaseModel):
    message: str
    next_step: str | None
    measurements: list[MeasurementOut]


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
_STATIC = Path(__file__).parent / "static"


@app.get("/", include_in_schema=False)
def index() -> FileResponse:
    """The web app itself."""
    return FileResponse(_STATIC / "index.html")


@app.get("/calibration-card.svg", include_in_schema=False)
def calibration_card() -> Response:
    """The printable grey reference card, generated on request."""
    svg, _ = render_calibration_card_svg()
    return Response(
        content=svg,
        media_type="image/svg+xml",
        headers={"Content-Disposition": 'inline; filename="artmentor-calibration-card.svg"'},
    )


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


@app.get("/mix", response_model=MixOut)
def mix(
    paints: Annotated[str, Query(description="Comma-separated paint ids, 1-4")],
    parts: Annotated[str, Query(description="Comma-separated parts, matching paints")],
) -> MixOut:
    """Predict the colour of a mix. Cheap enough to call on every slider move.

    A GET with no body, because this exists so a client can show the result of
    mixing update as the user drags a ratio — the point is immediacy, and the
    forward engine answers in well under a millisecond once spectra are cached.
    """
    chosen = _resolve_palette(paints)
    if not 1 <= len(chosen) <= 4:
        raise HTTPException(status_code=400, detail="Mix between 1 and 4 paints.")

    try:
        amounts = [float(v) for v in parts.split(",")]
    except ValueError as exc:
        raise HTTPException(status_code=400, detail="parts must be numbers.") from exc
    if len(amounts) != len(chosen):
        raise HTTPException(status_code=400, detail="parts must match paints in length.")
    if any(a < 0 for a in amounts) or sum(amounts) <= 0:
        raise HTTPException(status_code=400, detail="parts must be non-negative and not all zero.")

    result = predict_mix_color(chosen, amounts)
    return MixOut(
        hex=result.hex,
        lab=[round(result.lab.L, 2), round(result.lab.a, 2), round(result.lab.b, 2)],
        instructions=result.describe(),
    )


@app.post("/grid", response_model=GridOut)
def grid(
    image: Annotated[UploadFile, File(description="The reference photograph")],
    divisions: Annotated[int, Query(ge=2, le=40, description="Cells along the long edge")] = 6,
    labels: Annotated[bool, Query()] = True,
    diagonals: Annotated[bool, Query()] = False,
    max_dimension: Annotated[int, Query(ge=200, le=2000)] = 1000,
) -> GridOut:
    """Overlay a grid on a reference, for the grid drawing method.

    `cell_mm_a4` is the number an artist actually uses: how far apart to make
    pencil marks once the reference is scaled onto A4. Pixels are no use at a
    drawing board, and converting in your head is where the proportions go wrong.
    """
    loaded = _decode(_read_upload(image), max_dimension)
    svg, plan = render_grid_svg(
        loaded.rgb, divisions=divisions, labels=labels, diagonals=diagonals
    )
    return GridOut(
        columns=plan.columns,
        rows=plan.rows,
        cells=plan.columns * plan.rows,
        square_enough=plan.is_square_enough,
        cell_mm_a4=round(plan.cell_size_mm(297), 1),
        svg=svg,
    )


@app.post("/critique", response_model=CritiqueOut)
def critique(
    reference: Annotated[UploadFile, File(description="What they were copying")],
    attempt: Annotated[UploadFile, File(description="Photo of their painting")],
    max_dimension: Annotated[int, Query(ge=200, le=2000)] = 900,
) -> CritiqueOut:
    """Measure how closely a painting follows its reference.

    This reports fidelity, never merit. There is no score and no aesthetic
    judgement — see `core/critique.py` for why that distinction is load-bearing
    rather than cautious.
    """
    ref = _decode(_read_upload(reference), max_dimension)
    att = _decode(_read_upload(attempt), max_dimension)

    result = critique_attempt(ref.rgb, att.rgb)
    return CritiqueOut(
        message=result.message,
        next_step=result.suggested_next_step,
        measurements=[
            MeasurementOut(key=m.key, value=m.value, unit=m.unit, explanation=m.explanation)
            for m in result.measurements
        ],
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