"""FastAPI application.

Endpoints:
    GET  /health        liveness
    GET  /paints        the paint catalog a client can offer as "what do you own?"
    POST /patterns      image -> numbered pattern, SVG, and mixing recipes
    POST /grid          image -> grid overlay for the grid drawing method
    POST /critique      reference + attempt -> fidelity measurements
    POST /shade-check   photo of a mix + target color -> verdict and correction
    *    /gallery/...   saved pieces, attempts over time, and progress

The gallery routes are the only stateful ones. Everything else is a pure
function of its request, which is why the rest of this file can be read in any
order; the gallery section carries its own notes on where its state lives.

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

import json
import os
from pathlib import Path
from typing import Annotated, Any

from fastapi import Depends, FastAPI, File, Form, HTTPException, Query, UploadFile
from fastapi import Path as PathParam
from fastapi.responses import FileResponse, Response
from pydantic import BaseModel, Field

from ..core.color.mixing import predict_mix_color
from ..core.critique import critique_attempt
from ..core.imaging.loader import (
    MAX_BYTES,
    ImageTooLargeError,
    UnsupportedImageError,
    encode_image,
    load_image,
)
from ..core.mediums import LEVELS, MEDIUMS, describe_mix, describe_value, guidance_for
from ..core.paints import STARTER_ACRYLICS, catalog_by_id
from ..core.pipeline import PIPELINE_VERSION, PatternParams, generate_pattern
from ..core.recipes import recipes_for_pattern
from ..core.render.calibration import render_calibration_card_svg
from ..core.render.loomis import render_loomis_sheet_svg
from ..core.render.grid import render_grid_svg
from ..core.render.svg import render_pattern_svg
from ..core.swatch import check_shade
from ..core.types import RGBColor
from ..store import Attempt, Gallery, GalleryError, NotFound, Piece, Progress

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
    medium: str
    level: str
    notes: list[str] = Field(
        default_factory=list, description="Guidance specific to this medium and level"
    )
    palette: list[PaletteColorOut]
    svg: str


class OptionOut(BaseModel):
    id: str
    name: str


class GuidanceOut(BaseModel):
    medium: str
    level: str
    colors: int
    max_components: int
    min_region_area: int
    excluded_paint_ids: list[str]
    target_delta_e: float = Field(
        description="Average Lab distance this level aims to come within"
    )
    notes: list[str]


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


class AttemptOut(BaseModel):
    id: str
    created_at: str
    note: str | None
    message: str
    next_step: str | None
    measurements: list[MeasurementOut]
    image_url: str
    thumbnail_url: str


class PieceSummaryOut(BaseModel):
    id: str
    title: str
    medium: str | None
    created_at: str
    updated_at: str = Field(description="When it was last worked on, not when it was created")
    attempts: int
    has_pattern: bool
    image_url: str
    thumbnail_url: str


class PieceOut(BaseModel):
    id: str
    title: str
    medium: str | None
    created_at: str
    has_pattern: bool
    pattern: dict[str, Any] | None
    image_url: str
    thumbnail_url: str
    attempts: list[AttemptOut]


class TrendOut(BaseModel):
    key: str
    unit: str
    first: float
    latest: float
    change: float
    improved: bool | None = Field(
        default=None,
        description="None where the measure has no agreed better direction, or moved too "
        "little to call",
    )
    description: str


class ProgressOut(BaseModel):
    piece_id: str
    attempts: int
    first_at: str
    latest_at: str
    summary: str
    trends: list[TrendOut]


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


def _resolve_guidance(medium: str, level: str):
    """Translate a medium and skill level into settings, or 400 on a typo."""
    try:
        return guidance_for(medium, level)
    except KeyError as exc:
        raise HTTPException(status_code=400, detail=str(exc).strip("'")) from exc


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


#: The page is edited constantly during development and is small enough that
#: caching it buys nothing. Left cacheable, a browser reuses it without ever
#: revalidating, and every change looks like it silently reverted — which costs
#: far more time than the request it saves.
_NO_CACHE = {"Cache-Control": "no-store, must-revalidate", "Pragma": "no-cache"}


@app.get("/", include_in_schema=False)
def index() -> FileResponse:
    """The web app itself."""
    return FileResponse(_STATIC / "index.html", headers=_NO_CACHE)


@app.get("/calibration-card.svg", include_in_schema=False)
def calibration_card() -> Response:
    """The printable grey reference card, generated on request."""
    svg, _ = render_calibration_card_svg()
    return Response(
        content=svg,
        media_type="image/svg+xml",
        headers={"Content-Disposition": 'inline; filename="artmentor-calibration-card.svg"'},
    )

@app.get("/loomis-sheet.svg", include_in_schema=False)
def loomis_sheet() -> Response:
    """The printable grey reference card, generated on request."""
    svg= render_loomis_sheet_svg()
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
    medium: Annotated[str, Query(description="acrylic, oil, watercolour, …")] = "acrylic",
    level: Annotated[str, Query(description="beginner, improving, confident")] = "improving",
    colors: Annotated[int | None, Query(ge=2, le=24)] = None,
    max_dimension: Annotated[int, Query(ge=200, le=2000)] = 900,
    min_region_area: Annotated[int | None, Query(ge=10, le=5000)] = None,
    include_recipes: Annotated[bool, Query()] = True,
    max_components: Annotated[int | None, Query(ge=1, le=3)] = None,
    paints: Annotated[str | None, Query(description="Comma-separated paint ids")] = None,
    filled: Annotated[bool, Query(description="Color the SVG instead of outlines")] = False,
) -> PatternOut:
    """Turn an uploaded photograph into a numbered, paintable pattern.

    `medium` and `level` are not decoration. They decide how many colours the
    pattern has, how fine its shapes get, how many pigments a recipe may call
    for, and whether white appears in a mix at all — a watercolourist lightens
    with water and the paper, so a recipe asking for titanium white is an
    instruction they cannot carry out. Anything passed explicitly overrides
    them; leaving a parameter out means "use what suits this medium".
    """
    guide = _resolve_guidance(medium, level)
    data = _read_upload(image)
    loaded = _decode(data, max_dimension)

    pattern = generate_pattern(
        loaded.rgb,
        PatternParams(
            n_colors=colors if colors is not None else guide.colors,
            max_dimension=max_dimension,
            min_region_area=(
                min_region_area if min_region_area is not None else guide.min_region_area
            ),
            monochrome=guide.medium.monochrome,
        ),
    )

    recipes = None
    # A graphite pattern has nothing to mix. Running the palette solver anyway
    # would spend five seconds producing pigment names that mean nothing with a
    # pencil in your hand.
    if include_recipes and not guide.medium.monochrome:
        palette = [
            paint
            for paint in _resolve_palette(paints)
            if paint.id not in guide.excluded_paint_ids
        ]
        if not palette:
            raise HTTPException(
                status_code=400,
                detail=f"No paints left once {guide.medium.name.lower()} rules are applied.",
            )
        recipes = recipes_for_pattern(
            pattern,
            palette,
            max_components=(
                max_components if max_components is not None else guide.max_components
            ),
        )

    palette: list[PaletteColorOut] = []
    for index, entry in enumerate(pattern.palette):
        item = PaletteColorOut(number=entry.number, hex=entry.rgb.hex, share=entry.share)
        if guide.medium.monochrome:
            item.instructions = describe_value(entry.lab.L, guide.medium)
        elif recipes is not None:
            recipe = recipes[index]
            item.mix = [
                MixStep(paint=c.paint.name, paint_id=c.paint.id, parts=c.parts)
                for c in recipe.recipe.components
            ]
            item.instructions = describe_mix(
                [(c.paint.name, c.parts) for c in recipe.recipe.components], guide.medium
            )
            item.delta_e = round(recipe.recipe.delta_e, 2)
            item.achievable = recipe.is_achievable
        palette.append(item)

    return PatternOut(
        width=pattern.width,
        height=pattern.height,
        colors=len(pattern.palette),
        regions=len(pattern.regions),
        pipeline_version=pattern.pipeline_version,
        medium=guide.medium.id,
        level=guide.level.id,
        notes=list(guide.notes),
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


@app.get("/mediums", response_model=list[OptionOut])
def list_mediums() -> list[OptionOut]:
    return [OptionOut(id=m.id, name=m.name) for m in MEDIUMS]


@app.get("/levels", response_model=list[OptionOut])
def list_levels() -> list[OptionOut]:
    return [OptionOut(id=level.id, name=level.name) for level in LEVELS]


@app.get("/guidance", response_model=GuidanceOut)
def guidance(
    medium: Annotated[str, Query()] = "acrylic",
    level: Annotated[str, Query()] = "improving",
) -> GuidanceOut:
    """What this combination means, before any image is uploaded.

    Exposed separately so a client can show the learner what changes when they
    switch medium — the colour count, the mixing limit, the missing white —
    rather than discovering it only in the finished pattern.
    """
    g = _resolve_guidance(medium, level)
    return GuidanceOut(
        medium=g.medium.id,
        level=g.level.id,
        colors=g.colors,
        max_components=g.max_components,
        min_region_area=g.min_region_area,
        excluded_paint_ids=list(g.excluded_paint_ids),
        target_delta_e=g.target_delta_e,
        notes=list(g.notes),
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


# ---------------------------------------------------------------------------
# Gallery — the only stateful part of the API
# ---------------------------------------------------------------------------
# Where the gallery lives. One folder holding `gallery.db` and `media/`, chosen
# by environment variable so a deployment can point it at a mounted volume
# without a code change, and defaulting somewhere a laptop user never has to
# think about.
#
# It is resolved through a dependency rather than a module-level global so that
# tests get their own empty gallery per test. A global would make them share one
# database and depend on execution order, which is the kind of test suite that
# passes alone and fails in CI.
_gallery: Gallery | None = None


def gallery_store() -> Gallery:
    global _gallery
    if _gallery is None:
        root = os.environ.get("ARTMENTOR_DATA") or (Path.home() / ".artmentor")
        _gallery = Gallery(root)
    return _gallery


Store = Annotated[Gallery, Depends(gallery_store)]
Digest = Annotated[str, PathParam(pattern=r"^[0-9a-f]{64}$")]


def _gallery_error(exc: GalleryError) -> HTTPException:
    """404 for a thing that is not there, 400 for a request that is wrong."""
    return HTTPException(status_code=404 if isinstance(exc, NotFound) else 400, detail=str(exc))


def _store_upload(upload: UploadFile, *, max_dimension: int = 1600) -> bytes:
    """Validate, sanitise, and re-encode an upload into bytes safe to persist.

    Note the order: nothing a client sent is ever written to disk. What gets
    stored is the array that came out of the hardened loader, re-encoded — so
    EXIF, GPS and any trailing payload are gone before persistence, not after.
    """
    loaded = _decode(_read_upload(upload), max_dimension)
    return encode_image(loaded.rgb, fmt="JPEG")


def _attempt_out(attempt: Attempt) -> AttemptOut:
    return AttemptOut(
        id=attempt.id,
        created_at=attempt.created_at,
        note=attempt.note,
        message=attempt.message,
        next_step=attempt.next_step,
        measurements=[
            MeasurementOut(key=m.key, value=m.value, unit=m.unit, explanation=m.explanation)
            for m in attempt.measurements
        ],
        image_url=f"/gallery/images/{attempt.image_digest}",
        thumbnail_url=f"/gallery/images/{attempt.image_digest}/thumb",
    )


def _piece_out(piece: Piece) -> PieceOut:
    return PieceOut(
        id=piece.id,
        title=piece.title,
        medium=piece.medium,
        created_at=piece.created_at,
        has_pattern=piece.pattern is not None,
        pattern=piece.pattern,
        image_url=f"/gallery/images/{piece.reference_digest}",
        thumbnail_url=f"/gallery/images/{piece.reference_digest}/thumb",
        attempts=[_attempt_out(a) for a in piece.attempts],
    )


def _progress_out(progress: Progress) -> ProgressOut:
    return ProgressOut(
        piece_id=progress.piece_id,
        attempts=progress.attempts,
        first_at=progress.first_at,
        latest_at=progress.latest_at,
        summary=progress.summary,
        trends=[
            TrendOut(
                key=t.key,
                unit=t.unit,
                first=t.first,
                latest=t.latest,
                change=t.change,
                improved=t.improved,
                description=t.description,
            )
            for t in progress.trends
        ],
    )


@app.post("/gallery/pieces", response_model=PieceOut, status_code=201)
def create_piece(
    store: Store,
    image: Annotated[UploadFile, File(description="The reference the learner is painting from")],
    title: Annotated[str, Form(max_length=120)],
    medium: Annotated[str | None, Form(max_length=40)] = None,
    pattern: Annotated[
        str | None, Form(description="The JSON body of a previous /patterns response")
    ] = None,
) -> PieceOut:
    """Start a piece: one reference photograph, worked on over several sittings.

    The pattern is optional and stored verbatim. Keeping it means a learner can
    reopen the piece weeks later and get the same numbered guide rather than a
    regenerated one — quantization is deterministic, but the defaults it ran
    under will not be once the pipeline moves on.
    """
    stored = _store_upload(image)

    parsed: dict[str, Any] | None = None
    if pattern:
        try:
            parsed = json.loads(pattern)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail="pattern must be valid JSON.") from exc
        if not isinstance(parsed, dict):
            raise HTTPException(status_code=400, detail="pattern must be a JSON object.")

    try:
        return _piece_out(store.add_piece(title, stored, medium=medium, pattern=parsed))
    except GalleryError as exc:
        raise _gallery_error(exc) from exc


@app.get("/gallery/pieces", response_model=list[PieceSummaryOut])
def list_gallery(
    store: Store,
    limit: Annotated[int, Query(ge=1, le=200)] = 50,
    offset: Annotated[int, Query(ge=0)] = 0,
) -> list[PieceSummaryOut]:
    """Everything saved, most recently worked on first."""
    return [
        PieceSummaryOut(
            id=p.id,
            title=p.title,
            medium=p.medium,
            created_at=p.created_at,
            updated_at=p.updated_at,
            attempts=p.attempts,
            has_pattern=p.has_pattern,
            image_url=f"/gallery/images/{p.reference_digest}",
            thumbnail_url=f"/gallery/images/{p.reference_digest}/thumb",
        )
        for p in store.list_pieces(limit=limit, offset=offset)
    ]


@app.get("/gallery/pieces/{piece_id}", response_model=PieceOut)
def read_piece(store: Store, piece_id: str) -> PieceOut:
    try:
        return _piece_out(store.get_piece(piece_id))
    except GalleryError as exc:
        raise _gallery_error(exc) from exc


@app.patch("/gallery/pieces/{piece_id}", response_model=PieceOut)
def rename_piece(store: Store, piece_id: str, title: Annotated[str, Form(max_length=120)]) -> PieceOut:
    try:
        return _piece_out(store.rename_piece(piece_id, title))
    except GalleryError as exc:
        raise _gallery_error(exc) from exc


@app.delete("/gallery/pieces/{piece_id}", status_code=204)
def remove_piece(store: Store, piece_id: str) -> Response:
    """Delete a piece, its attempts, and any image nothing else is using."""
    try:
        store.delete_piece(piece_id)
    except GalleryError as exc:
        raise _gallery_error(exc) from exc
    return Response(status_code=204)


@app.post("/gallery/pieces/{piece_id}/attempts", response_model=AttemptOut, status_code=201)
def add_attempt(
    store: Store,
    piece_id: str,
    image: Annotated[UploadFile, File(description="Photo of the painting as it stands")],
    note: Annotated[str | None, Form(max_length=500)] = None,
) -> AttemptOut:
    """Save a photo of the work so far, critiqued against the stored reference.

    The critique is computed here and stored with the attempt rather than being
    recomputed on read. That is deliberate: `critique.py` will change, and a
    learner's history must keep saying what it said at the time. A timeline that
    silently rewrites last month's numbers when the code is updated is not a
    record of anything.
    """
    try:
        piece = store.get_piece(piece_id)
        reference_bytes, _ = store.media_bytes(piece.reference_digest)
    except GalleryError as exc:
        raise _gallery_error(exc) from exc

    attempt_bytes = _store_upload(image)
    reference = _decode(reference_bytes, 900)
    attempt = _decode(attempt_bytes, 900)
    result = critique_attempt(reference.rgb, attempt.rgb)

    try:
        return _attempt_out(store.add_attempt(piece_id, attempt_bytes, result, note=note))
    except GalleryError as exc:
        raise _gallery_error(exc) from exc


@app.delete("/gallery/attempts/{attempt_id}", status_code=204)
def remove_attempt(store: Store, attempt_id: str) -> Response:
    try:
        store.delete_attempt(attempt_id)
    except GalleryError as exc:
        raise _gallery_error(exc) from exc
    return Response(status_code=204)


@app.get("/gallery/pieces/{piece_id}/progress", response_model=ProgressOut)
def piece_progress(store: Store, piece_id: str) -> ProgressOut:
    """Latest attempt against the first one — the "am I improving?" answer.

    Still fidelity, not merit. The trends say the painting moved closer to its
    reference, which is a fact about two photographs, and the wording is kept
    free of praise for the same reason the critique is.
    """
    try:
        return _progress_out(store.progress(piece_id))
    except GalleryError as exc:
        raise _gallery_error(exc) from exc


@app.get("/gallery/images/{digest}", include_in_schema=False)
def gallery_image(store: Store, digest: Digest) -> Response:
    """Serve a stored image by content hash.

    The path pattern only admits 64 hex characters, so no caller-supplied string
    ever reaches the filesystem. Immutable by construction — the bytes are the
    name — so it can be cached hard.
    """
    try:
        data, media_type = store.media_bytes(digest)
    except GalleryError as exc:
        raise _gallery_error(exc) from exc
    return Response(
        content=data,
        media_type=media_type,
        headers={"Cache-Control": "public, max-age=31536000, immutable"},
    )


@app.get("/gallery/images/{digest}/thumb", include_in_schema=False)
def gallery_thumbnail(store: Store, digest: Digest) -> Response:
    try:
        data = store.thumbnail(digest)
    except GalleryError as exc:
        raise _gallery_error(exc) from exc
    return Response(
        content=data,
        media_type="image/jpeg",
        headers={"Cache-Control": "public, max-age=31536000, immutable"},
    )