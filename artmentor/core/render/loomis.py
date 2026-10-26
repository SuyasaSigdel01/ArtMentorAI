"""The Loomis head construction, drawn at any angle.

The second half of feature 5. The grid method is pure geometry with a right
answer; this is a *taught convention*, and that difference drove every
decision here.

Where the numbers come from
---------------------------
Two things every source agrees on: the brow line runs through the ball's
horizontal centre, and the face divides into thirds — hairline to brow, brow
to the base of the nose, nose to chin.

The side planes are another matter. No source consulted states how deep the
slice is cut. One gives "about two thirds the height of the sphere" for the
oval; another says the cut meets the sphere at 30-45 degrees and then
contradicts its own thirds a paragraph later. None of them quote *Drawing the
Head and Hands* for a single figure, though all of them cite it.

So every ratio lives in `HeadProportions` as a named constant, the defaults
are the commonly taught figures, and the UI says so. If someone opens Loomis's
book and finds different numbers, correcting this is editing one dataclass —
not unpicking arithmetic spread through a renderer. What is *not* adjustable
is the stuff that is geometry rather than convention: a sphere projects to a
circle, a circle on that sphere projects to an ellipse, and a point is visible
when it faces the viewer.

What this deliberately does not do
----------------------------------
It does not look at the learner's photograph. Fitting the construction to a
real face needs pose estimation — a pretrained model, a licence to check, a
failure mode on stylised references, and a face-data question to answer before
launch. The geometry here is a pure function of two angles, so that can be
added later behind `interfaces`-style indirection without touching any of the
drawing code.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

__all__ = [
    "DEFAULT_PROPORTIONS",
    "SHEET_ANGLES",
    "HeadProportions",
    "render_loomis_sheet_svg",
    "render_loomis_svg",
]


@dataclass(frozen=True)
class HeadProportions:
    """Every ratio the construction uses, in units of the ball's radius.

    Defaults are the commonly taught figures, not measurements from Loomis.
    See the module docstring before trusting any of them to three decimals.
    """

    #: Side-plane circle diameter as a fraction of the ball's diameter. The
    #: one figure the sources do offer, and the cut depth follows from it by
    #: Pythagoras rather than by being guessed separately.
    side_plane_ratio: float = 2 / 3

    #: Heights measured from the brow, which sits at the ball's equator.
    #:
    #: Three claims are each near-universal and cannot all be true at once:
    #: the brow is at the equator, the face divides into equal thirds, and the
    #: chin hangs below the ball. Equal thirds of half a radius put the chin
    #: exactly at the ball's lowest point, which leaves a head with no jaw —
    #: and every Loomis drawing plainly has one. So the thirds here are 0.72
    #: of a radius, which keeps the first two and gives the third somewhere to
    #: go. The casualty is a fourth claim, that the hairline sits halfway
    #: between brow and crown; here it lands a little lower.
    hairline: float = 0.72
    brow: float = 0.0
    nose_base: float = -0.72
    chin: float = -1.44

    #: The chin sits forward of the ball's centre, which is why a Loomis head
    #: in three-quarter view shows a jaw outside the sphere's silhouette even
    #: though the chin is level with its lowest point.
    chin_forward: float = 0.62

    #: The chin is a short flat, not a point. A single apex reads as a
    #: caricature and teaches the wrong shape.
    chin_width: float = 0.34

    def __post_init__(self) -> None:
        if not 0.2 <= self.side_plane_ratio <= 0.95:
            raise ValueError("side_plane_ratio must be between 0.2 and 0.95")
        if not self.hairline > self.brow > self.nose_base > self.chin:
            raise ValueError("the facial landmarks must descend in order")

    @property
    def side_plane_radius(self) -> float:
        """Radius of the circle exposed by the cut, in ball radii."""
        return self.side_plane_ratio

    @property
    def side_plane_offset(self) -> float:
        """How far from the centre the cut sits.

        Not a separate guess: the cut circle lies on the sphere, so once its
        radius is chosen the depth is fixed at sqrt(R^2 - r^2).
        """
        return math.sqrt(max(0.0, 1.0 - self.side_plane_radius ** 2))


DEFAULT_PROPORTIONS = HeadProportions()

#: A practice sheet of useful angles: front, two three-quarter turns, profile,
#: and the same turned downward. Looking up is left out — it is the view a
#: beginner almost never needs and it crowds the page.
SHEET_ANGLES: tuple[tuple[float, float], ...] = (
    (0, 0), (25, 0), (45, 0), (70, 0), (90, 0),
    (0, 20), (25, 20), (45, 20), (70, 20), (90, 20),
)


# --------------------------------------------------------------------------
# Projection
# --------------------------------------------------------------------------
# Head space: x to the subject's right, y up, z toward the viewer. Orthographic
# projection, because a construction drawing is a diagram rather than a
# photograph — perspective would make the two side planes different sizes and
# the learner would copy the distortion.
def _rotation(yaw_deg: float, pitch_deg: float) -> np.ndarray:
    yaw, pitch = math.radians(yaw_deg), math.radians(pitch_deg)
    cy, sy = math.cos(yaw), math.sin(yaw)
    cp, sp = math.cos(pitch), math.sin(pitch)
    turn = np.array([[cy, 0.0, sy], [0.0, 1.0, 0.0], [-sy, 0.0, cy]])
    nod = np.array([[1.0, 0.0, 0.0], [0.0, cp, -sp], [0.0, sp, cp]])
    return nod @ turn


def _project(points: np.ndarray, rotation: np.ndarray) -> np.ndarray:
    """(N,3) head-space points -> (N,3) of screen x, screen y, depth.

    Screen y is negated because SVG counts downward and heads do not.
    """
    turned = points @ rotation.T
    return np.column_stack([turned[:, 0], -turned[:, 1], turned[:, 2]])


def _circle(centre: tuple[float, float, float], u: np.ndarray, v: np.ndarray,
            radius: float, samples: int = 160) -> np.ndarray:
    t = np.linspace(0.0, 2.0 * math.pi, samples, endpoint=True)
    return np.asarray(centre) + radius * (
        np.cos(t)[:, None] * u + np.sin(t)[:, None] * v
    )


_X = np.array([1.0, 0.0, 0.0])
_Y = np.array([0.0, 1.0, 0.0])
_Z = np.array([0.0, 0.0, 1.0])


# --------------------------------------------------------------------------
# Hidden lines
# --------------------------------------------------------------------------
def _runs(projected: np.ndarray, visible: bool) -> list[np.ndarray]:
    """Split a projected curve into the stretches facing the viewer, or away.

    Everything this module draws lies on the sphere's surface — the ball's
    own outline, the cut circles, the brow and the centre line — so one test
    covers all of them: a point is visible when its depth is positive. Drawing
    the hidden half as a faint dash is what makes the thing read as a solid
    object instead of a wireframe.
    """
    wanted = (projected[:, 2] >= 0) if visible else (projected[:, 2] < 0)
    out, run = [], []
    for point, ok in zip(projected, wanted, strict=True):
        if ok:
            run.append(point[:2])
        elif run:
            out.append(np.asarray(run))
            run = []
    if run:
        out.append(np.asarray(run))
    return [r for r in out if len(r) > 1]


def _path(points: np.ndarray) -> str:
    head = f"M{points[0][0]:.2f} {points[0][1]:.2f}"
    rest = "".join(f"L{x:.2f} {y:.2f}" for x, y in points[1:])
    return head + rest


# --------------------------------------------------------------------------
# One head
# --------------------------------------------------------------------------
_INK = "#16120e"
_GUIDE = "#c2552b"
_FAINT = "#b9ab93"


def _head_markup(
    yaw_deg: float,
    pitch_deg: float,
    radius: float,
    proportions: HeadProportions,
    *,
    labels: bool,
) -> str:
    rot = _rotation(yaw_deg, pitch_deg)
    p = proportions
    r_side = p.side_plane_radius
    d = p.side_plane_offset

    def draw(curve: np.ndarray, colour: str, width: float) -> str:
        shown = _project(curve * radius, rot)
        bits = [
            f'<path d="{_path(run)}" fill="none" stroke="{colour}" '
            f'stroke-width="{width}" stroke-linecap="round"/>'
            for run in _runs(shown, visible=True)
        ]
        bits += [
            f'<path d="{_path(run)}" fill="none" stroke="{colour}" '
            f'stroke-width="{width * 0.8}" stroke-dasharray="3 4" opacity=".4"/>'
            for run in _runs(shown, visible=False)
        ]
        return "".join(bits)

    # The ball. A sphere projects to a circle at every angle, so this one
    # needs no projection at all.
    ball = (
        f'<circle cx="0" cy="0" r="{radius:.2f}" fill="none" '
        f'stroke="{_INK}" stroke-width="2.2"/>'
    )
    parts: list[str] = [ball]

    # The two cut faces. The far one is held back: both lie on the sphere so
    # both are genuinely half visible, but drawn at equal weight they read as
    # a diagram of a sphere rather than as a head with a near cheek.
    near = -1.0 if yaw_deg >= 0 else 1.0
    for sign in (-1.0, 1.0):
        on_top = sign == near
        parts.append(draw(_circle((sign * d, 0, 0), _Y, _Z, r_side),
                          _INK if on_top else _FAINT, 1.9 if on_top else 1.2))

    # Brow line and centre line: two great circles at right angles.
    parts.append(draw(_circle((0, 0, 0), _X, _Z, 1.0), _GUIDE, 1.8))
    parts.append(draw(_circle((0, 0, 0), _Y, _Z, 1.0), _GUIDE, 1.8))

    # The jaw. Three parts each side, because a single curve to a single
    # apex gives a witch's chin: the line leaves the bottom of the cheek
    # plane, turns at the jaw corner, and meets a chin that is a short flat
    # rather than a point. The chin sits forward of the ball, which is what
    # puts the jaw outside the sphere's silhouette in a three-quarter view.
    chin_half = p.chin_width / 2
    for sign in (-1.0, 1.0):
        start_pt = np.array([sign * d, -r_side * 0.92, 0.12])
        chin_pt = np.array([sign * chin_half, p.chin, p.chin_forward])
        control = np.array([
            sign * d * 0.88,
            p.chin * 0.74,
            p.chin_forward * 0.42,
        ])
        shown = _project(np.array([start_pt, control, chin_pt]) * radius, rot)
        parts.append(
            f'<path d="M{shown[0][0]:.2f} {shown[0][1]:.2f}'
            f'Q{shown[1][0]:.2f} {shown[1][1]:.2f} {shown[2][0]:.2f} {shown[2][1]:.2f}"'
            f' fill="none" stroke="{_INK}" stroke-width="2.2" stroke-linecap="round"/>'
        )
    flat = _project(np.array([
        [-chin_half, p.chin, p.chin_forward],
        [chin_half, p.chin, p.chin_forward],
    ]) * radius, rot)
    parts.append(
        f'<path d="M{flat[0][0]:.2f} {flat[0][1]:.2f}L{flat[1][0]:.2f} {flat[1][1]:.2f}"'
        f' fill="none" stroke="{_INK}" stroke-width="2.2" stroke-linecap="round"/>'
    )

    # Centre line carried down the face to the chin, so the learner can see
    # which way the head is turned below the ball as well as on it.
    face = np.array([[0.0, 0.0, 1.0], [0.0, p.nose_base, 0.95],
                     [0.0, p.chin, p.chin_forward]])
    shown = _project(face * radius, rot)
    parts.append(
        f'<path d="M{shown[0][0]:.2f} {shown[0][1]:.2f}'
        f'Q{shown[1][0]:.2f} {shown[1][1]:.2f} {shown[2][0]:.2f} {shown[2][1]:.2f}"'
        f' fill="none" stroke="{_GUIDE}" stroke-width="1.6" stroke-dasharray="5 4"/>'
    )

    # The thirds, drawn as arcs across the front of the form rather than as
    # straight rules: a flat line across a turned head is the single commonest
    # way a beginner flattens a face.
    for height, name in (
        (p.hairline, "hairline"), (p.brow, "brow"),
        (p.nose_base, "nose"), (p.chin, "chin"),
    ):
        if abs(height) >= 1.0:
            continue  # the chin has no arc on the ball; the jaw carries it
        ring = math.sqrt(max(0.0, 1.0 - height ** 2))
        arc = _circle((0, height, 0), _X, _Z, ring)
        front = arc[arc[:, 2] > 0]
        if len(front) > 1:
            shown = _project(front * radius, rot)
            parts.append(
                f'<path d="{_path(shown[:, :2])}" fill="none" stroke="{_FAINT}" '
                f'stroke-width="1.3" stroke-dasharray="2 3"/>'
            )
        if labels:
            anchor = _project(np.array([[0.0, height, 1.0]]) * radius, rot)[0]
            parts.append(
                f'<text x="{radius * 1.18:.1f}" y="{anchor[1]:.1f}" '
                f'font-family="ui-sans-serif,system-ui,sans-serif" font-size="9" '
                f'fill="{_FAINT}" dominant-baseline="middle">{name}</text>'
            )

    return "".join(parts)


def render_loomis_svg(
    yaw_deg: float = 25.0,
    pitch_deg: float = 0.0,
    *,
    size: int = 360,
    proportions: HeadProportions = DEFAULT_PROPORTIONS,
    labels: bool = True,
) -> str:
    """One Loomis head at the given turn and tilt, as a standalone SVG."""
    if not -90.0 <= yaw_deg <= 90.0:
        raise ValueError("yaw must be between -90 and 90 degrees")
    if not -60.0 <= pitch_deg <= 60.0:
        raise ValueError("pitch must be between -60 and 60 degrees")
    if not 80 <= size <= 2000:
        raise ValueError("size must be between 80 and 2000 pixels")

    radius = size * 0.30
    body = _head_markup(yaw_deg, pitch_deg, radius, proportions, labels=labels)
    half = size / 2
    return (
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{size}" height="{size}" '
        f'viewBox="{-half:.0f} {-half * 1.05:.0f} {size} {size}">'
        f'<g>{body}</g></svg>'
    )


def render_loomis_sheet_svg(
    angles: tuple[tuple[float, float], ...] = SHEET_ANGLES,
    *,
    columns: int = 5,
    cell: int = 215,
    proportions: HeadProportions = DEFAULT_PROPORTIONS,
) -> str:
    """A printable sheet of heads at several angles.

    More useful than a single head, and the reason the static version was
    worth building first: a page of turns pinned above a desk is how this
    method is actually practised.
    """
    if not angles:
        raise ValueError("a sheet needs at least one angle")
    if not 1 <= columns <= 8:
        raise ValueError("columns must be between 1 and 8")

    rows = math.ceil(len(angles) / columns)
    width, height = columns * cell, rows * cell + 46
    # The chin reaches 1.44 radii below centre, so the head has to be small
    # enough in its cell that the caption below it is not sitting on the jaw.
    radius = cell * 0.245

    cells = []
    for i, (yaw, pitch) in enumerate(angles):
        cx = (i % columns) * cell + cell / 2
        cy = (i // columns) * cell + cell / 2 + 40
        body = _head_markup(yaw, pitch, radius, proportions, labels=False)
        cells.append(
            f'<g transform="translate({cx:.1f} {cy:.1f})">{body}'
            f'<text x="0" y="{cell * 0.46:.0f}" text-anchor="middle" '
            f'font-family="ui-sans-serif,system-ui,sans-serif" font-size="10" '
            f'fill="{_FAINT}">{yaw:g}&#176; turn'
            f'{f", {pitch:g}&#176; down" if pitch else ""}</text></g>'
        )

    return (
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}">'
        f'<rect width="{width}" height="{height}" fill="#ffffff"/>'
        f'<text x="20" y="26" font-family="ui-sans-serif,system-ui,sans-serif" '
        f'font-size="13" font-weight="600" fill="{_INK}">The Loomis head '
        f'&#8212; commonly taught proportions, not measured from the book</text>'
        + "".join(cells) + "</svg>"
    )
