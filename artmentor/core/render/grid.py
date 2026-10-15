"""The grid method: a printable reference with a grid drawn over it.

The oldest drawing aid there is. Divide the reference into cells, draw the same
cells on your paper, and copy one cell at a time. It works because judging where
a line sits inside a small square is a much easier perceptual task than judging
where it sits on a whole page, and it is how most people are first taught to get
proportions right.

Three details decide whether a grid is actually usable, and all three are easy
to get wrong:

**Cells must be square, or near enough.** A grid of squares on the reference and
oblongs on the paper distorts everything copied through it. The image's aspect
ratio rarely divides evenly, so this picks the row count whose cells come
closest to square and reports how far off they are.

**The artist needs the cell size for their paper, not for the image.** Pixels
are useless at a drawing board. `cell_size_mm` converts: given the long edge of
your paper, it says how many millimetres to step between pencil marks.

**Cells need names.** "The eye sits in C4" is the whole point — it is how a
learner checks themselves, and how a teacher gives feedback. Columns are
lettered, rows numbered, and the labels print in the margin.
"""

from __future__ import annotations

import base64
import io
from dataclasses import dataclass

import numpy as np

__all__ = ["GridPlan", "column_label", "plan_grid", "render_grid_svg"]

_FONT = "Helvetica, Arial, sans-serif"
_LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"


@dataclass(frozen=True)
class GridPlan:
    """How an image divides into cells."""

    columns: int
    rows: int
    image_width: int
    image_height: int

    @property
    def cell_width(self) -> float:
        return self.image_width / self.columns

    @property
    def cell_height(self) -> float:
        return self.image_height / self.rows

    @property
    def cell_aspect(self) -> float:
        """Width over height of one cell. 1.0 is square."""
        return self.cell_width / self.cell_height

    @property
    def is_square_enough(self) -> bool:
        """Within 5% of square — below the threshold where copying distorts.

        Past this the drawing comes out subtly stretched, and a beginner will
        blame their own eye rather than the grid.
        """
        return 0.95 <= self.cell_aspect <= 1.05

    def cell_size_mm(self, paper_long_edge_mm: float) -> float:
        """Millimetres per cell when the image is scaled to fit that paper.

        The number to set a ruler by. Scaling is by the long edge, matching how
        a reference is normally enlarged onto a sheet.
        """
        if paper_long_edge_mm <= 0:
            raise ValueError("paper edge must be positive")
        longest_cells = max(self.columns, self.rows)
        return paper_long_edge_mm / longest_cells

    def label(self, column: int, row: int) -> str:
        """Human name for a cell, e.g. 'C4'. Zero-indexed input."""
        if not (0 <= column < self.columns and 0 <= row < self.rows):
            raise ValueError(f"cell ({column}, {row}) is outside the grid")
        return f"{column_label(column)}{row + 1}"


def column_label(column: int) -> str:
    """Letter for a column: A..Z, then AA, AB and so on.

    Shared by `GridPlan.label` and the rendered margin so the two can never
    disagree — at more than 26 columns a naive modulo prints a second "A" over
    a cell the plan calls "AA".
    """
    if column < len(_LETTERS):
        return _LETTERS[column]
    return _LETTERS[column // len(_LETTERS) - 1] + _LETTERS[column % len(_LETTERS)]


def plan_grid(width: int, height: int, divisions: int = 6) -> GridPlan:
    """Divide an image into roughly square cells.

    `divisions` sets the cell count along the *longer* edge; the shorter edge
    gets however many keep the cells closest to square. Specifying only one
    number is deliberate — asking a beginner for rows and columns separately
    invites the oblong-cell mistake this function exists to avoid.
    """
    if width <= 0 or height <= 0:
        raise ValueError("image dimensions must be positive")
    if not 2 <= divisions <= 40:
        raise ValueError("divisions must be between 2 and 40")

    if width >= height:
        columns = divisions
        rows = max(1, round(height / (width / columns)))
    else:
        rows = divisions
        columns = max(1, round(width / (height / rows)))

    return GridPlan(columns=columns, rows=rows, image_width=width, image_height=height)


def _data_uri(rgb: np.ndarray) -> str:
    """Encode an image for embedding, so the SVG is one self-contained file.

    A printable reference that depends on a separate image file is a reference
    that arrives broken. JPEG at quality 88 keeps a full-page print clean while
    staying small enough to hand around.
    """
    from PIL import Image

    buffer = io.BytesIO()
    Image.fromarray(np.asarray(rgb, dtype=np.uint8), "RGB").save(
        buffer, format="JPEG", quality=88, optimize=True
    )
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/jpeg;base64,{encoded}"


def render_grid_svg(
    rgb: np.ndarray,
    plan: GridPlan | None = None,
    *,
    divisions: int = 6,
    labels: bool = True,
    diagonals: bool = False,
    line_color: str = "#e02020",
    line_width: float = 1.0,
) -> tuple[str, GridPlan]:
    """Render the reference image with a grid drawn over it.

    Args:
        rgb: the reference photograph.
        plan: a grid to use; one is planned from `divisions` when omitted.
        labels: print column letters and row numbers in the margin.
        diagonals: add corner-to-corner lines. Some teachers swear by them for
            placing a subject's angle early; others find them clutter. Off by
            default, because an unwanted line is harder to ignore than a
            missing one is to add.
        line_color: red by default — it stays visible over almost any
            photograph, and reads as an overlay rather than part of the picture.

    Returns the SVG and the plan actually used.
    """
    if rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError(f"expected an (H, W, 3) image, got {rgb.shape}")

    height, width = rgb.shape[:2]
    plan = plan or plan_grid(width, height, divisions)

    margin = 22.0 if labels else 0.0
    total_w, total_h = width + margin, height + margin

    parts: list[str] = [
        f'<rect width="{total_w}" height="{total_h}" fill="#ffffff"/>',
        (
            f'<image x="{margin}" y="{margin}" width="{width}" height="{height}" '
            f'href="{_data_uri(rgb)}"/>'
        ),
    ]

    cw, ch = plan.cell_width, plan.cell_height

    # Interior lines only; the border is drawn once afterwards so corners meet.
    for c in range(1, plan.columns):
        x = margin + c * cw
        parts.append(
            f'<line x1="{x:.1f}" y1="{margin}" x2="{x:.1f}" y2="{total_h}" '
            f'stroke="{line_color}" stroke-width="{line_width}" opacity="0.85"/>'
        )
    for r in range(1, plan.rows):
        y = margin + r * ch
        parts.append(
            f'<line x1="{margin}" y1="{y:.1f}" x2="{total_w}" y2="{y:.1f}" '
            f'stroke="{line_color}" stroke-width="{line_width}" opacity="0.85"/>'
        )

    if diagonals:
        parts.append(
            f'<path d="M{margin},{margin} L{total_w},{total_h} '
            f'M{total_w},{margin} L{margin},{total_h}" fill="none" '
            f'stroke="{line_color}" stroke-width="{line_width * 0.8}" '
            f'opacity="0.5" stroke-dasharray="7 5"/>'
        )

    parts.append(
        f'<rect x="{margin}" y="{margin}" width="{width}" height="{height}" '
        f'fill="none" stroke="{line_color}" stroke-width="{line_width * 1.6}"/>'
    )

    if labels:
        size = min(13.0, max(7.0, min(cw, ch) * 0.3))
        for c in range(plan.columns):
            parts.append(
                f'<text x="{margin + (c + 0.5) * cw:.1f}" y="{margin - 7:.1f}" '
                f'font-family="{_FONT}" font-size="{size:.1f}" fill="#333333" '
                f'text-anchor="middle">{column_label(c)}</text>'
            )
        for r in range(plan.rows):
            parts.append(
                f'<text x="{margin - 8:.1f}" y="{margin + (r + 0.5) * ch:.1f}" '
                f'font-family="{_FONT}" font-size="{size:.1f}" fill="#333333" '
                f'text-anchor="end" dominant-baseline="central">{r + 1}</text>'
            )

    svg = "\n".join(
        [
            '<?xml version="1.0" encoding="UTF-8"?>',
            (
                f'<svg xmlns="http://www.w3.org/2000/svg" '
                f'xmlns:xlink="http://www.w3.org/1999/xlink" '
                f'width="{total_w:.0f}" height="{total_h:.0f}" '
                f'viewBox="0 0 {total_w:.0f} {total_h:.0f}">'
            ),
            *parts,
            "</svg>",
        ]
    )
    return svg, plan