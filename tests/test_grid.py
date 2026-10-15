"""Tests for the grid-method overlay.

Most of these guard things a learner would notice before a developer would: that
the cells are square, that the margin letters match what `label()` says, and
that the millimetre figure is the one to set a ruler by. A grid that is subtly
oblong silently distorts every drawing copied through it, and the person copying
will blame their own eye.
"""

from __future__ import annotations

import re

import numpy as np
import pytest

from artmentor.core.render.grid import (
    GridPlan,
    column_label,
    plan_grid,
    render_grid_svg,
)


def photo(width: int = 600, height: int = 400) -> np.ndarray:
    rng = np.random.default_rng(0)
    return rng.integers(40, 220, (height, width, 3), dtype=np.uint8)


class TestPlanning:
    @pytest.mark.parametrize(
        "width,height", [(900, 600), (600, 900), (800, 800), (1024, 768), (500, 1200)]
    )
    def test_cells_come_out_close_to_square(self, width, height):
        """An oblong grid distorts everything copied through it."""
        plan = plan_grid(width, height, 6)
        assert 0.75 <= plan.cell_aspect <= 1.34, (
            f"{width}x{height} gave cells of aspect {plan.cell_aspect:.2f}"
        )

    def test_divisions_apply_to_the_long_edge(self):
        assert plan_grid(900, 600, 6).columns == 6   # landscape
        assert plan_grid(600, 900, 6).rows == 6      # portrait

    def test_a_square_image_gives_a_square_grid(self):
        plan = plan_grid(800, 800, 5)
        assert plan.columns == plan.rows == 5
        assert plan.is_square_enough

    def test_squareness_flag_is_honest(self):
        """An awkward ratio must admit its cells are not square."""
        assert plan_grid(1200, 500, 6).is_square_enough is False
        assert plan_grid(900, 600, 6).is_square_enough is True

    def test_cells_tile_the_whole_image(self):
        plan = plan_grid(1024, 768, 8)
        assert plan.cell_width * plan.columns == pytest.approx(1024)
        assert plan.cell_height * plan.rows == pytest.approx(768)

    @pytest.mark.parametrize("bad", [1, 0, -3, 41, 100])
    def test_rejects_silly_division_counts(self, bad):
        with pytest.raises(ValueError):
            plan_grid(600, 400, bad)

    def test_rejects_empty_images(self):
        with pytest.raises(ValueError):
            plan_grid(0, 400, 6)


class TestPaperMath:
    def test_gives_a_millimetre_figure_for_the_paper(self):
        """The number an artist actually needs — pixels are useless at a desk."""
        plan = plan_grid(900, 600, 6)          # 6 x 4 cells
        assert plan.cell_size_mm(297) == pytest.approx(297 / 6)  # A4 long edge

    def test_scales_by_the_long_edge(self):
        portrait = plan_grid(600, 900, 6)      # 4 x 6 cells
        assert portrait.cell_size_mm(297) == pytest.approx(297 / 6)

    def test_rejects_nonsense_paper(self):
        with pytest.raises(ValueError):
            plan_grid(600, 400, 6).cell_size_mm(0)


class TestLabels:
    def test_reads_like_a_chessboard(self):
        plan = plan_grid(900, 600, 6)
        assert plan.label(0, 0) == "A1"
        assert plan.label(2, 3) == "C4"

    def test_continues_past_z(self):
        assert [column_label(i) for i in (0, 25, 26, 27)] == ["A", "Z", "AA", "AB"]

    def test_rendered_margin_matches_the_plan(self):
        """The two used to disagree past 26 columns — a second 'A' over cell AA."""
        svg, plan = render_grid_svg(photo(900, 900), divisions=30, labels=True)
        assert plan.columns > 26
        assert ">AA<" in svg
        assert plan.label(26, 0) == "AA1"

    def test_rejects_cells_outside_the_grid(self):
        plan = plan_grid(900, 600, 6)
        for bad in ((plan.columns, 0), (0, plan.rows), (-1, 0)):
            with pytest.raises(ValueError):
                plan.label(*bad)


class TestRendering:
    def test_draws_the_right_number_of_lines(self):
        svg, plan = render_grid_svg(photo(), divisions=6, labels=False)
        lines = len(re.findall(r"<line ", svg))
        assert lines == (plan.columns - 1) + (plan.rows - 1)

    def test_embeds_the_photo_so_the_file_stands_alone(self):
        """A printable reference that needs a second file arrives broken."""
        svg, _ = render_grid_svg(photo())
        assert "data:image/jpeg;base64," in svg
        # Namespace declarations are http:// URLs but are identifiers, never
        # fetched. What matters is that nothing the renderer *loads* is remote.
        assert not re.search(r'(href|src)="https?://', svg)

    def test_labels_can_be_turned_off(self):
        with_labels, _ = render_grid_svg(photo(), labels=True)
        without, _ = render_grid_svg(photo(), labels=False)
        assert "<text" in with_labels
        assert "<text" not in without

    def test_diagonals_are_opt_in(self):
        """An unwanted line is harder to ignore than a missing one is to add."""
        plain, _ = render_grid_svg(photo(), diagonals=False)
        crossed, _ = render_grid_svg(photo(), diagonals=True)
        assert "stroke-dasharray" not in plain
        assert "stroke-dasharray" in crossed

    def test_honours_a_supplied_plan(self):
        plan = GridPlan(columns=3, rows=2, image_width=600, image_height=400)
        _, used = render_grid_svg(photo(), plan)
        assert used is plan

    def test_is_valid_svg(self):
        svg, _ = render_grid_svg(photo())
        assert svg.startswith("<?xml")
        assert svg.rstrip().endswith("</svg>")

    def test_rejects_a_non_rgb_image(self):
        with pytest.raises(ValueError):
            render_grid_svg(np.zeros((40, 40), dtype=np.uint8))

    def test_stays_a_sane_size(self):
        """Embedding the photo must not produce a file nobody can email."""
        svg, _ = render_grid_svg(photo(1200, 900))
        assert len(svg) < 2_000_000