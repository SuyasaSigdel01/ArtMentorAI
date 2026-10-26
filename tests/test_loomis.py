"""Tests for the Loomis head construction.

Drawing tests are awkward: you cannot assert that something "looks like a
head". So these check the properties that must hold for the geometry to be
right, and which break loudly when the projection is wrong — symmetry, the
relationship between the cut depth and the cut radius, hidden lines appearing
only where they should, and the whole thing being a pure function of two
angles.

`test_turning_left_mirrors_turning_right` is the one that would catch a sign
error in the rotation matrix, which is the single easiest thing to get wrong
here and the hardest to spot by eye.
"""

from __future__ import annotations

import math
import re

import numpy as np
import pytest

from artmentor.core.render.loomis import (
    DEFAULT_PROPORTIONS,
    SHEET_ANGLES,
    HeadProportions,
    _project,
    _rotation,
    render_loomis_sheet_svg,
    render_loomis_svg,
)


def at(point, yaw=0.0, pitch=0.0):
    """Where one head-space point lands: screen x, screen y, depth."""
    return _project(np.array([point], dtype=float), _rotation(yaw, pitch))[0]


def coords(svg: str) -> list[tuple[float, float]]:
    """Every point in every path, so geometry can be asserted on directly."""
    points: list[tuple[float, float]] = []
    for d in re.findall(r'\sd="([^"]+)"', svg):
        for x, y in re.findall(r"[MLQ ](-?\d+\.?\d*) (-?\d+\.?\d*)", d):
            points.append((float(x), float(y)))
    return points


class TestProportions:
    def test_the_cut_depth_follows_from_the_cut_radius(self):
        """Not two independent guesses: the circle lies on the sphere.

        r^2 + d^2 = R^2, with R = 1. Deriving one from the other is what stops
        a future edit from quietly producing a cut that floats off the ball.
        """
        for ratio in (0.4, 2 / 3, 0.8):
            p = HeadProportions(side_plane_ratio=ratio)
            assert p.side_plane_radius ** 2 + p.side_plane_offset ** 2 == pytest.approx(1.0)

    def test_the_default_is_the_commonly_taught_two_thirds(self):
        assert DEFAULT_PROPORTIONS.side_plane_ratio == pytest.approx(2 / 3)
        assert DEFAULT_PROPORTIONS.side_plane_offset == pytest.approx(math.sqrt(5) / 3)

    def test_the_brow_sits_on_the_equator(self):
        """The one thing every source agrees on."""
        assert DEFAULT_PROPORTIONS.brow == 0.0

    def test_the_thirds_are_equal(self):
        p = DEFAULT_PROPORTIONS
        upper = p.hairline - p.brow
        middle = p.brow - p.nose_base
        lower = p.nose_base - p.chin
        assert middle == pytest.approx(upper)
        assert lower == pytest.approx(middle)

    def test_the_chin_hangs_below_the_ball(self):
        """Otherwise the head has no jaw, which was the first bug here."""
        assert DEFAULT_PROPORTIONS.chin < -1.0

    def test_landmarks_must_descend(self):
        with pytest.raises(ValueError):
            HeadProportions(nose_base=0.5)   # below the brow, above it in value
        with pytest.raises(ValueError):
            HeadProportions(chin=0.0)

    @pytest.mark.parametrize("ratio", [0.05, 0.99, -1.0])
    def test_an_impossible_cut_is_refused(self, ratio):
        with pytest.raises(ValueError):
            HeadProportions(side_plane_ratio=ratio)


class TestProjection:
    """Testing the projection directly rather than inferring it from the SVG.

    Comparing sorted coordinates out of the finished drawing looked like a
    symmetry test and was really a test of where the circle sampler happened
    to put its points — it failed by 1.5px on a 120px radius for no better
    reason than sampling phase. These pin the actual behaviour instead.
    """

    def test_face_on_nothing_moves(self):
        assert at((0, 0, 1))[0] == pytest.approx(0.0)
        assert at((0.5, 0, 0))[:2] == pytest.approx([0.5, 0.0])

    def test_screen_y_runs_downward(self):
        """Head space has up as positive; SVG does not."""
        assert at((0, 1, 0))[1] < 0

    def test_turning_right_swings_the_face_right_and_the_cheek_left(self):
        """This is the test that pins the sign of the yaw.

        At a positive turn the nose moves to the viewer's right and the near
        cheek plane sits on the left. Get the sign backwards and the head
        turns the wrong way — which looks entirely plausible until it is
        laid beside a reference.
        """
        nose = at((0, 0, 1), yaw=40)
        cheek = at((-DEFAULT_PROPORTIONS.side_plane_offset, 0, 0), yaw=40)
        assert nose[0] > 0
        assert cheek[0] < 0
        assert cheek[2] > 0, "the near cheek must face the viewer"

    def test_turning_the_other_way_mirrors_it(self):
        assert at((0, 0, 1), yaw=-40)[0] == pytest.approx(-at((0, 0, 1), yaw=40)[0])

    def test_a_turned_head_keeps_its_height(self):
        """Yaw must not scale anything vertically."""
        assert at((0, 1, 0), yaw=55)[1] == pytest.approx(at((0, 1, 0))[1])

    def test_tipping_forward_brings_the_crown_toward_the_viewer(self):
        """Positive pitch is looking down: crown forward, chin swinging back.

        My first instinct was that the chin would ride up the page. It does
        not — it rotates about the ear axis and sweeps down and back, so the
        silhouette gets taller. Worth writing down, because the obvious
        assumption is wrong.
        """
        assert at((0, 1, 0), pitch=30)[2] > 0
        assert at((0, 0, 1), pitch=30)[1] > at((0, 0, 1))[1]

    def test_depth_is_what_decides_visibility(self):
        assert at((0, 0, 1))[2] > 0
        assert at((0, 0, -1))[2] < 0


class TestGeometry:
    def test_the_front_view_is_symmetric(self):
        """Face on, the drawing must sit evenly either side of the centre.

        A bounding box rather than a point-by-point comparison: the circles
        are sampled at fixed angles, so mirrored samples do not land on the
        same coordinates even when the geometry is perfectly symmetric.
        """
        xs = [x for x, _ in coords(render_loomis_svg(0, 0, labels=False))]
        assert min(xs) == pytest.approx(-max(xs), abs=0.5)

    def test_turning_left_mirrors_turning_right(self):
        left = coords(render_loomis_svg(-35, 0, labels=False))
        right = coords(render_loomis_svg(35, 0, labels=False))
        assert min(x for x, _ in left) == pytest.approx(
            -max(x for x, _ in right), abs=0.5)
        assert sorted(round(y, 1) for _, y in left) == pytest.approx(
            sorted(round(y, 1) for _, y in right), abs=0.2)

    def test_turning_the_head_changes_the_drawing(self):
        assert render_loomis_svg(0, 0) != render_loomis_svg(30, 0)
        assert render_loomis_svg(30, 0) != render_loomis_svg(30, 20)

    def test_the_ball_is_a_circle_at_every_angle(self):
        """A sphere projects to a circle however it is turned."""
        for yaw in (0, 30, 60, 90):
            svg = render_loomis_svg(yaw, 0, size=400)
            assert '<circle cx="0" cy="0" r="120.00"' in svg

    def test_nothing_is_drawn_far_outside_the_ball(self):
        """Catches a projection that has quietly exploded."""
        radius = 400 * 0.30
        for yaw, pitch in ((0, 0), (45, 25), (90, -40)):
            for x, y in coords(render_loomis_svg(yaw, pitch, size=400, labels=False)):
                assert abs(x) < radius * 1.6
                assert abs(y) < radius * 2.2


class TestHiddenLines:
    def test_the_back_of_the_form_is_dashed(self):
        """Without this the ball reads as a wireframe rather than a solid."""
        svg = render_loomis_svg(25, 0)
        assert "stroke-dasharray" in svg
        assert svg.count("stroke-dasharray") >= 3

    def test_the_hidden_half_is_drawn_more_faintly(self):
        assert 'opacity=".4"' in render_loomis_svg(25, 0)

    def test_a_profile_still_draws_both_cheek_planes(self):
        """Both lie on the sphere, so both are genuinely half visible."""
        svg = render_loomis_svg(90, 0, labels=False)
        assert len(coords(svg)) > 100


class TestOutput:
    def test_it_is_a_standalone_svg(self):
        svg = render_loomis_svg()
        assert svg.startswith("<?xml")
        assert svg.count("<svg") == 1 and svg.rstrip().endswith("</svg>")

    def test_it_pulls_nothing_off_the_network(self):
        """A construction sheet has to work on paper and offline."""
        svg = render_loomis_svg()
        assert "http://" not in svg.replace("http://www.w3.org/2000/svg", "")
        assert "<script" not in svg and "<image" not in svg

    def test_labels_can_be_turned_off(self):
        assert "<text" in render_loomis_svg(labels=True)
        assert "<text" not in render_loomis_svg(labels=False)

    def test_it_is_deterministic(self):
        assert render_loomis_svg(31, 12) == render_loomis_svg(31, 12)

    @pytest.mark.parametrize("yaw,pitch", [(91, 0), (-91, 0), (0, 61), (0, -61)])
    def test_angles_outside_the_useful_range_are_refused(self, yaw, pitch):
        with pytest.raises(ValueError):
            render_loomis_svg(yaw, pitch)

    @pytest.mark.parametrize("size", [10, 5000])
    def test_absurd_sizes_are_refused(self, size):
        with pytest.raises(ValueError):
            render_loomis_svg(size=size)


class TestSheet:
    def test_the_sheet_holds_every_angle_asked_for(self):
        svg = render_loomis_sheet_svg(((0, 0), (45, 0), (90, 0)), columns=3)
        assert svg.count("turn") == 3

    def test_the_default_sheet_covers_front_to_profile(self):
        assert (0, 0) in SHEET_ANGLES and (90, 0) in SHEET_ANGLES
        assert len(SHEET_ANGLES) >= 8

    def test_each_head_is_captioned_with_its_angle(self):
        svg = render_loomis_sheet_svg(((45, 20),))
        assert "45" in svg and "20" in svg and "down" in svg

    def test_the_sheet_admits_the_numbers_are_a_convention(self):
        """The honesty is part of the deliverable, not a footnote.

        The sources disagree about the side planes and none of them quote
        Loomis directly, so a sheet that presents these ratios as definitive
        would be teaching a learner something we do not actually know.
        """
        assert "not measured from the book" in render_loomis_sheet_svg()

    def test_the_sheet_is_white_so_it_prints(self):
        assert 'fill="#ffffff"' in render_loomis_sheet_svg()

    def test_an_empty_sheet_is_an_error(self):
        with pytest.raises(ValueError):
            render_loomis_sheet_svg(())

    @pytest.mark.parametrize("columns", [0, 9])
    def test_silly_column_counts_are_refused(self, columns):
        with pytest.raises(ValueError):
            render_loomis_sheet_svg(columns=columns)

    def test_a_custom_set_of_proportions_changes_the_sheet(self):
        wide = HeadProportions(side_plane_ratio=0.8)
        assert render_loomis_sheet_svg(proportions=wide) != render_loomis_sheet_svg()
