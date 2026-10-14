"""Tests for the printable calibration card.

Most of these assert on things that are only obvious once you know why the card
exists: the reference must be grey rather than white, it must sit far enough
below the sensor ceiling to survive warm light, and the printed instructions
must survive being printed in greyscale.

`test_reference_survives_a_warm_cast` is the one that encodes the measurement
behind the whole design.
"""

from __future__ import annotations

import re

import pytest

from artmentor.core.render.calibration import (
    CalibrationCard,
    render_calibration_card_svg,
)


@pytest.fixture(scope="module")
def card() -> tuple[str, CalibrationCard]:
    return render_calibration_card_svg()


class TestReferencePatch:
    def test_reference_is_grey_not_white(self, card):
        """White clips under warm light and the correction fails silently."""
        _, geometry = card
        assert 120 <= geometry.reference_level <= 210, (
            "reference must be mid-grey: too light clips, too dark loses signal"
        )

    def test_reference_survives_a_warm_cast(self, card):
        """The measurement this card's design is based on.

        Tungsten light multiplies the red channel by roughly 1.2. The reference
        has to stay under 255 through that, or it clips and the white balance
        anchors to a value that is wrong in the direction it is correcting.
        """
        _, geometry = card
        assert geometry.reference_level * 1.22 < 250, (
            f"level {geometry.reference_level} clips under tungsten"
        )

    def test_reference_patch_is_actually_on_the_card(self, card):
        _, geometry = card
        x, y, w, h = geometry.reference_box_mm
        assert w > 10 and h > 10, "too small to sample reliably"
        assert 0 <= x and x + w <= geometry.width_mm
        assert 0 <= y and y + h <= geometry.height_mm

    def test_reference_is_drawn_in_neutral_grey(self, card):
        """R, G and B must be equal, or it is not a neutral reference."""
        svg, geometry = card
        level = geometry.reference_level
        assert f"#{level:02x}{level:02x}{level:02x}" in svg

    def test_fractions_match_the_millimetre_box(self, card):
        """The fraction form is what the API accepts; it must agree."""
        _, geometry = card
        x0, y0, x1, y1 = geometry.reference_fraction()
        bx, by, _, bh = geometry.reference_box_mm
        assert x0 == pytest.approx(bx / geometry.width_mm)
        assert y1 == pytest.approx((by + bh) / geometry.height_mm)
        assert 0 <= x0 < x1 <= 1
        assert 0 <= y0 < y1 <= 1


class TestStepWedge:
    def test_has_several_patches(self, card):
        """One patch cannot reveal overexposure; a wedge can."""
        svg, _ = card
        greys = set(re.findall(r'fill="#([0-9a-f]{6})"', svg))
        neutral = {g for g in greys if g[0:2] == g[2:4] == g[4:6]}
        assert len(neutral) >= 4

    def test_patches_span_a_useful_range(self, card):
        svg, _ = card
        levels = sorted(
            int(g[0:2], 16)
            for g in set(re.findall(r'fill="#([0-9a-f]{6})"', svg))
            if g[0:2] == g[2:4] == g[4:6] and g not in ("ffffff", "000000")
        )
        assert levels[0] < 100, "needs a dark patch"
        assert levels[-1] > 200, "needs a light patch"


class TestPrintability:
    def test_tells_the_user_to_print_in_greyscale(self, card):
        """A colour printer makes grey from CMY ink, which is not neutral."""
        svg, _ = card
        assert "GREYSCALE" in svg

    def test_the_key_instruction_prints_black(self, card):
        """Red becomes mid-grey in a greyscale print, losing all emphasis."""
        svg, _ = card
        line = next(ln for ln in svg.splitlines() if "GREYSCALE" in ln)
        assert 'fill="#000000"' in line

    def test_warns_about_overexposure(self, card):
        svg, _ = card
        assert "overexposed" in svg

    def test_tells_the_user_to_let_paint_dry(self, card):
        """Wet paint is glossy and reads too light — a real source of bad readings."""
        svg, _ = card
        assert "dry" in svg

    def test_has_physical_dimensions(self, card):
        """Printing at a known size matters; pixels would scale unpredictably."""
        svg, _ = card
        assert 'width="180.0mm"' in svg
        assert 'height="120.0mm"' in svg

    def test_is_valid_svg(self, card):
        svg, _ = card
        assert svg.startswith("<?xml")
        assert svg.rstrip().endswith("</svg>")


class TestLayout:
    def test_swatch_area_does_not_overlap_the_wedge(self, card):
        _, g = card
        _, wedge_y, _, wedge_h = g.reference_box_mm
        _, swatch_y, _, _ = g.swatch_box_mm
        assert swatch_y > wedge_y + wedge_h

    def test_swatch_area_is_big_enough_to_sample(self, card):
        _, g = card
        _, _, w, h = g.swatch_box_mm
        assert w >= 25 and h >= 25

    def test_everything_fits_inside_the_card(self, card):
        _, g = card
        for x, y, w, h in (g.reference_box_mm, g.swatch_box_mm):
            assert x + w <= g.width_mm
            assert y + h <= g.height_mm

    def test_custom_size_is_honoured(self):
        svg, g = render_calibration_card_svg(width_mm=200, height_mm=140)
        assert g.width_mm == 200
        assert 'width="200mm"' in svg

    def test_rejects_a_card_too_small_to_be_useful(self):
        with pytest.raises(ValueError):
            render_calibration_card_svg(width_mm=60, height_mm=40)