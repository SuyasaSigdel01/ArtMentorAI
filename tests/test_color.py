"""Tests for the color core.

These are the foundation every other feature's correctness rests on: if Lab
conversion or CIEDE2000 is subtly wrong, the mixing advice and the critique
metrics are wrong in ways no one will notice until users complain.
"""

from __future__ import annotations

import numpy as np
import pytest

from artmentor.core.color import (
    delta_e,
    delta_e_matrix,
    hex_to_lab,
    lab_to_rgb_array,
    mix_direction_hint,
    rgb_to_lab,
    rgb_to_lab_array,
    verdict_for,
)
from artmentor.core.types import LabColor, RGBColor


class TestConversions:
    def test_white_and_black_anchor_correctly(self):
        white = rgb_to_lab(RGBColor(255, 255, 255))
        black = rgb_to_lab(RGBColor(0, 0, 0))
        assert white.L == pytest.approx(100.0, abs=0.1)
        assert white.a == pytest.approx(0.0, abs=0.1)
        assert white.b == pytest.approx(0.0, abs=0.1)
        assert black.L == pytest.approx(0.0, abs=0.1)

    def test_roundtrip_is_stable(self):
        rng = np.random.default_rng(0)
        rgb = rng.integers(0, 256, size=(200, 3), dtype=np.uint8)
        back = lab_to_rgb_array(rgb_to_lab_array(rgb), as_uint8=True)
        # One unit of 8-bit rounding slack is expected; more means a broken transfer curve.
        assert np.max(np.abs(back.astype(int) - rgb.astype(int))) <= 1

    def test_uint8_and_float_inputs_agree(self):
        as_uint8 = rgb_to_lab_array(np.array([[128, 64, 200]], dtype=np.uint8))
        as_float = rgb_to_lab_array(np.array([[128 / 255, 64 / 255, 200 / 255]]))
        assert np.allclose(as_uint8, as_float, atol=1e-6)

    def test_hex_parsing_handles_shorthand(self):
        assert RGBColor.from_hex("#fff").hex == "#ffffff"
        assert RGBColor.from_hex("abc") == RGBColor(0xAA, 0xBB, 0xCC)

    def test_rejects_malformed_hex(self):
        with pytest.raises(ValueError):
            RGBColor.from_hex("#12345")

    def test_rejects_wrong_shape(self):
        with pytest.raises(ValueError):
            rgb_to_lab_array(np.zeros((4, 4)))


class TestDeltaE:
    def test_identical_colors_are_zero(self):
        c = hex_to_lab("#804080")
        assert delta_e(c, c) == pytest.approx(0.0, abs=1e-9)

    def test_is_symmetric(self):
        a, b = hex_to_lab("#800080"), hex_to_lab("#0000ff")
        assert delta_e(a, b) == pytest.approx(delta_e(b, a), abs=1e-9)

    def test_ordering_matches_perception(self):
        """A near-identical purple must score far closer than a blue."""
        target = hex_to_lab("#800080")
        near = hex_to_lab("#83027e")
        far = hex_to_lab("#0000ff")
        assert delta_e(target, near) < 2.0
        assert delta_e(target, far) > 10.0

    def test_matrix_matches_pairwise(self):
        colors = np.array([[50.0, 10.0, -20.0], [60.0, -5.0, 30.0], [20.0, 0.0, 0.0]])
        matrix = delta_e_matrix(colors, colors)
        assert matrix.shape == (3, 3)
        assert np.allclose(np.diag(matrix), 0.0, atol=1e-9)
        assert matrix[0, 1] == pytest.approx(delta_e(colors[0], colors[1]), abs=1e-6)


class TestVerdicts:
    @pytest.mark.parametrize(
        "value,band,acceptable",
        [
            (0.2, "match", True),
            (1.5, "very_close", True),
            (3.0, "close", False),
            (7.0, "off", False),
            (25.0, "far", False),
        ],
    )
    def test_bands(self, value, band, acceptable):
        verdict = verdict_for(value)
        assert verdict.band == band
        assert verdict.is_acceptable is acceptable
        assert verdict.message

    def test_hint_is_interpolated(self):
        assert "add more red" in verdict_for(3.0, "add more red").message

    def test_no_placeholder_leaks_into_user_text(self):
        """A missing hint must never surface as a literal '{hint}' to a learner."""
        for value in (0.5, 1.5, 3.0, 7.0, 40.0):
            assert "{hint}" not in verdict_for(value).message


class TestMixDirection:
    def test_suggests_white_when_target_is_lighter(self):
        current = LabColor(40.0, 5.0, 5.0)
        target = LabColor(75.0, 5.0, 5.0)
        assert "white" in mix_direction_hint(current, target)

    def test_suggests_red_when_target_is_redder(self):
        current = LabColor(50.0, 0.0, 0.0)
        target = LabColor(50.0, 30.0, 0.0)
        assert "red" in mix_direction_hint(current, target)

    def test_suggests_blue_when_target_is_bluer(self):
        current = LabColor(50.0, 0.0, 20.0)
        target = LabColor(50.0, 0.0, -20.0)
        assert "blue" in mix_direction_hint(current, target)

    def test_reports_only_one_axis(self):
        """Beginners cannot act on three simultaneous corrections."""
        hint = mix_direction_hint(LabColor(40.0, -10.0, -10.0), LabColor(80.0, 25.0, 30.0))
        assert sum(word in hint for word in ("red", "yellow", "white", "blue", "green")) == 1

    def test_negligible_difference_is_reported_as_such(self):
        hint = mix_direction_hint(LabColor(50.0, 0.0, 0.0), LabColor(50.1, 0.05, 0.05))
        assert "small" in hint
