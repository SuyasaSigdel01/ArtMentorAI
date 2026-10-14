"""Tests for measuring real paint from photographs.

The strength tests work by round-trip: synthesise a mixture using a strength the
fitter is never told, then check it recovers that number. If the fit is wrong,
the recovered value will not match, and no amount of plausible-looking output
can hide it.
"""

from __future__ import annotations

import numpy as np
import pytest

from artmentor.core.color.difference import delta_e
from artmentor.core.color.mixing import predict_mix_color
from artmentor.core.color.space import lab_to_rgb
from artmentor.core.measure import (
    estimate_tinting_strength,
    measure_paint_color,
    to_catalog_entry,
)
from artmentor.core.paints import catalog_by_id
from artmentor.core.render.calibration import render_calibration_card_svg

PAINTS = catalog_by_id()
WHITE = PAINTS["ti-white"]

HEIGHT, WIDTH = 600, 900
TUNGSTEN = (1.22, 1.0, 0.72)
OPEN_SHADE = (0.88, 0.97, 1.18)

_, CARD = render_calibration_card_svg()


def _slices(x_mm, y_mm, w_mm, h_mm):
    return (
        slice(int(y_mm / CARD.height_mm * HEIGHT), int((y_mm + h_mm) / CARD.height_mm * HEIGHT)),
        slice(int(x_mm / CARD.width_mm * WIDTH), int((x_mm + w_mm) / CARD.width_mm * WIDTH)),
    )


REFERENCE_BOX = _slices(*CARD.reference_box_mm)
SWATCH_BOX = _slices(*CARD.swatch_box_mm)


def card_photo(
    paint_id: str = "cad-red",
    cast: tuple[float, float, float] = (1.0, 1.0, 1.0),
    *,
    reference_level: int | None = None,
    patchy: bool = False,
    seed: int = 0,
) -> np.ndarray:
    """Simulate photographing the calibration card with paint in the square."""
    level = CARD.reference_level if reference_level is None else reference_level
    img = np.full((HEIGHT, WIDTH, 3), 250, dtype=np.uint8)
    img[REFERENCE_BOX] = (level,) * 3

    colour = lab_to_rgb(PAINTS[paint_id].lab)
    img[SWATCH_BOX] = (colour.r, colour.g, colour.b)
    if patchy:
        ys, xs = SWATCH_BOX
        half = slice(ys.start, (ys.start + ys.stop) // 2)
        other = lab_to_rgb(PAINTS["cad-yellow"].lab)
        img[half, xs] = (other.r, other.g, other.b)

    rng = np.random.default_rng(seed)
    lit = img.astype(np.float64) * np.array(cast) + rng.normal(0, 3.0, img.shape)
    return np.clip(lit, 0, 255).astype(np.uint8)


class TestMeasureColour:
    @pytest.mark.parametrize("paint_id", ["cad-red", "ultramarine", "burnt-umber", "cerulean"])
    def test_reads_the_right_colour_in_neutral_light(self, paint_id):
        result = measure_paint_color(
            card_photo(paint_id), reference_region=REFERENCE_BOX, swatch_region=SWATCH_BOX
        )
        assert delta_e(result.lab, PAINTS[paint_id].lab) < 2.0

    @pytest.mark.parametrize("cast", [TUNGSTEN, OPEN_SHADE])
    def test_lighting_is_corrected_away(self, cast):
        """The same tube must measure the same under a bulb and in shade."""
        result = measure_paint_color(
            card_photo("ultramarine", cast),
            reference_region=REFERENCE_BOX,
            swatch_region=SWATCH_BOX,
        )
        assert result.confidence == "calibrated"
        assert delta_e(result.lab, PAINTS["ultramarine"].lab) < 2.5

    def test_saturated_reds_are_the_worst_case(self):
        """Documents a real accuracy ceiling rather than pretending it is flat.

        Saturated reds sit near the edge of what sRGB can represent, so a warm
        cast pushes them around more than it does earths or blues. Measured at
        roughly 3 dE under tungsten, against under 1 for ultramarine.
        """
        result = measure_paint_color(
            card_photo("cad-red", TUNGSTEN),
            reference_region=REFERENCE_BOX,
            swatch_region=SWATCH_BOX,
        )
        assert delta_e(result.lab, PAINTS["cad-red"].lab) < 5.0


class TestTrustworthiness:
    def test_a_good_photo_is_trusted(self):
        result = measure_paint_color(
            card_photo(), reference_region=REFERENCE_BOX, swatch_region=SWATCH_BOX
        )
        assert result.is_trustworthy
        assert result.warnings() == []

    def test_a_clipped_reference_is_not_trusted(self):
        """A white patch instead of grey: clips under warm light, fails silently."""
        result = measure_paint_color(
            card_photo(cast=TUNGSTEN, reference_level=255),
            reference_region=REFERENCE_BOX,
            swatch_region=SWATCH_BOX,
        )
        assert not result.is_trustworthy
        assert any("grey reference" in w for w in result.warnings())

    def test_a_patchy_swatch_is_not_trusted(self):
        """An unevenly mixed square averages to a colour that exists nowhere."""
        result = measure_paint_color(
            card_photo(patchy=True), reference_region=REFERENCE_BOX, swatch_region=SWATCH_BOX
        )
        assert not result.is_trustworthy
        assert any("flat colour" in w for w in result.warnings())

    def test_warnings_are_actionable(self):
        """Every warning must tell the user what to physically do differently."""
        result = measure_paint_color(
            card_photo(cast=TUNGSTEN, reference_level=255, patchy=True),
            reference_region=REFERENCE_BOX,
            swatch_region=SWATCH_BOX,
        )
        assert len(result.warnings()) >= 2
        for warning in result.warnings():
            assert any(verb in warning.lower() for verb in ("re-shoot", "mix", "check"))


class TestTintingStrength:
    @pytest.mark.parametrize(
        "paint_id", ["cad-red", "ultramarine", "phthalo-blue", "burnt-umber", "cerulean"]
    )
    def test_recovers_a_strength_it_was_never_given(self, paint_id):
        """Round-trip: synthesise a mix at a known strength, then fit it back."""
        paint = PAINTS[paint_id]
        observed = predict_mix_color([paint, WHITE], [1, 2]).lab

        strength, residual = estimate_tinting_strength(
            paint.lab, observed, white=WHITE, paint_parts=1, white_parts=2
        )
        assert strength == pytest.approx(paint.tinting_strength, abs=0.15)
        assert residual < 1.0

    def test_separates_a_strong_pigment_from_a_weak_one(self):
        """Phthalo dominates a mix; cerulean does not. The fit must see that."""
        results = {}
        for paint_id in ("phthalo-blue", "cerulean"):
            paint = PAINTS[paint_id]
            observed = predict_mix_color([paint, WHITE], [1, 4]).lab
            results[paint_id], _ = estimate_tinting_strength(
                paint.lab, observed, white=WHITE, paint_parts=1, white_parts=4
            )
        assert results["phthalo-blue"] > results["cerulean"] * 2

    def test_a_bad_observation_reports_a_large_residual(self):
        """No strength can explain an impossible mix, and it must say so.

        A large residual usually means a mis-measured ratio, not an exotic
        pigment — so reporting it is what stops a wrong number entering the
        catalog with the authority of a measurement.
        """
        impossible = PAINTS["cad-yellow"].lab  # a blue + white mix cannot be yellow
        _, residual = estimate_tinting_strength(
            PAINTS["ultramarine"].lab, impossible, white=WHITE
        )
        assert residual > 10.0

    def test_rejects_nonsense_ratios(self):
        for parts in ((0, 1), (1, 0), (-1, 2)):
            with pytest.raises(ValueError):
                estimate_tinting_strength(
                    PAINTS["cad-red"].lab,
                    PAINTS["cad-red"].lab,
                    white=WHITE,
                    paint_parts=parts[0],
                    white_parts=parts[1],
                )


class TestCatalogOutput:
    def test_emits_a_usable_python_line(self):
        measured = measure_paint_color(
            card_photo("ultramarine"), reference_region=REFERENCE_BOX, swatch_region=SWATCH_BOX
        )
        line = to_catalog_entry("ultramarine", "Ultramarine Blue", measured)
        assert 'id="ultramarine"' in line
        assert "LabColor(" in line
        assert line.rstrip().endswith(",")

    def test_writes_lab_not_hex(self):
        """Hex would round-trip through 8-bit sRGB and discard measured precision."""
        measured = measure_paint_color(
            card_photo(), reference_region=REFERENCE_BOX, swatch_region=SWATCH_BOX
        )
        line = to_catalog_entry("x", "X", measured)
        assert "#" not in line.split("tinting_strength")[0]

    def test_flags_an_untrustworthy_measurement_in_the_output(self):
        """A bad reading must not enter the catalog looking like a good one."""
        measured = measure_paint_color(
            card_photo(patchy=True), reference_region=REFERENCE_BOX, swatch_region=SWATCH_BOX
        )
        assert "UNVERIFIED" in to_catalog_entry("x", "X", measured)