"""Tests for forward pigment mixing.

The assertions here are deliberately about *direction and hue*, not exact hex
values. The underlying model infers spectra from sRGB rather than measuring
them, so precise outputs will shift when measured paint data replaces the
starter catalog. What must not shift is the behaviour that makes the feature
worth having: paint mixing subtractively, white lightening, and complements
greying each other.

`test_blue_and_yellow_make_green` is the one that justifies the whole spectral
detour. If it ever fails, the model has silently fallen back to something
equivalent to averaging RGB, and every mixing suggestion in the product is wrong.
"""

from __future__ import annotations

import colorsys

import numpy as np
import pytest

from artmentor.core.color.mixing import mix_paints, mix_reflectances, predict_mix_color
from artmentor.core.color.spectral import (
    ks_to_reflectance,
    reflectance_to_ks,
    reflectance_to_srgb,
    srgb_to_reflectance,
)
from artmentor.core.interfaces import MixComponent
from artmentor.core.paints import STARTER_ACRYLICS, catalog_by_id, make_paint
from artmentor.core.types import RGBColor

PAINTS = catalog_by_id()


def hue_degrees(rgb: RGBColor) -> float:
    """Hue in degrees: 0 red, 60 yellow, 120 green, 240 blue."""
    h, _, _ = colorsys.rgb_to_hsv(rgb.r / 255.0, rgb.g / 255.0, rgb.b / 255.0)
    return h * 360.0


class TestSpectralRoundTrip:
    @pytest.mark.parametrize(
        "hex_color", ["#ffffff", "#000000", "#cc1a1a", "#1a26b2", "#f2d91a", "#4a7023"]
    )
    def test_reflectance_reproduces_its_color(self, hex_color):
        """A recovered spectrum must render back to the color it came from."""
        original = RGBColor.from_hex(hex_color)
        target = np.array([original.r, original.g, original.b]) / 255.0
        recovered = reflectance_to_srgb(srgb_to_reflectance(target))
        assert np.max(np.abs(recovered - target)) < 0.01

    def test_reflectance_stays_physical(self):
        reflectance = srgb_to_reflectance((0.8, 0.2, 0.1))
        assert np.all(reflectance > 0.0)
        assert np.all(reflectance <= 1.0)

    def test_ks_inverts_cleanly(self):
        reflectance = srgb_to_reflectance((0.4, 0.6, 0.3))
        assert np.allclose(ks_to_reflectance(reflectance_to_ks(reflectance)), reflectance, atol=1e-6)

    def test_accepts_0_255_input(self):
        a = srgb_to_reflectance((204, 26, 26))
        b = srgb_to_reflectance((0.8, 26 / 255, 26 / 255))
        assert np.allclose(a, b, atol=1e-3)


class TestSubtractiveBehaviour:
    def test_blue_and_yellow_make_green(self):
        """The defining property of subtractive mixing.

        Averaging RGB gives grey here. Anything that passes this test is doing
        real pigment mixing.
        """
        result = predict_mix_color(
            [PAINTS["phthalo-blue"], PAINTS["cad-yellow"]], [1, 2]
        )
        assert result.rgb.g > result.rgb.r, f"not green-dominant: {result.hex}"
        hue = hue_degrees(result.rgb)
        assert 90.0 <= hue <= 210.0, f"hue {hue:.0f} deg is outside green/teal: {result.hex}"

    def test_beats_naive_rgb_averaging(self):
        """Explicitly contrast with the approach this module exists to replace."""
        from artmentor.core.color.space import lab_to_rgb

        blue, yellow = PAINTS["phthalo-blue"], PAINTS["cad-yellow"]
        mixed = predict_mix_color([blue, yellow], [1, 1])

        b_rgb, y_rgb = lab_to_rgb(blue.lab), lab_to_rgb(yellow.lab)
        naive = RGBColor(
            (b_rgb.r + y_rgb.r) // 2, (b_rgb.g + y_rgb.g) // 2, (b_rgb.b + y_rgb.b) // 2
        )

        # Averaging leaves red sitting in the result, which is what turns the
        # mix muddy. Subtractive mixing absorbs it: both pigments block red, so
        # the product blocks more of it than either average would suggest.
        assert mixed.rgb.r < naive.r, (
            f"pigment mixing should absorb more red than averaging: "
            f"mix r={mixed.rgb.r}, naive r={naive.r}"
        )
        # Green should lead the result more decisively than it does in the average.
        assert (mixed.rgb.g - mixed.rgb.r) > (naive.g - naive.r), (
            f"green should dominate more than in the naive average: {mixed.hex} vs {naive.hex}"
        )

    def test_yellow_and_red_make_orange(self):
        result = predict_mix_color([PAINTS["cad-yellow"], PAINTS["cad-red"]], [1, 1])
        hue = hue_degrees(result.rgb)
        assert 10.0 <= hue <= 55.0, f"hue {hue:.0f} deg is not orange: {result.hex}"

    def test_red_and_blue_make_violet(self):
        result = predict_mix_color([PAINTS["crimson"], PAINTS["ultramarine"]], [1, 1])
        assert result.rgb.b > result.rgb.g, f"not violet-leaning: {result.hex}"
        assert result.rgb.r > result.rgb.g, f"not violet-leaning: {result.hex}"


class TestWhiteAndBlack:
    @pytest.mark.parametrize("paint_id", ["cad-red", "burnt-umber", "ultramarine"])
    def test_white_lightens_substantially(self, paint_id):
        """The case single-constant Kubelka-Munk gets wrong without tinting strength.

        Threshold is +12 L*, which the model currently clears with room on earths
        and blues (+20 to +24) and only just on saturated reds (+13). That gap is
        a known limitation, not a tuning failure: without a separate scattering
        term, white cannot fully overcome a strong absorber. Measured K and S per
        pigment is the real fix. Do not paper over it by lowering this bound.
        """
        paint = PAINTS[paint_id]
        tinted = predict_mix_color([paint, PAINTS["ti-white"]], [1, 1])
        assert tinted.lab.L > paint.lab.L + 12.0, (
            f"white barely lightened {paint.name}: L {paint.lab.L:.1f} -> {tinted.lab.L:.1f}"
        )

    def test_more_white_means_lighter(self):
        red = PAINTS["cad-red"]
        light = predict_mix_color([red, PAINTS["ti-white"]], [1, 4])
        less = predict_mix_color([red, PAINTS["ti-white"]], [1, 1])
        assert light.lab.L > less.lab.L

    def test_white_desaturates(self):
        """A tint is paler, not just brighter."""
        red = PAINTS["cad-red"]
        tinted = predict_mix_color([red, PAINTS["ti-white"]], [1, 3])
        chroma = lambda c: (c.a**2 + c.b**2) ** 0.5
        assert chroma(tinted.lab) < chroma(red.lab)

    def test_black_darkens(self):
        yellow = PAINTS["cad-yellow"]
        shaded = predict_mix_color([yellow, PAINTS["ivory-black"]], [3, 1])
        assert shaded.lab.L < yellow.lab.L - 10.0

    def test_umber_and_white_make_a_tan(self):
        umber = PAINTS["burnt-umber"]
        result = predict_mix_color([umber, PAINTS["ti-white"]], [1, 1])
        # Relative to the umber, not an absolute L: an absolute threshold would
        # silently become a test of the catalog's hex value rather than of mixing.
        assert result.lab.L > umber.lab.L + 20.0, f"too dark for a tint: {result.hex}"
        assert result.rgb.r > result.rgb.g > result.rgb.b, f"not a warm tan: {result.hex}"


class TestTintingStrength:
    def test_stronger_paint_dominates(self):
        """Phthalo blue overpowers white; cerulean does not, at the same ratio."""
        white = PAINTS["ti-white"]
        phthalo = predict_mix_color([PAINTS["phthalo-blue"], white], [1, 8])
        cerulean = predict_mix_color([PAINTS["cerulean"], white], [1, 8])
        assert phthalo.lab.L < cerulean.lab.L, (
            "the stronger pigment should resist being washed out"
        )

    def test_strength_is_what_drives_it(self):
        """Same color, different strength, different outcome — isolating the knob."""
        weak = make_paint("weak", "Weak", "#2e3192", tinting_strength=0.5)
        strong = make_paint("strong", "Strong", "#2e3192", tinting_strength=5.0)
        white = PAINTS["ti-white"]
        assert (
            predict_mix_color([strong, white], [1, 4]).lab.L
            < predict_mix_color([weak, white], [1, 4]).lab.L
        )


class TestProportionsAndApi:
    def test_ratios_are_what_matter_not_magnitudes(self):
        a = predict_mix_color([PAINTS["cad-red"], PAINTS["ti-white"]], [1, 2])
        b = predict_mix_color([PAINTS["cad-red"], PAINTS["ti-white"]], [50, 100])
        assert a.hex == b.hex

    def test_single_paint_returns_itself(self):
        for paint in (PAINTS["cad-red"], PAINTS["ultramarine"], PAINTS["burnt-umber"]):
            result = predict_mix_color([paint], [1])
            assert result.distance_to(paint.lab) < 2.0, f"{paint.name} drifted: {result.hex}"

    def test_zero_parts_excludes_a_paint(self):
        with_zero = predict_mix_color(
            [PAINTS["cad-red"], PAINTS["ultramarine"]], [1, 0]
        )
        alone = predict_mix_color([PAINTS["cad-red"]], [1])
        assert with_zero.hex == alone.hex

    def test_three_way_mixes_work(self):
        result = predict_mix_color(
            [PAINTS["cad-yellow"], PAINTS["ultramarine"], PAINTS["ti-white"]], [2, 1, 1]
        )
        assert 0 <= result.rgb.r <= 255

    def test_describe_reads_like_instructions(self):
        result = predict_mix_color([PAINTS["ultramarine"], PAINTS["ti-white"]], [1, 2])
        text = result.describe()
        assert "1 part Ultramarine Blue" in text
        assert "2 parts Titanium White" in text

    def test_distance_to_target(self):
        red = PAINTS["cad-red"]
        assert predict_mix_color([red], [1]).distance_to(red.lab) < 2.0
        assert predict_mix_color([red], [1]).distance_to(PAINTS["ultramarine"].lab) > 20.0


class TestValidation:
    def test_rejects_empty_mix(self):
        with pytest.raises(ValueError):
            mix_paints([])

    def test_rejects_negative_parts(self):
        with pytest.raises(ValueError):
            mix_paints([MixComponent(paint=PAINTS["cad-red"], parts=-1.0)])

    def test_rejects_all_zero_parts(self):
        with pytest.raises(ValueError):
            predict_mix_color([PAINTS["cad-red"], PAINTS["ti-white"]], [0, 0])

    def test_rejects_mismatched_lengths(self):
        with pytest.raises(ValueError):
            predict_mix_color([PAINTS["cad-red"]], [1, 2])

    def test_mix_reflectances_rejects_mismatch(self):
        r = srgb_to_reflectance((0.5, 0.5, 0.5))
        with pytest.raises(ValueError):
            mix_reflectances([r], [1.0, 2.0])


class TestCatalog:
    def test_ids_are_unique(self):
        ids = [p.id for p in STARTER_ACRYLICS]
        assert len(ids) == len(set(ids))

    def test_every_paint_has_a_sane_lab(self):
        for paint in STARTER_ACRYLICS:
            assert 0.0 <= paint.lab.L <= 100.0
            assert paint.tinting_strength > 0

    def test_white_is_the_lightest_and_black_the_darkest(self):
        lightness = {p.id: p.lab.L for p in STARTER_ACRYLICS}
        assert max(lightness, key=lightness.get) == "ti-white"
        assert min(lightness, key=lightness.get) == "ivory-black"

    def test_catalog_lookup_round_trips(self):
        assert catalog_by_id()["ultramarine"].name == "Ultramarine Blue"

    def test_paints_are_distinguishable(self):
        """No two tubes in the starter set should be perceptually the same."""
        from artmentor.core.color.difference import delta_e

        for i, a in enumerate(STARTER_ACRYLICS):
            for b in STARTER_ACRYLICS[i + 1 :]:
                assert delta_e(a.lab, b.lab) > 5.0, f"{a.name} and {b.name} are too close"


class TestCaching:
    def test_repeated_mixes_are_consistent(self):
        """The spectral solve is cached; caching must not alter results."""
        first = predict_mix_color([PAINTS["cad-red"], PAINTS["ti-white"]], [1, 2])
        second = predict_mix_color([PAINTS["cad-red"], PAINTS["ti-white"]], [1, 2])
        assert first.hex == second.hex
        assert np.allclose(first.reflectance, second.reflectance)