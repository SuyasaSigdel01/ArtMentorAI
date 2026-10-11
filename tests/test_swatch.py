"""Tests for reading paint color out of a photograph.

The tests simulate phone photos: a paint patch in the middle of a frame, a
reference card at the top, a colored cast from the room lighting, and sensor
noise. That setup is what lets `test_grey_card_recovers_the_true_color` assert
something real — that the same paint photographed under tungsten and under open
shade reads as the same color once corrected.
"""

from __future__ import annotations

import numpy as np
import pytest

from artmentor.core.color.space import hex_to_lab
from artmentor.core.swatch import check_shade, extract_swatch, white_balance
from artmentor.core.types import RGBColor

PAINT = "#4a7a3a"
TUNGSTEN = (1.22, 1.0, 0.72)   # warm indoor bulb
OPEN_SHADE = (0.88, 0.97, 1.18)  # cool daylight in shadow

REFERENCE_BAND = (slice(0, 25), slice(0, 200))


def photo(
    paint_hex: str = PAINT,
    cast: tuple[float, float, float] = (1.0, 1.0, 1.0),
    reference_level: int | None = 200,
    noise: float = 3.0,
    seed: int = 0,
) -> np.ndarray:
    """Simulate a photo of a paint swatch with an optional reference card."""
    img = np.zeros((200, 200, 3), dtype=np.uint8)
    img[:, :] = (90, 90, 95)  # the table
    colour = RGBColor.from_hex(paint_hex)
    img[60:160, 50:150] = (colour.r, colour.g, colour.b)
    if reference_level is not None:
        img[0:30, :] = (reference_level,) * 3

    rng = np.random.default_rng(seed)
    lit = img.astype(np.float64) * np.array(cast) + rng.normal(0, noise, img.shape)
    return np.clip(lit, 0, 255).astype(np.uint8)


class TestExtractSwatch:
    def test_reads_a_flat_patch_accurately(self):
        reading = extract_swatch(photo(noise=0.0))
        assert reading.lab.L == pytest.approx(hex_to_lab(PAINT).L, abs=1.0)
        assert reading.is_uniform

    def test_ignores_a_specular_highlight(self):
        """One blown-out spot must not drag the reading lighter."""
        clean = photo(noise=0.0)
        glared = clean.copy()
        glared[95:110, 95:110] = (255, 255, 255)  # a glare spot on wet paint

        assert extract_swatch(glared).lab.L == pytest.approx(
            extract_swatch(clean).lab.L, abs=1.5
        )

    def test_reports_rejected_pixels(self):
        glared = photo(noise=0.0)
        glared[95:110, 95:110] = (255, 255, 255)
        assert extract_swatch(glared).pixels_rejected > 0

    def test_flags_an_unevenly_mixed_patch(self):
        """Two colors smeared together is not a mix, and should be called out."""
        img = photo(noise=0.0)
        img[60:110, 50:150] = (210, 70, 60)  # half the swatch is a different paint
        assert not extract_swatch(img).is_uniform

    def test_honours_an_explicit_region(self):
        img = photo(noise=0.0)
        img[0:30, :] = (255, 0, 0)
        reading = extract_swatch(img, region=(slice(5, 25), slice(5, 195)))
        assert reading.rgb.r > 200
        assert reading.rgb.g < 60

    def test_rejects_a_non_image(self):
        with pytest.raises(ValueError):
            extract_swatch(np.zeros((10, 10), dtype=np.uint8))

    def test_rejects_an_empty_region(self):
        with pytest.raises(ValueError):
            extract_swatch(photo(), region=(slice(0, 0), slice(0, 0)))


class TestWhiteBalance:
    @pytest.mark.parametrize("cast", [TUNGSTEN, OPEN_SHADE])
    def test_grey_card_recovers_the_true_color(self, cast):
        """The headline claim: lighting stops mattering once corrected."""
        truth = hex_to_lab(PAINT)
        result = check_shade(photo(cast=cast), truth, white_region=REFERENCE_BAND)
        assert result.confidence == "calibrated"
        assert result.delta_e < 2.0, f"{result.delta_e:.2f} under {cast}"

    def test_correction_beats_no_correction(self):
        truth = hex_to_lab(PAINT)
        image = photo(cast=TUNGSTEN)
        uncorrected = check_shade(image, truth, correct_lighting=False).delta_e
        corrected = check_shade(image, truth, white_region=REFERENCE_BAND).delta_e
        assert corrected < uncorrected / 2

    def test_a_clipped_reference_is_not_trusted(self):
        """A white card saturates under warm light; saying so beats failing quietly.

        This is why the product asks for grey rather than white. The correction
        still runs, but confidence must drop so the wording hedges.
        """
        image = photo(cast=TUNGSTEN, reference_level=255)
        result = check_shade(image, hex_to_lab(PAINT), white_region=REFERENCE_BAND)
        assert result.confidence == "estimated"
        assert "rough read" in result.message()

    def test_grey_world_is_flagged_as_estimated(self):
        result = check_shade(photo(cast=TUNGSTEN), hex_to_lab(PAINT))
        assert result.confidence == "estimated"

    def test_neutral_light_is_left_alone(self):
        balanced, confidence = white_balance(photo(noise=0.0), REFERENCE_BAND)
        assert confidence == "calibrated"
        assert np.abs(balanced.astype(int) - photo(noise=0.0).astype(int)).mean() < 4

    def test_rejects_an_empty_reference_region(self):
        with pytest.raises(ValueError):
            white_balance(photo(), (slice(0, 0), slice(0, 0)))


class TestCheckShade:
    def test_a_correct_mix_is_accepted(self):
        result = check_shade(photo(noise=0.0), hex_to_lab(PAINT), white_region=REFERENCE_BAND)
        assert result.is_match
        assert result.delta_e < 2.0

    def test_a_wrong_mix_is_rejected(self):
        result = check_shade(
            photo(noise=0.0), hex_to_lab("#b5653f"), white_region=REFERENCE_BAND
        )
        assert not result.is_match
        assert result.delta_e > 5.0

    def test_points_in_the_right_direction(self):
        """Paint photographed too blue should be told to add yellow."""
        result = check_shade(
            photo("#3a6b8a", noise=0.0),
            hex_to_lab("#3a6b3a"),
            white_region=REFERENCE_BAND,
        )
        assert "yellow" in result.hint

    def test_message_hedges_without_calibration(self):
        plain = check_shade(photo(noise=0.0), hex_to_lab(PAINT), white_region=REFERENCE_BAND)
        hedged = check_shade(photo(noise=0.0), hex_to_lab(PAINT))
        assert "rough read" not in plain.message()
        assert "rough read" in hedged.message()

    def test_message_mentions_an_uneven_mix(self):
        img = photo(noise=0.0)
        img[60:110, 50:150] = (210, 70, 60)
        result = check_shade(img, hex_to_lab(PAINT), white_region=REFERENCE_BAND)
        assert "unevenly mixed" in result.message()