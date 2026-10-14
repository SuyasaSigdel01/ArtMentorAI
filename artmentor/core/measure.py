"""Turn photographs of real paint into catalog entries.

The starter catalog in `paints.py` is eyeballed, and four of the project's seven
known limitations trace back to that. This module is how it gets replaced: you
photograph each tube's swatch on the calibration card, and it returns measured
values.

Two things get measured, and they need different photos.

**Colour** comes from one photo of the pure paint. Straightforward: white
balance against the card's grey reference, read the swatch, convert to Lab.

**Tinting strength** cannot be seen in a single swatch. It describes how
strongly a pigment dominates a *mixture*, so measuring it requires a second
photo of that paint mixed with white in a known ratio. The strength is then
whatever value makes the mixing model predict the mixture you actually got.
That is a one-dimensional fit against a real observation, which is a far better
basis than the guesses currently in the catalog.

What this does not fix
----------------------
It records a Lab colour, not a reflectance spectrum. The spectral model still
infers a spectrum from three numbers, so the hue drift on selective pigments
(ultramarine plus yellow reading teal) will remain. Fixing that needs a
spectrophotometer, not a camera. Measuring colour and strength properly is
still a large step forward, and it is the step a phone can actually take.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .color.difference import delta_e
from .color.mixing import predict_mix_color
from .interfaces import Paint
from .swatch import Confidence, SwatchReading, extract_swatch, white_balance
from .types import LabColor

__all__ = [
    "MeasuredPaint",
    "estimate_tinting_strength",
    "measure_paint_color",
    "to_catalog_entry",
]

# Searched range for tinting strength. The low end covers weak earths, the high
# end covers titanium white and the phthalos, which dominate a mix out of all
# proportion to their volume.
_STRENGTH_RANGE = (0.2, 12.0)
_STRENGTH_STEPS = 160


@dataclass(frozen=True)
class MeasuredPaint:
    """A paint colour read off a photograph, with the caveats attached."""

    lab: LabColor
    reading: SwatchReading
    confidence: Confidence
    tinting_strength: float | None = None
    strength_residual: float | None = None  # dE between fitted and observed mix

    @property
    def is_trustworthy(self) -> bool:
        """Should this value go into the catalog?

        Requires a proper reference patch and a swatch that was actually one
        flat colour. An uncalibrated or patchy reading is worse than the guess
        it would replace, because it carries the authority of a measurement.
        """
        return self.confidence == "calibrated" and self.reading.is_uniform

    def warnings(self) -> list[str]:
        out: list[str] = []
        if self.confidence != "calibrated":
            out.append(
                "No usable grey reference — lighting was not corrected. "
                "Re-shoot with the calibration card in frame."
            )
        if not self.reading.is_uniform:
            out.append(
                "The swatch is not one flat colour. Mix more thoroughly, apply "
                "an even layer, and let it dry before photographing."
            )
        if self.strength_residual is not None and self.strength_residual > 5.0:
            out.append(
                f"Tinting strength fit is poor (residual {self.strength_residual:.1f} dE). "
                "Check the mix ratio was accurate."
            )
        return out


def measure_paint_color(
    photo_rgb: np.ndarray,
    *,
    reference_region: tuple[slice, slice],
    swatch_region: tuple[slice, slice],
) -> MeasuredPaint:
    """Read a paint's colour from a photo of it on the calibration card.

    Args:
        photo_rgb: the photograph.
        reference_region: where the card's grey REFERENCE patch is.
        swatch_region: where the painted square is.

    `CalibrationCard.reference_fraction()` gives the reference box for a photo
    cropped to the card; multiply by the image dimensions to get slices.
    """
    corrected, confidence = white_balance(photo_rgb, reference_region)
    reading = extract_swatch(corrected, swatch_region)
    return MeasuredPaint(lab=reading.lab, reading=reading, confidence=confidence)


def estimate_tinting_strength(
    paint_lab: LabColor,
    observed_mix_lab: LabColor,
    *,
    white: Paint,
    paint_parts: float = 1.0,
    white_parts: float = 1.0,
) -> tuple[float, float]:
    """Fit the tinting strength that explains an observed mixture.

    Mix a known ratio of the paint with white, photograph it, and pass the
    measured result here. Returns `(strength, residual_delta_e)`.

    A sweep rather than an optimizer: the search space is one bounded dimension,
    the objective is cheap, and a sweep cannot land in a local minimum or fail
    to converge. 160 steps resolves to about 0.07 of a strength unit, finer than
    the quantity is meaningful to.

    The residual matters as much as the strength. A large one means the model
    cannot reproduce the observed mix at *any* strength, which usually points at
    a mis-measured ratio or a bad photo rather than an unusual pigment.
    """
    if paint_parts <= 0 or white_parts <= 0:
        raise ValueError("both parts must be positive")

    low, high = _STRENGTH_RANGE
    candidates = np.linspace(low, high, _STRENGTH_STEPS)

    best_strength, best_error = float(candidates[0]), float("inf")
    for strength in candidates:
        trial = Paint(
            id="trial",
            name="trial",
            brand="trial",
            lab=paint_lab,
            tinting_strength=float(strength),
        )
        predicted = predict_mix_color([trial, white], [paint_parts, white_parts])
        error = delta_e(predicted.lab, observed_mix_lab)
        if error < best_error:
            best_strength, best_error = float(strength), error

    return best_strength, best_error


def to_catalog_entry(
    paint_id: str,
    name: str,
    measured: MeasuredPaint,
    *,
    brand: str = "measured",
) -> str:
    """Render a measured paint as a line of Python for `paints.py`.

    Emits the Lab value directly rather than a hex string. Hex would round-trip
    through 8-bit sRGB and throw away precision that was just measured — the
    whole point of doing this.
    """
    lab = measured.lab
    strength = measured.tinting_strength if measured.tinting_strength is not None else 1.0
    flag = "" if measured.is_trustworthy else "  # UNVERIFIED — see warnings"
    return (
        f'    Paint(id="{paint_id}", name="{name}", brand="{brand}", '
        f"lab=LabColor({lab.L:.2f}, {lab.a:.2f}, {lab.b:.2f}), "
        f"tinting_strength={strength:.2f}),{flag}"
    )