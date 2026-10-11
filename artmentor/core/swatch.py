"""Read a paint color out of a photograph.

This is the input side of the "is this shade right?" feature. The learner mixes
paint, photographs it, and the app says whether it matches and which way to
correct it. The judging half already exists — `verdict_for` and
`mix_direction_hint` in `color/difference.py`. What is hard is the step before:
getting a color out of a photo that is worth judging at all.

Why a photo's RGB is not the paint's color
-------------------------------------------
A camera reports light, not pigment. The same purple photographs warm under a
tungsten bulb, cold under an overcast sky, and different again after the phone's
auto white balance has had an opinion. The error easily exceeds the difference
the learner is asking about, so a naive pixel reading produces advice that is
confidently wrong — which is worse than no advice, because they will act on it.

Two defences, in order of preference:

1. **A neutral grey reference in the frame.** Knowing what one patch *should*
   be, the illumination can be divided out. This turns an unsolvable problem
   into a solved one at the cost of one instruction to the user.

   It must be **grey, not white**. A white patch is already near the top of the
   sensor's range, so any warm light pushes its red channel past 255 and it
   clips. The measured reference is then wrong in exactly the way the
   correction is meant to fix, and the whole thing fails silently. Measured on
   simulated tungsten light: a white patch corrected an error of 5.3 dE down to
   only 4.6, while a mid-grey patch took the same error to 0.9. `white_balance`
   detects clipping and downgrades its confidence rather than quietly lying.

2. **Grey-world assumption.** With no reference, assume the scene averages to
   neutral. Often roughly true, sometimes badly wrong — a photo dominated by one
   paint breaks it completely.

Whichever is used, the result carries a `confidence`, and callers are expected
to soften their language when it is not `calibrated`. Honest uncertainty beats
confident nonsense.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

from .color.difference import DeltaEVerdict, delta_e, mix_direction_hint, verdict_for
from .color.space import rgb_to_lab_array
from .types import LabColor, RGBColor

__all__ = [
    "ShadeCheck",
    "SwatchReading",
    "check_shade",
    "extract_swatch",
    "white_balance",
]

Confidence = Literal["calibrated", "estimated", "none"]

# A channel at or above this 8-bit value may have been clipped by the sensor,
# which makes it useless as a reference: its true value is unknowable.
_SATURATION_LIMIT = 250

# Pixels further than this from the region's median are treated as not-paint:
# specular highlights, shadows, a stray brush hair, the palette showing through.
_OUTLIER_DELTA_E = 12.0


@dataclass(frozen=True)
class SwatchReading:
    """A color measured from an image region."""

    rgb: RGBColor
    lab: LabColor
    pixels_used: int
    pixels_rejected: int
    spread: float  # mean CIEDE2000 from the median; high means a patchy mix

    @property
    def is_uniform(self) -> bool:
        """Was the sampled area actually one flat color?

        A high spread usually means the paint is unevenly mixed, or the crop
        caught an edge. Either way the reading describes an average that may not
        exist anywhere on the palette.
        """
        return self.spread < 5.0


@dataclass(frozen=True)
class ShadeCheck:
    """The verdict on a photographed mix against its target."""

    measured: SwatchReading
    target: LabColor
    verdict: DeltaEVerdict
    hint: str
    confidence: Confidence

    @property
    def delta_e(self) -> float:
        return self.verdict.delta_e

    @property
    def is_match(self) -> bool:
        return self.verdict.is_acceptable

    def message(self) -> str:
        """What to actually say to the learner."""
        text = self.verdict.message
        if self.confidence != "calibrated":
            text += (
                " (Lighting is uncorrected, so treat this as a rough read — "
                "include a grey card in the shot for an accurate one.)"
            )
        if not self.measured.is_uniform:
            text += " The paint looks unevenly mixed — blend it more and retry."
        return text


def _central_region(shape: tuple[int, int], fraction: float) -> tuple[slice, slice]:
    """A centered box covering `fraction` of each dimension."""
    h, w = shape
    dh, dw = int(h * fraction / 2), int(w * fraction / 2)
    cy, cx = h // 2, w // 2
    return (
        slice(max(0, cy - dh), min(h, cy + dh + 1)),
        slice(max(0, cx - dw), min(w, cx + dw + 1)),
    )


def extract_swatch(
    rgb: np.ndarray,
    region: tuple[slice, slice] | None = None,
    *,
    fraction: float = 0.5,
) -> SwatchReading:
    """Measure the dominant color of a region, ignoring what is not paint.

    Uses the median rather than the mean, then rejects pixels far from it and
    re-averages. A single specular highlight can drag a mean several delta-E
    units; the median does not notice it.
    """
    if rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError(f"expected an (H, W, 3) image, got {rgb.shape}")

    box = region if region is not None else _central_region(rgb.shape[:2], fraction)
    crop = rgb[box]
    if crop.size == 0:
        raise ValueError("the selected region contains no pixels")

    lab = rgb_to_lab_array(crop).reshape(-1, 3)
    median = np.median(lab, axis=0)

    # Approximate distance for the outlier pass: Euclidean in Lab is close enough
    # to decide what is paint and what is a highlight, and far cheaper.
    distances = np.linalg.norm(lab - median, axis=1)
    keep = distances <= _OUTLIER_DELTA_E

    if keep.sum() < max(8, int(0.05 * lab.shape[0])):
        # Nothing agrees with anything: the crop is probably not a flat swatch.
        keep = np.ones(lab.shape[0], dtype=bool)

    kept = lab[keep]
    mean_lab = kept.mean(axis=0)
    measured = LabColor.from_array(mean_lab)

    spread = float(np.mean([delta_e(mean_lab, row) for row in kept[:: max(1, len(kept) // 200)]]))

    from .color.space import lab_to_rgb

    return SwatchReading(
        rgb=lab_to_rgb(measured),
        lab=measured,
        pixels_used=int(keep.sum()),
        pixels_rejected=int((~keep).sum()),
        spread=spread,
    )


def white_balance(
    rgb: np.ndarray,
    white_region: tuple[slice, slice] | None = None,
) -> tuple[np.ndarray, Confidence]:
    """Divide the illumination out of an image.

    With `white_region`, the correction is anchored to a patch the caller knows
    is neutral — ideally a mid-grey card, not white (see the module docstring:
    a white patch clips under warm light and the correction fails silently).
    Without one, the grey-world assumption is used and the result is flagged as
    estimated.

    If the reference turns out to be clipped, the correction is still applied —
    it is better than nothing — but confidence drops to "estimated" so callers
    hedge their wording.

    Gains are applied in *linear* light. Scaling sRGB values directly is a common
    shortcut and is wrong: the transfer curve is not linear, so the correction
    comes out too strong in shadows and too weak in highlights.
    """
    if rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError(f"expected an (H, W, 3) image, got {rgb.shape}")

    linear = _srgb_to_linear(rgb.astype(np.float64) / 255.0)

    if white_region is not None:
        patch = linear[white_region]
        if patch.size == 0:
            raise ValueError("the white reference region contains no pixels")
        flat = patch.reshape(-1, 3)
        reference = np.median(flat, axis=0)

        # Check for clipping against the ORIGINAL 8-bit values: once linearized,
        # a saturated channel is no longer recognisable as such.
        original = rgb[white_region].reshape(-1, 3)
        clipped = np.median(original, axis=0) >= _SATURATION_LIMIT
        confidence: Confidence = "estimated" if clipped.any() else "calibrated"
    else:
        reference = linear.reshape(-1, 3).mean(axis=0)
        confidence = "estimated"

    if np.any(reference <= 1e-6):
        # A channel with no signal cannot be corrected without inventing data.
        return rgb.copy(), "none"

    # Preserve overall brightness: only the color cast is removed.
    gains = reference.mean() / reference
    corrected = np.clip(linear * gains, 0.0, 1.0)

    return (_linear_to_srgb(corrected) * 255.0).round().astype(np.uint8), confidence


def _srgb_to_linear(values: np.ndarray) -> np.ndarray:
    return np.where(values <= 0.04045, values / 12.92, ((values + 0.055) / 1.055) ** 2.4)


def _linear_to_srgb(values: np.ndarray) -> np.ndarray:
    return np.where(
        values <= 0.0031308, values * 12.92, 1.055 * np.power(values, 1 / 2.4) - 0.055
    )


def check_shade(
    photo_rgb: np.ndarray,
    target: LabColor,
    *,
    swatch_region: tuple[slice, slice] | None = None,
    white_region: tuple[slice, slice] | None = None,
    correct_lighting: bool = True,
) -> ShadeCheck:
    """Judge a photographed paint mix against the color it was meant to be.

    Args:
        photo_rgb: the learner's photo.
        target: the color they were aiming for.
        swatch_region: where the paint is. Defaults to the middle of the frame,
            which matches the obvious instruction — "fill the middle with paint".
        white_region: where a known-neutral reference is, if present.
        correct_lighting: set False only when the image is already calibrated.
    """
    confidence: Confidence = "none"
    working = photo_rgb

    if correct_lighting:
        working, confidence = white_balance(photo_rgb, white_region)

    reading = extract_swatch(working, swatch_region)
    difference = delta_e(reading.lab, target)
    hint = mix_direction_hint(reading.lab, target)

    return ShadeCheck(
        measured=reading,
        target=target,
        verdict=verdict_for(difference, hint),
        hint=hint,
        confidence=confidence,
    )