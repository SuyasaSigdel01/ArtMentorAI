"""Perceptual color difference (CIEDE2000) and the advice it drives.

The verdict bands below are the contract between the math and the product voice.
When the shade-checking feature tells a learner "very close, good enough to paint
with", that sentence traces back to a number computed here — not to a language
model's impression of a photo. Keep it that way: a model may rephrase a verdict,
but must never originate one.
"""

from __future__ import annotations

from dataclasses import dataclass

import colour
import numpy as np

from ..types import LabColor

__all__ = [
    "DeltaEVerdict",
    "delta_e",
    "delta_e_matrix",
    "mix_direction_hint",
    "verdict_for",
]


def delta_e(a, b) -> float:
    """CIEDE2000 difference between two Lab colors.

    Accepts LabColor or any 3-element array-like.
    """
    av = a.as_array() if isinstance(a, LabColor) else np.asarray(a, dtype=np.float64)
    bv = b.as_array() if isinstance(b, LabColor) else np.asarray(b, dtype=np.float64)
    return float(colour.difference.delta_E_CIE2000(av.reshape(3), bv.reshape(3)))


def delta_e_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Pairwise CIEDE2000 between two sets of Lab colors.

    `a` is (N, 3), `b` is (M, 3); returns (N, M). Used for palette matching, where
    N and M are both small — this is not optimized for per-pixel use.
    """
    a = np.asarray(a, dtype=np.float64).reshape(-1, 1, 3)
    b = np.asarray(b, dtype=np.float64).reshape(1, -1, 3)
    a_b = np.broadcast_to(a, (a.shape[0], b.shape[1], 3))
    b_b = np.broadcast_to(b, (a.shape[0], b.shape[1], 3))
    return np.asarray(colour.difference.delta_E_CIE2000(a_b, b_b), dtype=np.float64)


@dataclass(frozen=True)
class DeltaEVerdict:
    """How a measured difference should be communicated to a learner."""

    band: str          # machine-readable key
    delta_e: float
    is_acceptable: bool
    message: str       # may contain a {hint} placeholder for a direction


# Thresholds are the standard CIEDE2000 interpretation bands. The wording is
# deliberately encouraging: this product talks to beginners, and "quite far off"
# is as negative as it should ever get.
_BANDS = (
    (1.0, "match", True, "That's a match."),
    (2.0, "very_close", True, "Very close — good enough to paint with."),
    (5.0, "close", False, "Close. {hint} would nail it."),
    (10.0, "off", False, "Not there yet — {hint}."),
    (float("inf"), "far", False, "Quite far off. Easier to restart the mix than correct it."),
)


def verdict_for(difference: float, hint: str | None = None) -> DeltaEVerdict:
    """Map a CIEDE2000 value onto a band and a sentence."""
    for threshold, band, acceptable, template in _BANDS:
        if difference < threshold:
            message = template.replace("{hint}", hint or "a small adjustment")
            return DeltaEVerdict(band, float(difference), acceptable, message)
    raise AssertionError("unreachable: bands end at infinity")


def mix_direction_hint(current: LabColor, target: LabColor) -> str:
    """Translate a Lab delta into paint the learner should add.

    The Lab axes map cleanly onto painters' language:
      L*  lightness   -> white / darker
      a*  green..red  -> red / green
      b*  blue..yellow-> yellow / blue

    Only the dominant axis is reported. Telling a beginner to adjust three things
    at once guarantees they adjust none of them correctly.
    """
    dL = target.L - current.L
    da = target.a - current.a
    db = target.b - current.b

    # Weight lightness slightly lower: it is the easiest axis to overshoot with
    # white, and chroma errors read as "wrong color" more strongly to the eye.
    candidates = [
        (abs(dL) * 0.8, "add a little white" if dL > 0 else "add a touch of a darker tone"),
        (abs(da), "add more red" if da > 0 else "add more green"),
        (abs(db), "add more yellow" if db > 0 else "add more blue"),
    ]
    magnitude, hint = max(candidates, key=lambda c: c[0])
    if magnitude < 0.5:
        return "a very small adjustment"
    return hint
