"""Forward pigment mixing: given paints and proportions, what color results?

This is the forward half of the mixing problem. The product ultimately needs the
inverse — "which paints, in what ratio, hit this target?" — but the inverse
solver optimizes over repeated forward evaluations, so this has to be right and
fast first.

Two approximations are in force, both deliberate and both documented here
because they bound the quality of every answer this module gives:

**Single-constant Kubelka-Munk.** Full K-M tracks absorption (K) and scattering
(S) separately per pigment. Recovering both from an sRGB value is impossible, so
K/S is mixed as one quantity, which implicitly assumes every pigment scatters
equally. That assumption is worst exactly where it is most visible: titanium
white is an extreme scatterer, and under the naive model it barely lightens a
mix at all.

**Tinting strength as the correction.** Rather than pretending otherwise, each
paint carries a `tinting_strength` that scales its effective concentration. This
is the painter's own term for how strongly a color dominates a mixture, and it
restores the behaviour the single-constant model loses. Titanium white needs a
high value (~6) to lighten the way real white does.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from ..interfaces import MixComponent, Paint
from ..types import LabColor, RGBColor
from .difference import delta_e
from .space import rgb_to_lab
from .spectral import (
    ks_to_reflectance,
    reflectance_to_ks,
    reflectance_to_srgb,
    srgb_to_reflectance,
)

__all__ = ["MixResult", "mix_paints", "mix_reflectances", "predict_mix_color"]


@dataclass(frozen=True)
class MixResult:
    """The predicted outcome of mixing paints."""

    rgb: RGBColor
    lab: LabColor
    reflectance: np.ndarray
    components: tuple[MixComponent, ...]

    @property
    def hex(self) -> str:
        return self.rgb.hex

    def describe(self) -> str:
        return " + ".join(
            f"{c.parts:g} part{'s' if c.parts != 1 else ''} {c.paint.name}"
            for c in self.components
        )

    def distance_to(self, target: LabColor) -> float:
        """CIEDE2000 between this mix and a target color."""
        return delta_e(self.lab, target)


def mix_reflectances(reflectances: Sequence[np.ndarray], weights: Sequence[float]) -> np.ndarray:
    """Combine reflectance spectra by Kubelka-Munk, weighted by concentration.

    Weights are normalized, so only their ratio matters.
    """
    if len(reflectances) != len(weights):
        raise ValueError("reflectances and weights must be the same length")
    if not reflectances:
        raise ValueError("nothing to mix")

    w = np.asarray(weights, dtype=np.float64)
    if np.any(w < 0):
        raise ValueError("weights must be non-negative")
    total = w.sum()
    if total <= 0:
        raise ValueError("weights must not sum to zero")
    w = w / total

    ks = np.zeros_like(np.asarray(reflectances[0], dtype=np.float64))
    for weight, reflectance in zip(w, reflectances):
        ks += weight * reflectance_to_ks(reflectance)

    return ks_to_reflectance(ks)


def mix_paints(components: Sequence[MixComponent]) -> MixResult:
    """Predict the color produced by mixing paints in the given proportions.

    Each component's effective concentration is its parts scaled by the paint's
    tinting strength, which is what makes a little phthalo blue overwhelm a lot
    of white, exactly as it does on a real palette.
    """
    if not components:
        raise ValueError("no components to mix")

    weights: list[float] = []
    reflectances: list[np.ndarray] = []
    for component in components:
        if component.parts < 0:
            raise ValueError(f"negative parts for {component.paint.name}")
        weights.append(component.parts * component.paint.tinting_strength)
        reflectances.append(paint_reflectance(component.paint))

    mixed = mix_reflectances(reflectances, weights)
    srgb = reflectance_to_srgb(mixed)
    rgb = RGBColor(*(int(round(float(v) * 255.0)) for v in srgb))

    return MixResult(
        rgb=rgb,
        lab=rgb_to_lab(rgb),
        reflectance=mixed,
        components=tuple(components),
    )


def paint_reflectance(paint: Paint) -> np.ndarray:
    """Reflectance spectrum for a paint, derived from its Lab value.

    When measured spectral data becomes available per paint, this is the single
    function to change — nothing downstream needs to know the difference.
    """
    from .space import lab_to_rgb  # local import keeps the module import graph flat

    rgb = lab_to_rgb(paint.lab)
    return srgb_to_reflectance((rgb.r / 255.0, rgb.g / 255.0, rgb.b / 255.0))


def predict_mix_color(paints: Sequence[Paint], parts: Sequence[float]) -> MixResult:
    """Convenience wrapper over `mix_paints` for plain paint/ratio pairs."""
    if len(paints) != len(parts):
        raise ValueError("paints and parts must be the same length")
    return mix_paints([MixComponent(paint=p, parts=float(n)) for p, n in zip(paints, parts)])
