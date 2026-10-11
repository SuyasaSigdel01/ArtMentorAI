"""Inverse pigment mixing: which paints, in what ratio, hit a target color?

The forward engine in `mixing.py` answers "what do these make?". This answers the
question the product actually asks: a learner points at a leaf in their reference
photo, and needs to be told what to squeeze onto the palette.

Why enumeration rather than continuous optimization
---------------------------------------------------
The obvious approach is to optimize concentrations as continuous variables. It
converges fine and produces answers like "0.6187 ultramarine, 0.3813 white",
which is useless: nobody can measure that with a palette knife. Rounding
afterwards then moves the result somewhere the optimizer never evaluated, so the
reported delta-E is a lie.

Instead this searches *only over ratios a human can actually execute* — small
integer parts like 1:2 or 2:1:1. Every candidate is a recipe someone can follow,
and its predicted color is the real predicted color of that real recipe. The
search space is small enough to evaluate exhaustively and vectorized, so this is
also faster than running an optimizer per paint combination.

Preferences beyond accuracy
---------------------------
The closest match is not always the best advice. Two paints beat three, and 1:2
beats 5:3, even at slightly worse accuracy — fewer components and rounder
numbers mean fewer chances to get it wrong by hand. Small penalties encode that,
sized well below the threshold where a human notices a color difference at all.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations, product
from math import gcd, log2

import colour
import numpy as np
from skimage import color as _skcolor

from ..interfaces import MixComponent, MixRecipe, Paint, PigmentMixer
from ..types import LabColor
from .mixing import paint_reflectance
from .spectral import ks_to_reflectance, reflectance_to_ks, reflectance_to_xyz

__all__ = ["RatioSearchMixer", "solve_mix"]


# Allowed part values, by number of paints in the recipe. Not a uniform range:
# the ladder has to reach lopsided ratios like 20:1, because tinting strength
# means a strong pigment is already dominant at one part. Without the high end,
# a pale tint is inexpressible — the search jumps from pure white straight past
# the target, and settles for pure white as the "closest" option.
#
# The values are ones a person can actually portion out by eye. Three-paint
# recipes stop at 8: "6 parts A, 1 part B, 3 parts C" is near the limit of what
# anyone will follow at a palette.
_PART_LADDER = {
    1: (1,),
    2: (1, 2, 3, 4, 5, 6, 8, 10, 12, 16, 20, 30, 40),
    3: (1, 2, 3, 4, 6, 8),
}

# Penalties, in delta-E units. Both sit well under 1.0, the threshold where a
# color difference becomes perceptible at all, so neither can trade away a
# visible improvement in accuracy — they only break near-ties.
_EXTRA_COMPONENT_PENALTY = 0.6
# Scored on the logarithm of the largest part, not the sum. A linear penalty
# would punish 20:1 so heavily that pale tints become unreachable again, which
# is the exact bug the ladder above exists to fix.
_RATIO_PENALTY = 0.08


def _ratios_for(n: int) -> list[tuple[int, ...]]:
    """Every distinct ratio over `n` components, drawn from the part ladder.

    Ratios sharing a common factor are skipped: 2:2 is 1:1 with extra arithmetic.
    """
    ladder = _PART_LADDER.get(n, (1, 2, 3))
    out: list[tuple[int, ...]] = []
    for parts in product(ladder, repeat=n):
        divisor = parts[0]
        for value in parts[1:]:
            divisor = gcd(divisor, value)
        if divisor == 1:
            out.append(parts)
    return out


@dataclass(frozen=True)
class _Candidate:
    paints: tuple[int, ...]
    parts: tuple[int, ...]


class RatioSearchMixer(PigmentMixer):
    """Exhaustive search over human-executable paint ratios.

    Usage:
        mixer = RatioSearchMixer()
        recipe = mixer.solve(target_lab, my_paints, max_components=2)
        print(recipe.describe(), recipe.delta_e)
    """

    def __init__(
        self,
        *,
        extra_component_penalty: float = _EXTRA_COMPONENT_PENALTY,
        ratio_penalty: float = _RATIO_PENALTY,
    ) -> None:
        self.extra_component_penalty = extra_component_penalty
        self.ratio_penalty = ratio_penalty
        # Candidate recipes and their predicted colors depend only on the palette,
        # not on the target. Solving a whole pattern means many targets against
        # one palette, so this is built once and reused.
        self._prepared: dict[tuple, tuple[list[_Candidate], np.ndarray, np.ndarray]] = {}

    def _prepare(
        self, palette: list[Paint], max_components: int
    ) -> tuple[list[_Candidate], np.ndarray, np.ndarray]:
        """Enumerate every recipe for this palette and predict each one's color."""
        key = (
            tuple((p.id, p.lab.L, p.lab.a, p.lab.b, p.tinting_strength) for p in palette),
            max_components,
        )
        cached = self._prepared.get(key)
        if cached is not None:
            return cached

        # Spectra are cached per color, so this costs one solve per distinct
        # paint the first time and nothing afterwards.
        ks_per_paint = np.stack([reflectance_to_ks(paint_reflectance(p)) for p in palette])
        strengths = np.array([p.tinting_strength for p in palette], dtype=np.float64)

        candidates: list[_Candidate] = []
        for size in range(1, max_components + 1):
            ratios = _ratios_for(size)
            for chosen in combinations(range(len(palette)), size):
                for parts in ratios:
                    candidates.append(_Candidate(chosen, parts))

        if not candidates:
            raise ValueError("no candidate recipes could be built")

        # One weight matrix for every recipe at once, so the whole search becomes
        # a single matrix multiply rather than thousands of Python-level mixes.
        weights = np.zeros((len(candidates), len(palette)), dtype=np.float64)
        for row, candidate in enumerate(candidates):
            for index, part in zip(candidate.paints, candidate.parts):
                weights[row, index] = part * strengths[index]
        weights /= weights.sum(axis=1, keepdims=True)

        reflectance = ks_to_reflectance(weights @ ks_per_paint)
        srgb = np.clip(colour.XYZ_to_sRGB(reflectance_to_xyz(reflectance)), 0.0, 1.0)

        # Round to 8-bit before measuring. The learner sees a mixed paint, and
        # the engine's own forward prediction is 8-bit, so scoring against a
        # higher-precision value would report an accuracy that does not exist.
        srgb_8bit = np.round(srgb * 255.0) / 255.0
        lab = _skcolor.rgb2lab(srgb_8bit.reshape(-1, 1, 3)).reshape(-1, 3)

        complexity = np.array(
            [
                self.extra_component_penalty * (len(c.paints) - 1)
                + self.ratio_penalty * log2(max(c.parts))
                for c in candidates
            ],
            dtype=np.float64,
        )

        self._prepared[key] = (candidates, lab, complexity)
        return candidates, lab, complexity

    def solve(
        self,
        target: LabColor,
        palette: list[Paint],
        max_components: int = 3,
    ) -> MixRecipe:
        if not palette:
            raise ValueError("palette is empty")
        if max_components < 1:
            raise ValueError("max_components must be at least 1")
        max_components = min(max_components, len(palette), 3)

        candidates, lab, complexity = self._prepare(palette, max_components)

        target_lab = np.broadcast_to(target.as_array(), lab.shape)
        distances = np.asarray(
            colour.difference.delta_E_CIE2000(lab, target_lab), dtype=np.float64
        )

        best = int(np.argmin(distances + complexity))
        winner = candidates[best]

        return MixRecipe(
            components=tuple(
                MixComponent(paint=palette[index], parts=float(part))
                for index, part in zip(winner.paints, winner.parts)
            ),
            predicted_lab=LabColor.from_array(lab[best]),
            # The honest accuracy of the chosen recipe, with no penalty folded in.
            delta_e=float(distances[best]),
        )

def solve_mix(
    target: LabColor, palette: list[Paint], max_components: int = 3
) -> MixRecipe:
    """Convenience wrapper for the default mixer."""
    return RatioSearchMixer().solve(target, palette, max_components)