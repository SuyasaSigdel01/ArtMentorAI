"""Join the pattern generator to the mixing engine.

Until now the two halves of this project have never met. The pipeline produces a
numbered pattern; the mixer knows how to hit a color. Neither is much use to a
painter alone — a pattern that says "region 7 is #4a7a3a" is only actionable if
you already know how to mix that, which is exactly what a beginner does not.

This module answers the real question: for every number in the printed legend,
what goes on the palette?

One deliberate behaviour: recipes carry an honest verdict. The palette a learner
owns cannot reach every color in a photograph, and saying "mix these two and
you'll have it" when the result is visibly different teaches them to distrust
the app. A recipe that lands outside perceptual tolerance is still returned —
it is the best available — but flagged as approximate.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from .color.difference import DeltaEVerdict, verdict_for
from .color.inverse import RatioSearchMixer
from .interfaces import MixRecipe, Paint
from .types import PaletteEntry, Pattern

__all__ = ["PatternRecipe", "recipes_for_pattern", "format_mixing_guide"]


@dataclass(frozen=True)
class PatternRecipe:
    """How to mix one numbered color in a pattern."""

    entry: PaletteEntry
    recipe: MixRecipe
    verdict: DeltaEVerdict

    @property
    def number(self) -> int:
        return self.entry.number

    @property
    def is_achievable(self) -> bool:
        """Can the learner's palette actually reach this color?"""
        return self.verdict.is_acceptable

    def as_dict(self) -> dict:
        return {
            "number": self.entry.number,
            "target_hex": self.entry.rgb.hex,
            "share": round(self.entry.share, 4),
            "mix": [
                {"paint": c.paint.name, "paint_id": c.paint.id, "parts": c.parts}
                for c in self.recipe.components
            ],
            "instructions": self.recipe.describe(),
            "delta_e": round(self.recipe.delta_e, 2),
            "accuracy": self.verdict.band,
            "achievable": self.is_achievable,
        }


def recipes_for_pattern(
    pattern: Pattern,
    palette: Sequence[Paint],
    *,
    max_components: int = 3,
) -> list[PatternRecipe]:
    """Work out a mixing recipe for every number in a pattern's legend.

    Solving is per distinct palette color, not per region, so cost scales with
    the difficulty setting rather than with image complexity — a twelve-color
    pattern costs twelve solves whether it has 20 regions or 2000.
    """
    if not palette:
        raise ValueError("palette is empty: the learner owns no paints")

    mixer = RatioSearchMixer()
    paints = list(palette)

    out: list[PatternRecipe] = []
    for entry in pattern.palette:
        recipe = mixer.solve(entry.lab, paints, max_components=max_components)
        out.append(
            PatternRecipe(
                entry=entry,
                recipe=recipe,
                verdict=verdict_for(recipe.delta_e),
            )
        )
    return out


def format_mixing_guide(recipes: Sequence[PatternRecipe]) -> str:
    """Render recipes as a plain-text guide to sit beside the printed pattern.

    Ordered as the pattern numbers the colors, which is by coverage — so working
    top to bottom means blocking in the largest areas first, the way painters are
    taught to work anyway.
    """
    if not recipes:
        return "No colors to mix."

    width = max(len(r.recipe.describe()) for r in recipes)
    lines = ["  #   COLOR     MIX" + " " * max(0, width - 3) + "  AREA   MATCH", ""]

    for item in recipes:
        marker = " " if item.is_achievable else "~"
        lines.append(
            f"{marker}{item.number:>3}  {item.entry.rgb.hex}  "
            f"{item.recipe.describe():<{width}}  "
            f"{item.entry.share * 100:4.0f}%  {item.verdict.band}"
        )

    approximate = [r for r in recipes if not r.is_achievable]
    if approximate:
        numbers = ", ".join(str(r.number) for r in approximate)
        lines += [
            "",
            f"~ Colors {numbers} are approximate — the closest your paints reach.",
            "  Matching them exactly would need pigments outside this set.",
        ]
    return "\n".join(lines)