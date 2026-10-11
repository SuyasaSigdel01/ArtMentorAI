"""Tests for inverse pigment mixing.

The most important test here is `test_prediction_matches_the_forward_engine`.
The solver scores thousands of candidate recipes through a fast vectorized path,
while `mix_paints` computes one mix the straightforward way. Those two must agree
exactly, or the solver is optimizing against a model the product does not use,
and the delta-E it reports to a learner is fiction.
"""

from __future__ import annotations

import pytest

from artmentor.core.color.difference import delta_e
from artmentor.core.color.inverse import RatioSearchMixer, solve_mix
from artmentor.core.color.mixing import predict_mix_color
from artmentor.core.color.space import hex_to_lab
from artmentor.core.paints import STARTER_ACRYLICS, catalog_by_id

PAINTS = catalog_by_id()
PALETTE = list(STARTER_ACRYLICS)


class TestConsistencyWithForwardEngine:
    def test_prediction_matches_the_forward_engine(self):
        """The search's own prediction must equal the real forward mix."""
        for target_hex in ("#4a7a3a", "#8fb8de", "#b5653f", "#3a3a3c"):
            recipe = solve_mix(hex_to_lab(target_hex), PALETTE, max_components=3)
            forward = predict_mix_color(
                [c.paint for c in recipe.components],
                [c.parts for c in recipe.components],
            )
            assert delta_e(recipe.predicted_lab, forward.lab) < 0.1, (
                f"solver and forward engine disagree for {target_hex}: "
                f"{recipe.predicted_lab} vs {forward.lab}"
            )

    def test_reported_delta_e_is_honest(self):
        """delta_e must describe the recipe actually returned, penalties excluded."""
        target = hex_to_lab("#6b6b35")
        recipe = solve_mix(target, PALETTE, max_components=3)
        forward = predict_mix_color(
            [c.paint for c in recipe.components], [c.parts for c in recipe.components]
        )
        assert recipe.delta_e == pytest.approx(delta_e(forward.lab, target), abs=0.1)


class TestAccuracy:
    def test_a_palette_color_resolves_to_itself(self):
        """Asking for a paint you own should return that paint, not a mixture."""
        for paint_id in ("cad-red", "ultramarine", "burnt-umber", "cad-yellow"):
            paint = PAINTS[paint_id]
            recipe = solve_mix(paint.lab, PALETTE, max_components=3)
            assert len(recipe.components) == 1, f"{paint.name} should need no mixing"
            assert recipe.components[0].paint.id == paint_id
            assert recipe.delta_e < 1.0

    @pytest.mark.parametrize(
        "target_hex",
        ["#4a7a3a", "#b5653f", "#4a2c6b", "#3a3a3c", "#6b6b35", "#e8b998"],
    )
    def test_mixable_targets_come_out_close(self, target_hex):
        """Colors well inside the palette's reach should land under dE 5."""
        recipe = solve_mix(hex_to_lab(target_hex), PALETTE, max_components=3)
        assert recipe.delta_e < 5.0, f"{target_hex}: {recipe.delta_e:.2f} — {recipe.describe()}"

    def test_pale_tints_are_the_weak_spot(self):
        """Documents a real limitation rather than pretending it is not there.

        Very pale tints of saturated pigments are where the forward model is
        least accurate: a spectrum inferred from sRGB is broader than a real
        pigment's, so white tints drift in hue. Pale pink is reachable only to
        about dE 4, and no amount of extra white fixes it — at 100:1 the mix is
        still magenta where the target is warm.

        The bound is loose on purpose. It exists to catch a regression into
        nonsense, and to fail loudly if measured paint data ever makes this
        genuinely good, so the comment above gets deleted.
        """
        recipe = solve_mix(hex_to_lab("#f7d9d6"), PALETTE, max_components=2)
        assert recipe.delta_e < 8.0
        assert recipe.delta_e > 1.0, "pale tints got accurate — update this test and the docs"

    def test_more_components_never_hurt(self):
        """Allowing three paints cannot do worse than allowing two."""
        target = hex_to_lab("#5b6b7a")
        two = solve_mix(target, PALETTE, max_components=2)
        three = solve_mix(target, PALETTE, max_components=3)
        assert three.delta_e <= two.delta_e + 1.0


class TestUsability:
    def test_respects_max_components(self):
        for limit in (1, 2, 3):
            recipe = solve_mix(hex_to_lab("#6b6b35"), PALETTE, max_components=limit)
            assert len(recipe.components) <= limit

    def test_parts_are_whole_numbers(self):
        """A recipe nobody can measure out is not an answer."""
        recipe = solve_mix(hex_to_lab("#8fb8de"), PALETTE, max_components=3)
        for component in recipe.components:
            assert component.parts >= 1
            assert component.parts == int(component.parts)

    def test_prefers_fewer_paints_on_a_near_tie(self):
        """Two tubes beat three when the accuracy difference is imperceptible."""
        mixer = RatioSearchMixer(extra_component_penalty=5.0)
        recipe = mixer.solve(hex_to_lab("#6b6b35"), PALETTE, max_components=3)
        assert len(recipe.components) <= 2

    def test_describe_reads_like_palette_instructions(self):
        recipe = solve_mix(hex_to_lab("#e8b998"), PALETTE, max_components=2)
        text = recipe.describe()
        assert "part" in text
        assert any(c.paint.name in text for c in recipe.components)

    def test_uses_only_paints_from_the_given_palette(self):
        """A learner can only mix what they own."""
        small = [PAINTS["ti-white"], PAINTS["ultramarine"], PAINTS["cad-red"]]
        recipe = solve_mix(hex_to_lab("#4a7a3a"), small, max_components=3)
        allowed = {p.id for p in small}
        assert all(c.paint.id in allowed for c in recipe.components)


class TestDeterminismAndValidation:
    def test_same_target_gives_same_recipe(self):
        target = hex_to_lab("#b5653f")
        first = solve_mix(target, PALETTE, max_components=3)
        second = solve_mix(target, PALETTE, max_components=3)
        assert first.describe() == second.describe()
        assert first.delta_e == pytest.approx(second.delta_e)

    def test_rejects_empty_palette(self):
        with pytest.raises(ValueError):
            solve_mix(hex_to_lab("#808080"), [], max_components=2)

    def test_rejects_zero_components(self):
        with pytest.raises(ValueError):
            solve_mix(hex_to_lab("#808080"), PALETTE, max_components=0)

    def test_single_paint_palette_still_answers(self):
        recipe = solve_mix(hex_to_lab("#4a7a3a"), [PAINTS["ti-white"]], max_components=3)
        assert len(recipe.components) == 1
        assert recipe.components[0].paint.id == "ti-white"