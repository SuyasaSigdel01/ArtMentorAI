"""Tests for joining patterns to mixing recipes."""

from __future__ import annotations

import numpy as np
import pytest

from artmentor.core.paints import STARTER_ACRYLICS, catalog_by_id
from artmentor.core.pipeline import PatternParams, generate_pattern
from artmentor.core.recipes import format_mixing_guide, recipes_for_pattern

PAINTS = catalog_by_id()


@pytest.fixture(scope="module")
def pattern():
    h, w = 120, 100
    img = np.zeros((h, w, 3), dtype=np.uint8)
    img[:40] = (120, 170, 220)
    img[40:80] = (80, 140, 70)
    img[80:] = (200, 90, 60)
    rng = np.random.default_rng(1)
    noisy = np.clip(img + rng.normal(0, 2.0, img.shape), 0, 255).astype(np.uint8)
    return generate_pattern(noisy, PatternParams(n_colors=5, max_dimension=120, seed=0))


@pytest.fixture(scope="module")
def recipes(pattern):
    return recipes_for_pattern(pattern, STARTER_ACRYLICS, max_components=3)


class TestCoverage:
    def test_one_recipe_per_legend_entry(self, pattern, recipes):
        """Every printed number must have mixing instructions."""
        assert len(recipes) == len(pattern.palette)
        assert [r.number for r in recipes] == [e.number for e in pattern.palette]

    def test_recipes_follow_pattern_order(self, recipes):
        """Ordered by coverage, so the painter blocks in big areas first."""
        shares = [r.entry.share for r in recipes]
        assert shares == sorted(shares, reverse=True)

    def test_every_recipe_is_mixable(self, recipes):
        for item in recipes:
            assert item.recipe.components
            assert all(c.parts >= 1 for c in item.recipe.components)
            assert all(c.paint.id in PAINTS for c in item.recipe.components)


class TestHonesty:
    def test_verdict_matches_delta_e(self, recipes):
        """The badge shown to a learner must follow from the measured error."""
        for item in recipes:
            assert item.verdict.delta_e == pytest.approx(item.recipe.delta_e)
            assert item.is_achievable == item.verdict.is_acceptable

    def test_unreachable_colors_are_flagged_not_hidden(self):
        """A palette that cannot reach a color must say so, not quietly approximate.

        Two paints, neither near the target: whatever comes back is the closest
        available, and it must be marked approximate.
        """
        h, w = 60, 60
        img = np.full((h, w, 3), (20, 200, 180), dtype=np.uint8)  # vivid turquoise
        img[:30] = (240, 240, 10)
        pattern = generate_pattern(img, PatternParams(n_colors=2, max_dimension=60))
        tiny = [PAINTS["burnt-umber"], PAINTS["ivory-black"]]

        results = recipes_for_pattern(pattern, tiny, max_components=2)
        assert results, "should still return the best available"
        assert any(not r.is_achievable for r in results)

    def test_dict_export_carries_the_accuracy(self, recipes):
        payload = recipes[0].as_dict()
        for key in ("number", "target_hex", "instructions", "delta_e", "achievable"):
            assert key in payload
        assert isinstance(payload["achievable"], bool)


class TestGuide:
    def test_guide_lists_every_color(self, recipes):
        text = format_mixing_guide(recipes)
        for item in recipes:
            assert item.entry.rgb.hex in text

    def test_guide_marks_approximate_colors(self, recipes):
        text = format_mixing_guide(recipes)
        if any(not r.is_achievable for r in recipes):
            assert "approximate" in text

    def test_guide_handles_nothing_to_mix(self):
        assert "No colors" in format_mixing_guide([])


class TestValidation:
    def test_rejects_empty_palette(self, pattern):
        with pytest.raises(ValueError):
            recipes_for_pattern(pattern, [], max_components=2)

    def test_respects_max_components(self, pattern):
        results = recipes_for_pattern(pattern, STARTER_ACRYLICS, max_components=2)
        assert all(len(r.recipe.components) <= 2 for r in results)


class TestPerformance:
    def test_cost_scales_with_colors_not_regions(self):
        """Solving is per palette entry, so image complexity must not matter.

        This guards the caching in RatioSearchMixer: without it, each color
        rebuilt the full candidate table and a pattern took seconds per color
        rather than milliseconds.
        """
        import time

        rng = np.random.default_rng(5)
        busy = rng.integers(0, 255, (200, 200, 3), dtype=np.uint8)
        pattern = generate_pattern(busy, PatternParams(n_colors=6, max_dimension=200))

        recipes_for_pattern(pattern, STARTER_ACRYLICS, max_components=2)  # warm caches
        start = time.time()
        recipes_for_pattern(pattern, STARTER_ACRYLICS, max_components=2)
        elapsed = time.time() - start

        # Generous: the point is to catch a return to per-color table rebuilds,
        # not to pin a number that will drift with hardware.
        assert elapsed < 2.0, f"{elapsed:.2f}s for {len(pattern.palette)} colors"