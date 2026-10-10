"""End-to-end tests for the pattern pipeline.

The most valuable assertion in this file is `test_label_points_fall_inside_region`.
It is the one that catches the centroid bug — using a region's centroid to place
its number works on every convex test shape and silently puts the digit in the
wrong region for crescents and L-shapes, which is most of a real painting.
"""

from __future__ import annotations

import numpy as np
import pytest

from artmentor.core.imaging.segment import build_regions, smooth_label_map
from artmentor.core.pipeline import PIPELINE_VERSION, PatternParams, generate_pattern


@pytest.fixture(scope="module")
def sample_image() -> np.ndarray:
    """A small scene with flat areas, a gradient, a thin bar, and a hole."""
    h, w = 220, 180
    img = np.zeros((h, w, 3), dtype=np.uint8)

    # Vertical gradient background.
    for y in range(h):
        t = y / h
        img[y, :] = (int(120 + 100 * t), int(170 + 60 * t), int(220 + 20 * t))

    img[40:150, 30:140] = (210, 70, 60)      # a big block
    img[70:100, 60:110] = (250, 250, 250)    # a hole inside it
    img[78:92, 72:98] = (20, 20, 20)         # a hole inside the hole
    img[170:178, 10:170] = (90, 60, 40)      # a thin bar

    rng = np.random.default_rng(3)
    noisy = img.astype(np.float64) + rng.normal(0, 3.0, img.shape)
    return np.clip(noisy, 0, 255).astype(np.uint8)


@pytest.fixture(scope="module")
def pattern(sample_image):
    return generate_pattern(sample_image, PatternParams(n_colors=8, max_dimension=200, seed=1))


class TestStructure:
    def test_records_version_and_params(self, pattern):
        assert pattern.pipeline_version == PIPELINE_VERSION
        assert pattern.params["n_colors"] == 8
        assert pattern.params["seed"] == 1

    def test_dimensions_are_consistent(self, pattern):
        assert pattern.label_map.shape == (pattern.height, pattern.width)
        assert pattern.region_map.shape == (pattern.height, pattern.width)

    def test_respects_max_dimension(self, pattern):
        assert max(pattern.width, pattern.height) <= 200

    def test_palette_is_within_requested_size(self, pattern):
        assert 0 < len(pattern.palette) <= 8

    def test_palette_is_ordered_by_coverage(self, pattern):
        shares = [entry.share for entry in pattern.palette]
        assert shares == sorted(shares, reverse=True)

    def test_numbers_are_contiguous_from_one(self, pattern):
        assert [e.number for e in pattern.palette] == list(range(1, len(pattern.palette) + 1))

    def test_no_empty_palette_entries(self, pattern):
        """Every number in the legend must appear somewhere in the artwork."""
        used = set(np.unique(pattern.label_map).tolist())
        assert all(entry.index in used for entry in pattern.palette)
        assert all(entry.pixel_count > 0 for entry in pattern.palette)

    def test_shares_sum_to_one(self, pattern):
        assert sum(e.share for e in pattern.palette) == pytest.approx(1.0, abs=1e-6)


class TestRegions:
    def test_every_region_has_a_valid_palette_index(self, pattern):
        n_colors = len(pattern.palette)
        assert all(0 <= r.palette_index < n_colors for r in pattern.regions)

    def test_region_ids_match_the_map(self, pattern):
        ids_in_map = set(np.unique(pattern.region_map).tolist()) - {0}
        assert ids_in_map == {r.id for r in pattern.regions}

    def test_label_map_agrees_with_region_palette(self, pattern):
        """The color printed in the legend must match the color of the pixels."""
        for region in pattern.regions[:40]:
            mask = pattern.region_map == region.id
            assert np.all(pattern.label_map[mask] == region.palette_index)

    def test_small_regions_were_merged_away(self, sample_image):
        min_area = 120
        result = generate_pattern(
            sample_image, PatternParams(n_colors=8, max_dimension=200, min_region_area=min_area)
        )
        tiny = [r for r in result.regions if r.area < min_area]
        # A region with no neighbour to merge into can legitimately survive, but
        # that should be rare rather than routine.
        assert len(tiny) <= 1

    def test_label_points_fall_inside_region(self, pattern):
        """The printed number must land in the region it describes.

        This is why label placement uses the pole of inaccessibility: a centroid
        falls outside any concave shape, putting the digit in a neighbour.
        """
        for region in pattern.regions:
            if region.label_point is None:
                continue
            x, y = region.label_point
            assert pattern.region_map[y, x] == region.id, (
                f"region {region.id} label at {(x, y)} landed in "
                f"region {pattern.region_map[y, x]}"
            )

    def test_label_radius_describes_available_room(self, pattern):
        for region in pattern.regions:
            if region.label_point is None:
                continue
            assert region.label_radius > 0
            assert region.label_radius**2 <= region.area

    def test_regions_have_contours(self, pattern):
        assert all(r.contours for r in pattern.regions)
        for region in pattern.regions[:20]:
            for polygon in region.contours:
                assert polygon.ndim == 2 and polygon.shape[1] == 2
                assert polygon.shape[0] >= 3


class TestDeterminism:
    def test_same_input_and_seed_produce_identical_output(self, sample_image):
        params = PatternParams(n_colors=6, max_dimension=200, seed=42)
        a = generate_pattern(sample_image, params)
        b = generate_pattern(sample_image, params)
        assert np.array_equal(a.label_map, b.label_map)
        assert np.array_equal(a.region_map, b.region_map)
        assert [e.rgb.hex for e in a.palette] == [e.rgb.hex for e in b.palette]


class TestValidation:
    def test_rejects_grayscale_input(self):
        with pytest.raises(ValueError):
            generate_pattern(np.zeros((20, 20), dtype=np.uint8))

    def test_rejects_too_few_colors(self, sample_image):
        with pytest.raises(ValueError):
            generate_pattern(sample_image, PatternParams(n_colors=1))


class TestSegmentationUnits:
    def test_majority_filter_removes_an_isolated_speck(self):
        labels = np.zeros((21, 21), dtype=np.int32)
        labels[10, 10] = 1  # a single stray pixel
        assert smooth_label_map(labels, 2, kernel=5)[10, 10] == 0

    def test_majority_filter_keeps_a_substantial_block(self):
        labels = np.zeros((21, 21), dtype=np.int32)
        labels[6:15, 6:15] = 1
        assert smooth_label_map(labels, 2, kernel=5)[10, 10] == 1

    def test_merging_absorbs_a_region_below_the_threshold(self):
        labels = np.zeros((40, 40), dtype=np.int32)
        labels[20:23, 20:23] = 1  # 9 px, well under the minimum
        palette = np.array([[50.0, 0.0, 0.0], [55.0, 2.0, 2.0]])
        result = build_regions(labels, palette, min_area=50)
        assert result.n_regions == 1
        assert np.all(result.label_map == 0)
