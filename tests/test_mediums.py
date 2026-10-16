"""Tests for medium and skill-level guidance.

The risk with this module is that it becomes decorative — six blocks of advice
that read well and change nothing. So almost every test here asserts on a
number the pipeline consumes, not on wording. The two that do look at text
check a physical claim: that a watercolour recipe never tells a learner to add
white, and that a pencil "mix" is not phrased as a mix at all.

`test_neither_limit_can_raise_the_other` is the one that matters most: a
confident painter in watercolour must still get watercolour's two-pigment
ceiling, and a beginner in oil must still get the beginner's.
"""

from __future__ import annotations

from typing import ClassVar

import pytest

from artmentor.core.mediums import (
    LEVELS,
    MEDIUMS,
    describe_mix,
    describe_value,
    get_level,
    get_medium,
    guidance_for,
    next_level,
    should_advance,
)
from artmentor.core.paints import catalog_by_id


class TestCatalogue:
    def test_every_medium_in_the_ui_picker_exists(self):
        """The studio form offers these; an unknown one would 400 on save."""
        offered = {"acrylic", "oil", "watercolour", "gouache", "pencil", "crayon"}
        assert {m.id for m in MEDIUMS} == offered

    def test_ids_are_unique(self):
        assert len({m.id for m in MEDIUMS}) == len(MEDIUMS)
        assert len({level.id for level in LEVELS}) == len(LEVELS)

    def test_every_medium_says_how_to_lighten_a_colour(self):
        """There is no fourth way, and a missing one would break describe_mix."""
        for m in MEDIUMS:
            assert m.lightens_with in {"white", "water", "pressure"}

    def test_every_medium_has_a_note_a_learner_can_act_on(self):
        for m in MEDIUMS:
            assert m.note.endswith(".")
            assert len(m.note) > 40

    def test_unknown_ids_raise_with_a_list_of_the_real_ones(self):
        with pytest.raises(KeyError, match="acrylic"):
            get_medium("tempera")
        with pytest.raises(KeyError, match="beginner"):
            get_level("expert")

    def test_the_drying_description_follows_the_number(self):
        assert get_medium("watercolour").dries == "lighter"
        assert get_medium("acrylic").dries == "darker"
        assert get_medium("pencil").dries == "much the same"


class TestPhysics:
    """The claims here are about paint, and they are what the maths obeys."""

    def test_only_watercolour_and_dry_media_leave_white_out(self):
        assert get_medium("acrylic").mixes_white
        assert get_medium("oil").mixes_white
        assert get_medium("gouache").mixes_white
        assert not get_medium("watercolour").mixes_white
        assert not get_medium("pencil").mixes_white

    def test_watercolour_excludes_white_from_the_palette(self):
        """The paper is the white. A recipe calling for it cannot be followed."""
        g = guidance_for("watercolour", "confident")
        assert "ti-white" in g.excluded_paint_ids
        assert "ti-white" in catalog_by_id(), "the exclusion must name a real paint"

    def test_acrylic_keeps_every_paint(self):
        assert guidance_for("acrylic", "confident").excluded_paint_ids == ()

    def test_wet_media_need_coarser_regions_than_a_pencil(self):
        """A wet brush cannot hold the edge a pencil can."""
        pencil = guidance_for("pencil").min_region_area
        acrylic = guidance_for("acrylic").min_region_area
        watercolour = guidance_for("watercolour").min_region_area
        assert pencil < acrylic < watercolour

    def test_a_pencil_mixes_nothing(self):
        assert get_medium("pencil").max_components == 1

    def test_watercolour_allows_fewer_pigments_than_oil(self):
        """Three pigments stay recoverable in oil and go to mud on wet paper."""
        assert get_medium("watercolour").max_components < get_medium("oil").max_components


class TestGuidance:
    def test_a_beginner_is_given_fewer_colours_than_a_confident_painter(self):
        assert (
            guidance_for("acrylic", "beginner").colors
            < guidance_for("acrylic", "improving").colors
            < guidance_for("acrylic", "confident").colors
        )

    def test_the_colour_count_stays_inside_what_the_api_accepts(self):
        """/patterns rejects anything outside 2..24, so nothing here may exceed it."""
        for m in MEDIUMS:
            for level in LEVELS:
                assert 4 <= guidance_for(m.id, level.id).colors <= 24

    def test_neither_limit_can_raise_the_other(self):
        """Medium and level each cap the mix; the stricter one always wins."""
        # Watercolour allows 2; the confident level would allow 3.
        assert guidance_for("watercolour", "confident").max_components == 2
        # Oil allows 3; the beginner level would allow 2.
        assert guidance_for("oil", "beginner").max_components == 2
        assert guidance_for("oil", "confident").max_components == 3

    def test_defaults_are_the_middle_of_the_road(self):
        g = guidance_for()
        assert g.medium.id == "acrylic"
        assert g.level.id == "improving"

    def test_the_notes_explain_a_missing_white(self):
        notes = " ".join(guidance_for("watercolour").notes).lower()
        assert "water" in notes and "paper" in notes

    def test_a_medium_that_cannot_be_reworked_says_so(self):
        assert any("light to dark" in n for n in guidance_for("watercolour").notes)
        assert not any("light to dark" in n for n in guidance_for("acrylic").notes)

    def test_the_top_level_is_not_told_what_comes_next(self):
        assert not any("When this feels easy" in n for n in guidance_for("oil", "confident").notes)
        assert any("When this feels easy" in n for n in guidance_for("oil", "beginner").notes)

    def test_the_target_comes_from_the_level_not_the_medium(self):
        assert (
            guidance_for("oil", "beginner").target_delta_e
            == guidance_for("watercolour", "beginner").target_delta_e
        )


class TestWording:
    MIX: ClassVar = [("Ultramarine", 2.0), ("Titanium White", 1.0)]

    def test_acrylic_counts_out_parts(self):
        text = describe_mix(self.MIX, get_medium("acrylic"))
        assert text == "2 parts Ultramarine + 1 part Titanium White"

    def test_watercolour_says_to_thin_it(self):
        text = describe_mix([("Ultramarine", 1.0)], get_medium("watercolour"))
        assert "water" in text
        assert "part" in text

    def test_a_pencil_is_never_told_to_mix(self):
        """Layering is optical here; 'two parts' would be a physical impossibility."""
        text = describe_mix([("Burnt Sienna", 1.0)], get_medium("pencil"))
        assert "part" not in text.lower()

    def test_layered_media_say_which_goes_on_top(self):
        text = describe_mix([("Crimson", 2.0), ("Ochre", 1.0)], get_medium("crayon"))
        assert "layered over" in text
        assert text.index("Crimson") < text.index("Ochre")

    def test_an_even_layer_is_described_as_even(self):
        text = describe_mix([("Crimson", 1.0), ("Ochre", 1.0)], get_medium("crayon"))
        assert "evenly" in text

    def test_singular_and_plural_parts(self):
        text = describe_mix([("A", 1.0), ("B", 3.0)], get_medium("oil"))
        assert "1 part A" in text and "3 parts B" in text

    def test_zero_part_components_are_dropped(self):
        text = describe_mix([("A", 2.0), ("B", 0.0)], get_medium("acrylic"))
        assert "B" not in text

    @pytest.mark.parametrize("bad", [[], [("A", 0.0)]])
    def test_an_empty_mix_is_an_error_not_an_empty_string(self, bad):
        with pytest.raises(ValueError):
            describe_mix(bad, get_medium("acrylic"))


class TestAdvancement:
    def test_a_run_of_on_target_attempts_suggests_moving_up(self):
        level = get_level("beginner")  # aiming for 16
        assert should_advance(level, [20.0, 12.0, 11.0, 10.0]) is True

    def test_one_good_attempt_is_not_enough(self):
        """A single lucky photograph under good light must not promote anybody."""
        level = get_level("beginner")
        assert should_advance(level, [4.0]) is False
        assert should_advance(level, [4.0, 3.0]) is False

    def test_one_bad_session_does_not_hold_someone_back(self):
        """Median, not mean: the outlier is outvoted by the two good attempts."""
        level = get_level("beginner")
        assert should_advance(level, [9.0, 40.0, 10.0]) is True

    def test_missing_the_target_keeps_them_where_they_are(self):
        level = get_level("improving")  # aiming for 11
        assert should_advance(level, [19.0, 20.0, 18.0]) is False

    def test_only_the_most_recent_attempts_count(self):
        """Last year's work should not decide today's suggestion."""
        level = get_level("beginner")
        old_and_bad = [60.0, 55.0, 58.0, 9.0, 8.0, 10.0]
        assert should_advance(level, old_and_bad) is True

    def test_the_top_of_the_ladder_has_nowhere_to_go(self):
        assert should_advance(get_level("confident"), [1.0, 1.0, 1.0, 1.0]) is False
        assert next_level(get_level("confident")) is None

    def test_the_ladder_goes_up_in_order(self):
        assert next_level(get_level("beginner")).id == "improving"
        assert next_level(get_level("improving")).id == "confident"

    def test_each_rung_asks_for_more_than_the_last(self):
        targets = [level.target_delta_e for level in LEVELS]
        assert targets == sorted(targets, reverse=True), "targets must tighten"

    def test_nonsense_values_are_ignored_rather_than_counted(self):
        level = get_level("beginner")
        assert should_advance(level, [-1.0, -1.0, 9.0]) is False


class TestMonochrome:
    def test_only_graphite_is_monochrome(self):
        assert get_medium("pencil").monochrome
        assert not get_medium("crayon").monochrome
        assert not get_medium("acrylic").monochrome

    def test_white_paper_is_left_alone(self):
        assert "bare paper" in describe_value(100.0, get_medium("pencil"))

    def test_tones_get_darker_as_lightness_falls(self):
        pencil = get_medium("pencil")
        steps = [
            int(describe_value(L, pencil).split("about ")[1].split(" ")[0])
            for L in (88.0, 70.0, 55.0, 40.0)
        ]
        assert steps == sorted(steps), steps

    def test_the_scale_stops_at_what_graphite_can_reach(self):
        """Chasing a black graphite cannot make just tears the paper."""
        pencil = get_medium("pencil")
        assert "10 out of 10" in describe_value(pencil.value_floor, pencil)
        assert "10 out of 10" in describe_value(0.0, pencil)
        assert "11 out of" not in describe_value(0.0, pencil)

    def test_out_of_range_lightness_does_not_escape_the_scale(self):
        for L in (-40.0, 180.0):
            text = describe_value(L, get_medium("pencil"))
            assert "out of 10" in text or "bare paper" in text

    def test_the_phrase_reads_as_english(self):
        for L in range(0, 101, 5):
            text = describe_value(float(L), get_medium("pencil"))
            assert "A your" not in text
            assert text[0].isupper()