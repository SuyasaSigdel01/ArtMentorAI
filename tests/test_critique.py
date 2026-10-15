"""Tests for painting feedback.

Each test paints a deliberate, specific mistake and checks the critique names
that mistake rather than a different one. Synthetic images make the ground
truth exact: when an attempt is built by multiplying lightness by 1.35, "the
whole painting is running light" is verifiably the right answer, in a way no
real photograph could establish.

`test_values_outrank_colour` is the one that encodes the teaching order.
"""

from __future__ import annotations

import numpy as np
import pytest

from artmentor.core.critique import DeterministicFeedback, critique_attempt
from artmentor.core.interfaces import FeedbackProvider


def scene(
    *,
    light: float = 1.0,
    chroma: float = 1.0,
    invert_values: bool = False,
    flat: bool = False,
    seed: int = 0,
) -> np.ndarray:
    """A simple landscape, optionally spoiled in one specific way."""
    size = 240
    img = np.zeros((size, size, 3), dtype=np.float64)
    if flat:
        img[:, :] = (128, 128, 128)
    else:
        img[:80] = (120, 170, 220)      # sky
        img[80:160] = (80, 140, 70)     # field
        img[160:] = (200, 90, 60)       # earth
        img[100:180, 80:160] = (240, 220, 60)  # a bright subject

    if invert_values:
        img = 255 - img

    grey = img.mean(axis=2, keepdims=True)
    img = grey + (img - grey) * chroma
    img = img * light

    rng = np.random.default_rng(seed)
    return np.clip(img + rng.normal(0, 2.0, img.shape), 0, 255).astype(np.uint8)


def values_of(critique) -> dict[str, float]:
    return {m.key: m.value for m in critique.measurements}


REFERENCE = scene()


class TestMeasurements:
    def test_reports_all_four_measures(self):
        c = critique_attempt(REFERENCE, scene(seed=1))
        assert set(values_of(c)) == {
            "colour_accuracy",
            "value_structure",
            "lightness_bias",
            "chroma_bias",
        }

    def test_every_measure_is_explainable(self):
        """A number a learner cannot interpret is not feedback."""
        for m in critique_attempt(REFERENCE, scene(seed=1)).measurements:
            assert m.unit
            assert m.explanation
            assert m.explanation[0].isupper() and m.explanation.endswith(".")

    def test_a_faithful_copy_scores_well(self):
        v = values_of(critique_attempt(REFERENCE, scene(seed=1)))
        assert v["value_structure"] > 0.95
        assert v["colour_accuracy"] < 5
        assert abs(v["lightness_bias"]) < 3

    def test_identical_images_are_near_perfect(self):
        v = values_of(critique_attempt(REFERENCE, REFERENCE))
        assert v["value_structure"] == pytest.approx(1.0, abs=1e-6)
        assert v["colour_accuracy"] < 0.01

    def test_lightness_bias_is_signed(self):
        """Direction matters: 'too light' and 'too dark' need opposite advice."""
        assert values_of(critique_attempt(REFERENCE, scene(light=1.35)))["lightness_bias"] > 10
        assert values_of(critique_attempt(REFERENCE, scene(light=0.62)))["lightness_bias"] < -10

    def test_chroma_bias_is_signed(self):
        assert values_of(critique_attempt(REFERENCE, scene(chroma=0.35)))["chroma_bias"] < -9
        assert values_of(critique_attempt(REFERENCE, scene(chroma=1.9)))["chroma_bias"] > 9

    def test_value_correlation_survives_a_uniform_shift(self):
        """A painting that is uniformly darker still has the right relationships.

        Correlation rather than difference, so a learner who nailed the tonal
        structure but mixed everything a shade dark is told about the shade,
        not told their values are wrong.
        """
        v = values_of(critique_attempt(REFERENCE, scene(light=0.75)))
        assert v["value_structure"] > 0.9
        assert v["lightness_bias"] < -8

    def test_inverted_values_score_negatively(self):
        assert values_of(critique_attempt(REFERENCE, scene(invert_values=True)))[
            "value_structure"
        ] < 0

    def test_a_near_blank_canvas_scores_no_structure(self):
        """A flat attempt must not score as matching the reference.

        Note the loose bound: a real near-blank canvas carries sensor noise, so
        its correlation lands near zero rather than exactly zero. Asserting
        equality here would test the divide-by-zero guard instead of the
        property that matters.
        """
        v = values_of(critique_attempt(REFERENCE, scene(flat=True)))
        assert abs(v["value_structure"]) < 0.15

    def test_a_perfectly_flat_image_does_not_divide_by_zero(self):
        """The guard path: zero variance has no correlation to compute."""
        blank = np.full((240, 240, 3), 128, dtype=np.uint8)
        assert values_of(critique_attempt(REFERENCE, blank))["value_structure"] == 0.0


class TestAdvice:
    @pytest.mark.parametrize(
        "attempt,expected",
        [
            (scene(light=1.35), "running light"),
            (scene(light=0.62), "running dark"),
            (scene(chroma=0.35), "muddier"),
            (scene(chroma=1.9), "stronger"),
            (scene(invert_values=True), "values"),
            (scene(seed=7), "closely"),
        ],
    )
    def test_names_the_actual_mistake(self, attempt, expected):
        assert expected in critique_attempt(REFERENCE, attempt).message.lower()

    def test_values_outrank_colour(self):
        """Tonal structure is addressed before hue, the way tutors teach it.

        The inverted-values attempt has the largest colour error of any case
        here, so a critique sorted by magnitude would talk about colour. It
        must talk about values instead.
        """
        c = critique_attempt(REFERENCE, scene(invert_values=True))
        v = values_of(c)
        assert v["colour_accuracy"] > 40, "this attempt is also wildly off in colour"
        assert "values" in c.message.lower()

    def test_gives_exactly_one_instruction(self):
        """Four corrections at once teaches a learner they are bad at this."""
        for attempt in (scene(light=1.35), scene(chroma=0.35), scene(invert_values=True)):
            step = critique_attempt(REFERENCE, attempt).suggested_next_step
            assert step and len(step.split(".")) <= 4

    def test_the_next_step_is_actionable(self):
        """Advice must describe something to do, not restate the measurement."""
        step = critique_attempt(REFERENCE, scene(light=1.35)).suggested_next_step
        assert any(verb in step.lower() for verb in ("push", "lift", "try", "work", "squint"))

    def test_a_good_attempt_is_told_to_move_on(self):
        c = critique_attempt(REFERENCE, scene(seed=9))
        assert "closely" in c.message.lower()
        assert c.suggested_next_step


class TestContract:
    def test_it_is_a_feedback_provider(self):
        """The interface defined on day one, finally implemented."""
        assert isinstance(DeterministicFeedback(), FeedbackProvider)

    def test_it_is_deterministic(self):
        attempt = scene(light=1.2, seed=3)
        assert values_of(critique_attempt(REFERENCE, attempt)) == values_of(
            critique_attempt(REFERENCE, attempt)
        )

    def test_different_sized_images_still_compare(self):
        """A photo of a painting is never the reference's exact dimensions."""
        small = scene(seed=2)[::3, ::2]
        assert critique_attempt(REFERENCE, small).measurements

    def test_rejects_a_non_rgb_image(self):
        with pytest.raises(ValueError):
            critique_attempt(REFERENCE, np.zeros((40, 40), dtype=np.uint8))

    def test_rejects_a_useless_comparison_size(self):
        with pytest.raises(ValueError):
            DeterministicFeedback(size=8)

    def test_makes_no_claim_about_artistic_quality(self):
        """Guards the project's stance: this measures fidelity, not merit.

        If someone later adds "beautiful" or a score out of ten, this fails and
        makes them read the module docstring explaining why that is a different
        and much harder problem.
        """
        banned = ("beautiful", "talented", "good art", "score", "/10", "grade")
        for attempt in (scene(seed=4), scene(light=1.4), scene(invert_values=True)):
            c = critique_attempt(REFERENCE, attempt)
            text = (c.message + " " + (c.suggested_next_step or "")).lower()
            assert not any(word in text for word in banned)