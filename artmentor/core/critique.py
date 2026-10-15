"""Feedback on a painting, measured rather than judged.

The original brief asked for AI that analyses a learner's work and says whether
they have improved. Taken literally that is not a solvable problem: there is no
ground truth for "good painting", professionals disagree, and a model trained on
someone else's taste would hand a beginner a confident verdict it has no right
to. Worse, a wrong discouraging critique is actively harmful — it is the exact
opposite of what a tutor is for.

So this measures something real instead: **fidelity to a stated target**. Not
"is this good art" but "where does this differ from what you were copying, and
by how much". Every number here is computed, explainable, and checkable by the
learner against their own eyes. Nothing is inferred from taste.

That reframing also makes progress trackable. Fidelity is comparable across
attempts in a way that aesthetic judgement never is, so "your values were 12%
off last week and 6% off today" is a real statement about improvement.

What this deliberately does not do
----------------------------------
No composition advice, no "the mood would be stronger if", no score out of ten.
A later layer may put a language model in front of these numbers to phrase them
warmly, but the model must never originate a measurement — see
`interfaces.FeedbackProvider`.

Alignment is assumed, not solved. The attempt is scaled to the reference's
dimensions and compared directly, which holds for a photo taken square-on and
cropped to the artwork. Perspective correction via feature matching would lift
that restriction and is a sizeable piece of work in its own right.
"""

from __future__ import annotations

import cv2
import numpy as np

from .color.space import rgb_to_lab_array
from .interfaces import Critique, FeedbackProvider, Measurement
from .types import Pattern

__all__ = ["DeterministicFeedback", "critique_attempt"]

# Comparisons run at this resolution. Brushwork is not the subject here —
# colour, value and proportion are — and downsampling both images to a common
# small size removes texture noise that would otherwise swamp those signals.
_COMPARE_SIZE = 160


def _prepare(rgb: np.ndarray, size: int) -> np.ndarray:
    """Scale to a square working size and convert to Lab."""
    if rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError(f"expected an (H, W, 3) image, got {rgb.shape}")
    resized = cv2.resize(rgb, (size, size), interpolation=cv2.INTER_AREA)
    return rgb_to_lab_array(resized)


def _chroma(lab: np.ndarray) -> np.ndarray:
    return np.sqrt(lab[..., 1] ** 2 + lab[..., 2] ** 2)


class DeterministicFeedback(FeedbackProvider):
    """Fidelity metrics computed from the two images. No model, no opinion."""

    def __init__(self, *, size: int = _COMPARE_SIZE) -> None:
        if size < 32:
            raise ValueError("comparison size must be at least 32px")
        self.size = size

    def critique(
        self,
        reference_rgb: np.ndarray,
        attempt_rgb: np.ndarray,
        pattern: Pattern | None = None,
    ) -> Critique:
        ref = _prepare(reference_rgb, self.size)
        att = _prepare(attempt_rgb, self.size)

        measurements = (
            self._colour_accuracy(ref, att),
            self._value_structure(ref, att),
            self._lightness_bias(ref, att),
            self._chroma_bias(ref, att),
        )
        headline, next_step = self._advise(measurements)
        return Critique(
            measurements=measurements, message=headline, suggested_next_step=next_step
        )

    # -- individual measures ------------------------------------------------
    def _colour_accuracy(self, ref: np.ndarray, att: np.ndarray) -> Measurement:
        """Average perceptual distance, pixel for pixel.

        Euclidean in Lab rather than CIEDE2000: this runs over 25,600 pixels and
        is reporting a bulk average, where the two correlate closely. CIEDE2000
        is reserved for the single-colour judgements a learner acts on directly.
        """
        distance = float(np.mean(np.linalg.norm(ref - att, axis=-1)))
        return Measurement(
            key="colour_accuracy",
            value=round(distance, 2),
            unit="dE",
            explanation=f"Your colours sit about {distance:.0f} units from the reference.",
        )

    def _value_structure(self, ref: np.ndarray, att: np.ndarray) -> Measurement:
        """How well the light-to-dark relationships hold up.

        The measure that matters most. A painting with the wrong hues but the
        right values reads correctly; the reverse does not, which is why tutors
        push value studies before colour. Correlation rather than difference,
        because it rewards getting the *relationships* right even if the whole
        painting is lighter or darker than the reference.
        """
        a, b = ref[..., 0].ravel(), att[..., 0].ravel()
        if a.std() < 1e-6 or b.std() < 1e-6:
            correlation = 0.0  # a flat image has no tonal structure to match
        else:
            correlation = float(np.corrcoef(a, b)[0, 1])
        percent = max(0.0, correlation) * 100
        return Measurement(
            key="value_structure",
            value=round(correlation, 3),
            unit="correlation",
            explanation=f"Your light-and-dark pattern matches the reference about {percent:.0f}%.",
        )

    def _lightness_bias(self, ref: np.ndarray, att: np.ndarray) -> Measurement:
        """Whether the whole painting runs light or dark. Signed on purpose."""
        delta = float(att[..., 0].mean() - ref[..., 0].mean())
        direction = "lighter" if delta > 0 else "darker"
        return Measurement(
            key="lightness_bias",
            value=round(delta, 2),
            unit="L*",
            explanation=f"Overall it is {abs(delta):.0f} units {direction} than the reference.",
        )

    def _chroma_bias(self, ref: np.ndarray, att: np.ndarray) -> Measurement:
        """Too grey or too vivid overall — the commonest beginner's mix error."""
        delta = float(_chroma(att).mean() - _chroma(ref).mean())
        direction = "more saturated" if delta > 0 else "more muted"
        return Measurement(
            key="chroma_bias",
            value=round(delta, 2),
            unit="C*",
            explanation=f"Your colours are {abs(delta):.0f} units {direction} overall.",
        )

    # -- turning numbers into one sentence ----------------------------------
    def _advise(self, measurements: tuple[Measurement, ...]) -> tuple[str, str | None]:
        """Pick the single most useful thing to say.

        One instruction, not four. A list of everything wrong is how a learner
        is taught that they are bad at this; the next brushstroke can only act
        on one correction anyway. Ordered by what a tutor would address first —
        values before colour, because values carry the picture.
        """
        by_key = {m.key: m.value for m in measurements}

        if by_key["value_structure"] < 0.55:
            return (
                "Your values need work before the colour does.",
                (
                    "Squint at both images until the detail disappears. The pattern "
                    "of light and dark shapes should match even when you cannot see "
                    "what they are."
                ),
            )
        if abs(by_key["lightness_bias"]) > 12:
            if by_key["lightness_bias"] > 0:
                return (
                    "The whole painting is running light.",
                    (
                        "Push your darks further. Beginners almost always stop "
                        "short of how dark the darkest areas really are."
                    ),
                )
            return (
                "The whole painting is running dark.",
                (
                    "Lift your lightest areas. Reserve your palest mix for the very "
                    "brightest spot and work down from there."
                ),
            )
        if by_key["chroma_bias"] < -9:
            return (
                "Your colours are coming out muddier than the reference.",
                (
                    "Mud usually means too many pigments in one mix. Try hitting "
                    "each colour with two paints instead of three."
                ),
            )
        if by_key["chroma_bias"] > 9:
            return (
                "Your colours are stronger than the reference.",
                (
                    "Knock the intensity back with a touch of the opposite colour "
                    "rather than with grey, which flattens it."
                ),
            )
        if by_key["colour_accuracy"] > 18:
            return (
                "The structure is there; the colours are drifting.",
                (
                    "Work through the mixing guide one number at a time and check "
                    "each mix against its target before it goes on the painting."
                ),
            )
        return (
            "This is tracking the reference closely.",
            "Try the same subject at a higher colour count, or without the pattern.",
        )


def critique_attempt(
    reference_rgb: np.ndarray,
    attempt_rgb: np.ndarray,
    pattern: Pattern | None = None,
) -> Critique:
    """Convenience wrapper around the default provider."""
    return DeterministicFeedback().critique(reference_rgb, attempt_rgb, pattern)