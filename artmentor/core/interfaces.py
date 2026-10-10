"""Abstract interfaces for every swappable capability.

This is the file that makes the Tier A -> Tier D ladder survivable. Each
capability below will be reimplemented at least once as the product matures
(deterministic -> pretrained model -> hosted VLM -> fine-tuned). Code outside
`core` must depend only on these interfaces, never on a concrete class, so a
swap stays confined to one module behind a feature flag.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np

from .types import LabColor, Pattern


# --------------------------------------------------------------------------
# Segmentation
# --------------------------------------------------------------------------
class Segmenter(ABC):
    """Image -> integer region map.

    Tier A: color-quantization + connected components (shipping now).
    Tier B: a pretrained model such as SAM for subject-aware regions.
    """

    @abstractmethod
    def segment(self, image_rgb: np.ndarray) -> np.ndarray:
        """Return an int32 array, same HxW as the image, of region ids."""


# --------------------------------------------------------------------------
# Pigment mixing
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class Paint:
    """A physical tube of paint the learner actually owns."""

    id: str
    name: str
    brand: str
    lab: LabColor
    opacity: float = 1.0          # 0 transparent .. 1 opaque
    tinting_strength: float = 1.0  # how strongly it dominates a mix


@dataclass(frozen=True)
class MixComponent:
    paint: Paint
    parts: float


@dataclass(frozen=True)
class MixRecipe:
    """A human-executable recipe: '2 parts ultramarine, 1 part titanium white'."""

    components: tuple[MixComponent, ...]
    predicted_lab: LabColor
    delta_e: float   # CIEDE2000 between predicted mix and the requested target

    def describe(self) -> str:
        return " + ".join(
            f"{c.parts:g} part{'s' if c.parts != 1 else ''} {c.paint.name}"
            for c in self.components
        )


class PigmentMixer(ABC):
    """Solve the inverse mixing problem: which paints, in what ratio, hit a target?

    Note the direction. Forward mixing ("what do these make?") is the easy half.
    The product needs the inverse, constrained to ratios a human can actually
    measure out by hand.
    """

    @abstractmethod
    def solve(
        self,
        target: LabColor,
        palette: list[Paint],
        max_components: int = 3,
    ) -> MixRecipe:
        """Best achievable recipe for `target` from `palette`."""


# --------------------------------------------------------------------------
# Feedback on a learner's work
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class Measurement:
    """One objective, explainable number about an attempt."""

    key: str          # e.g. "color_accuracy"
    value: float
    unit: str         # e.g. "dE00"
    explanation: str  # plain language, shown to the user


@dataclass(frozen=True)
class Critique:
    """Feedback on an attempt.

    `measurements` are computed deterministically and are the source of truth.
    `message` is presentation only. A language model may phrase the message, but
    must never invent the numbers.
    """

    measurements: tuple[Measurement, ...]
    message: str
    suggested_next_step: str | None = None


class FeedbackProvider(ABC):
    """Reference + attempt -> structured critique.

    Tier A: deterministic fidelity metrics (color accuracy, value structure,
    proportion error). Tier C: a VLM phrasing those same metrics warmly.
    """

    @abstractmethod
    def critique(
        self,
        reference_rgb: np.ndarray,
        attempt_rgb: np.ndarray,
        pattern: Pattern | None = None,
    ) -> Critique:
        ...
