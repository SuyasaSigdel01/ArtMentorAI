"""Domain types shared across the core.

These are deliberately plain dataclasses with no I/O and no library objects in
their public surface, so they can cross the core/API boundary without dragging
OpenCV or scikit-image along with them.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass(frozen=True)
class RGBColor:
    """An 8-bit sRGB color."""

    r: int
    g: int
    b: int

    def __post_init__(self) -> None:
        for name, v in (("r", self.r), ("g", self.g), ("b", self.b)):
            if not 0 <= v <= 255:
                raise ValueError(f"{name}={v} outside 0..255")

    @property
    def hex(self) -> str:
        return f"#{self.r:02x}{self.g:02x}{self.b:02x}"

    @classmethod
    def from_hex(cls, value: str) -> RGBColor:
        s = value.lstrip("#")
        if len(s) == 3:
            s = "".join(c * 2 for c in s)
        if len(s) != 6:
            raise ValueError(f"not a hex color: {value!r}")
        return cls(int(s[0:2], 16), int(s[2:4], 16), int(s[4:6], 16))

    def as_array(self) -> np.ndarray:
        return np.array([self.r, self.g, self.b], dtype=np.uint8)


@dataclass(frozen=True)
class LabColor:
    """A CIELAB color (D65). This is the space all color comparison happens in."""

    L: float
    a: float
    b: float

    def as_array(self) -> np.ndarray:
        return np.array([self.L, self.a, self.b], dtype=np.float64)

    @classmethod
    def from_array(cls, arr) -> LabColor:
        a = np.asarray(arr, dtype=np.float64).reshape(3)
        return cls(float(a[0]), float(a[1]), float(a[2]))


@dataclass(frozen=True)
class PaletteEntry:
    """One color in a generated pattern's palette."""

    index: int          # 0-based internal index
    number: int         # 1-based, the digit actually printed in the pattern
    rgb: RGBColor
    lab: LabColor
    pixel_count: int
    share: float        # fraction of the image, 0..1

    @property
    def label(self) -> str:
        return str(self.number)


@dataclass
class Region:
    """A single contiguous paintable area."""

    id: int
    palette_index: int
    area: int
    centroid: tuple[float, float]           # (x, y)
    label_point: tuple[int, int] | None     # pole of inaccessibility, (x, y)
    label_radius: float                     # distance to nearest edge; drives font size
    contours: list[np.ndarray] = field(default_factory=list)  # (N,2) int arrays, x/y

    @property
    def is_labelable(self) -> bool:
        """Too small to carry a legible number?"""
        return self.label_point is not None and self.label_radius >= 3.0


@dataclass
class Pattern:
    """The full output of the pattern pipeline.

    `pipeline_version` and `params` are stored so a pattern can be explained,
    reproduced, or regenerated through a newer pipeline and compared. Do not drop
    these — without them you cannot answer "why does this look different now?".
    """

    width: int
    height: int
    palette: list[PaletteEntry]
    regions: list[Region]
    pipeline_version: str
    params: dict

    # Per-pixel maps. Kept on the Pattern because the critique feature later needs
    # to compare a learner's photographed attempt region by region against the
    # intended colors, and recomputing them would risk drift from what was printed.
    label_map: np.ndarray | None = None   # (H, W) palette index per pixel
    region_map: np.ndarray | None = None  # (H, W) region id per pixel

    @property
    def labelable_regions(self) -> list[Region]:
        return [r for r in self.regions if r.is_labelable]

    def summary(self) -> dict:
        return {
            "size": [self.width, self.height],
            "colors": len(self.palette),
            "regions": len(self.regions),
            "labelable_regions": len(self.labelable_regions),
            "pipeline_version": self.pipeline_version,
            "params": self.params,
        }
