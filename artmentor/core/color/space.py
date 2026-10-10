"""sRGB <-> CIELAB conversions.

Why this module exists at all: it is tempting to compare colors in RGB because
the numbers are right there. Don't. RGB distance does not match human perception,
so k-means in RGB produces palettes that look wrong, and "how far off is this
mix?" in RGB produces advice that is confidently incorrect.

Everything in ArtMentorAI compares colors in CIELAB (D65), which is approximately
perceptually uniform, and measures distance with CIEDE2000 (see `difference.py`).

Conversions delegate to scikit-image rather than being hand-rolled. The sRGB
transfer function has a linear segment near black that is easy to get subtly
wrong, and a subtle error here poisons every downstream feature.
"""

from __future__ import annotations

import numpy as np
from skimage import color as _skcolor

from ..types import LabColor, RGBColor

__all__ = [
    "hex_to_lab",
    "lab_to_hex",
    "lab_to_rgb",
    "lab_to_rgb_array",
    "rgb_to_lab",
    "rgb_to_lab_array",
]


def rgb_to_lab_array(rgb: np.ndarray) -> np.ndarray:
    """Convert an sRGB array to CIELAB.

    Accepts uint8 (0..255) or float (0..1) input of shape (..., 3).
    Returns float64 of the same shape.
    """
    arr = np.asarray(rgb)
    if arr.shape[-1] != 3:
        raise ValueError(f"expected trailing dimension 3, got shape {arr.shape}")

    if arr.dtype == np.uint8:
        arr = arr.astype(np.float64) / 255.0
    else:
        arr = arr.astype(np.float64)
        # Tolerate 0..255 floats, a common accidental input.
        if arr.size and arr.max() > 1.0 + 1e-6:
            arr = arr / 255.0
    arr = np.clip(arr, 0.0, 1.0)

    # skimage needs at least 2 spatial dims; reshape round-trip keeps this general.
    flat = arr.reshape(-1, 1, 3)
    return _skcolor.rgb2lab(flat).reshape(arr.shape)


def lab_to_rgb_array(lab: np.ndarray, *, as_uint8: bool = True) -> np.ndarray:
    """Convert CIELAB back to sRGB, clipped into gamut.

    Out-of-gamut Lab values are clipped, not flagged. If you need to know whether
    a target color is physically reachable, check before converting.
    """
    arr = np.asarray(lab, dtype=np.float64)
    if arr.shape[-1] != 3:
        raise ValueError(f"expected trailing dimension 3, got shape {arr.shape}")

    flat = arr.reshape(-1, 1, 3)
    rgb = _skcolor.lab2rgb(flat).reshape(arr.shape)
    rgb = np.clip(rgb, 0.0, 1.0)
    if as_uint8:
        return np.round(rgb * 255.0).astype(np.uint8)
    return rgb


def rgb_to_lab(color: RGBColor) -> LabColor:
    """Single-color convenience wrapper."""
    return LabColor.from_array(rgb_to_lab_array(color.as_array().reshape(1, 3))[0])


def lab_to_rgb(color: LabColor) -> RGBColor:
    """Single-color convenience wrapper."""
    arr = lab_to_rgb_array(color.as_array().reshape(1, 3))[0]
    return RGBColor(int(arr[0]), int(arr[1]), int(arr[2]))


def hex_to_lab(value: str) -> LabColor:
    return rgb_to_lab(RGBColor.from_hex(value))


def lab_to_hex(color: LabColor) -> str:
    return lab_to_rgb(color).hex
