"""Spectral reflectance machinery underpinning pigment mixing.

Paint does not mix like light. Light is additive — red plus green gives yellow.
Paint is subtractive: each pigment *removes* wavelengths, so blue plus yellow
gives green. Averaging RGB models neither, which is why naive digital mixing of
blue and yellow produces grey.

Modelling this properly means leaving RGB entirely and working with reflectance
spectra under Kubelka-Munk theory:

    1. recover a plausible reflectance spectrum for each paint's color
    2. convert reflectance to K/S (absorption over scattering)
    3. mix K/S linearly, weighted by concentration
    4. convert back to reflectance, then to XYZ, then to sRGB

Step 1 is underdetermined: infinitely many spectra render to the same RGB
(metamerism). The standard resolution is to pick the *smoothest* spectrum that
reproduces the color, since real pigment reflectance curves are smooth. That is
what `srgb_to_reflectance` solves for.

LICENSING NOTE. Mixbox is the best-known implementation of this idea and is
CC BY-NC 4.0 — non-commercial only, with `pymixbox` inheriting the restriction.
This module is an independent implementation from published Kubelka-Munk theory,
carrying no such restriction. If its quality ceiling becomes limiting, the MIT
licensed Spectral.js is the alternative to port; Mixbox requires negotiating a
commercial license first.
"""

from __future__ import annotations

from functools import lru_cache

import colour
import numpy as np
from scipy.optimize import minimize

__all__ = [
    "WAVELENGTHS",
    "ks_to_reflectance",
    "reflectance_to_ks",
    "reflectance_to_srgb",
    "reflectance_to_xyz",
    "srgb_to_reflectance",
]

# 380-730nm at 10nm steps: the visible range, at the resolution pigment data is
# normally published in. Finer steps buy nothing here and cost optimizer time.
_SHAPE = colour.SpectralShape(380, 730, 10)
_CMFS = colour.MSDS_CMFS["CIE 1931 2 Degree Standard Observer"].copy().align(_SHAPE)
_ILLUMINANT = colour.SDS_ILLUMINANTS["D65"].copy().align(_SHAPE)

WAVELENGTHS: np.ndarray = np.asarray(_CMFS.wavelengths, dtype=np.float64)

_A = np.asarray(_CMFS.values, dtype=np.float64)        # (N, 3) color matching
_E = np.asarray(_ILLUMINANT.values, dtype=np.float64)  # (N,)   D65

_N = _A.shape[0]
# Normalized so a perfect reflector (R = 1 everywhere) yields Y = 1.
_WEIGHTS = (_E[:, None] * _A) / float((_E * _A[:, 1]).sum())

# Second-difference operator, the smoothness penalty in the inverse solve.
_D2 = np.zeros((_N - 2, _N), dtype=np.float64)
for _i in range(_N - 2):
    _D2[_i, _i], _D2[_i, _i + 1], _D2[_i, _i + 2] = 1.0, -2.0, 1.0

# Reflectance is clamped away from 0, where K/S diverges.
_EPSILON = 1e-6


def reflectance_to_xyz(reflectance: np.ndarray) -> np.ndarray:
    """Integrate a reflectance spectrum against D65 and the CIE observer."""
    return np.asarray(reflectance, dtype=np.float64) @ _WEIGHTS


def reflectance_to_srgb(reflectance: np.ndarray) -> np.ndarray:
    """Reflectance spectrum to sRGB in 0..1, clipped into gamut."""
    return np.clip(colour.XYZ_to_sRGB(reflectance_to_xyz(reflectance)), 0.0, 1.0)


def _solve_reflectance(rgb: tuple[float, float, float], smoothness: float) -> np.ndarray:
    target = colour.sRGB_to_XYZ(np.asarray(rgb, dtype=np.float64))

    def unpack(z: np.ndarray) -> np.ndarray:
        # Logistic transform keeps reflectance strictly inside (0, 1) without
        # needing explicit bound constraints, which L-BFGS handles poorly here.
        return 1.0 / (1.0 + np.exp(-z))

    def objective(z: np.ndarray) -> float:
        reflectance = unpack(z)
        error = reflectance_to_xyz(reflectance) - target
        roughness = float(((_D2 @ reflectance) ** 2).sum())
        return float(error @ error) + smoothness * roughness

    result = minimize(
        objective,
        np.zeros(_N),
        method="L-BFGS-B",
        options={"maxiter": 2000, "ftol": 1e-14, "gtol": 1e-12},
    )
    return np.clip(unpack(result.x), _EPSILON, 1.0)


@lru_cache(maxsize=512)
def _cached_reflectance(key: tuple[int, int, int], smoothness: float) -> tuple[float, ...]:
    rgb = (key[0] / 65535.0, key[1] / 65535.0, key[2] / 65535.0)
    return tuple(_solve_reflectance(rgb, smoothness))


def srgb_to_reflectance(rgb, *, smoothness: float = 4e-4) -> np.ndarray:
    """Smoothest physically plausible reflectance spectrum rendering to `rgb`.

    Args:
        rgb: sRGB in 0..1, or 0..255 if any component exceeds 1.
        smoothness: weight on the roundness penalty. Round-trip accuracy is
            insensitive to it across several orders of magnitude, so it is not a
            useful tuning knob — the quality ceiling lies elsewhere (see below).

    This runs a 36-dimensional optimization, so results are cached by color. A
    palette's spectra are solved once and reused for every subsequent mix.

    Accuracy: round-trips to within ~0.001 per channel. The limitation is not
    precision but realism — a spectrum inferred from three numbers is smoother
    and broader than a real pigment's, which is why mixes involving strongly
    selective pigments drift from what the paint actually does. Measured
    spectra are the fix, not more tuning.
    """
    values = np.asarray(rgb, dtype=np.float64).reshape(3)
    if values.max() > 1.0 + 1e-9:
        values = values / 255.0
    values = np.clip(values, 0.0, 1.0)

    key = tuple(int(round(v * 65535.0)) for v in values)
    return np.asarray(_cached_reflectance(key, smoothness), dtype=np.float64)  # type: ignore[arg-type]


def reflectance_to_ks(reflectance: np.ndarray) -> np.ndarray:
    """Kubelka-Munk remission: K/S = (1 - R)^2 / 2R.

    K is absorption, S is scattering. Unlike reflectance, K/S is additive across
    pigments, which is the entire reason this detour through spectra is worth it.
    """
    r = np.clip(np.asarray(reflectance, dtype=np.float64), _EPSILON, 1.0)
    return (1.0 - r) ** 2 / (2.0 * r)


def ks_to_reflectance(ks: np.ndarray) -> np.ndarray:
    """Invert the K/S relation: R = 1 + K/S - sqrt((K/S)^2 + 2(K/S))."""
    k = np.maximum(np.asarray(ks, dtype=np.float64), 0.0)
    return 1.0 + k - np.sqrt(k * k + 2.0 * k)
