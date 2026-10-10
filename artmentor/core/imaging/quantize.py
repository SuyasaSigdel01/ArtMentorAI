"""Color quantization: reduce an image to the N paints a learner will actually use.

Two decisions here matter more than the clustering algorithm itself:

1. Clustering happens in CIELAB, not RGB. Clusters in RGB are grouped by numeric
   proximity, which does not match what the eye groups, and produces palettes
   that look arbitrary.
2. The image is edge-preserving-smoothed first. Photographic noise and JPEG
   artifacts otherwise fragment flat areas into thousands of unpaintable specks,
   and no amount of downstream region merging fully recovers from that.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
from sklearn.cluster import KMeans

from ..color.space import lab_to_rgb_array, rgb_to_lab_array

__all__ = ["Quantization", "presmooth", "quantize"]


@dataclass
class Quantization:
    """Result of reducing an image to a fixed palette."""

    labels: np.ndarray       # (H, W) int32, index into the palette
    palette_lab: np.ndarray  # (K, 3) float64
    palette_rgb: np.ndarray  # (K, 3) uint8
    counts: np.ndarray       # (K,) int64, pixels per palette entry

    @property
    def n_colors(self) -> int:
        return int(self.palette_lab.shape[0])

    def to_rgb_image(self) -> np.ndarray:
        """Render the quantized image back to RGB, for previews."""
        return self.palette_rgb[self.labels]


def presmooth(
    rgb: np.ndarray,
    *,
    diameter: int = 9,
    sigma_color: float = 35.0,
    sigma_space: float = 9.0,
) -> np.ndarray:
    """Edge-preserving smoothing.

    A bilateral filter is the right tool: it flattens gradients and noise inside
    an area while leaving the boundaries between areas sharp, which is exactly
    the structure a paint-by-numbers pattern needs to find.
    """
    if diameter <= 0:
        return rgb
    return cv2.bilateralFilter(rgb, diameter, sigma_color, sigma_space)


def quantize(
    rgb: np.ndarray,
    n_colors: int,
    *,
    sample_size: int = 40_000,
    seed: int = 0,
) -> Quantization:
    """Cluster an image into `n_colors` perceptually-chosen colors.

    K-means is fitted on a random pixel sample for speed, then every pixel is
    assigned. On a 1000px image, sampling 40k pixels gives centroids
    indistinguishable from a full fit at a fraction of the cost.

    Palette entries are ordered by frequency, descending, so number 1 is always
    the color covering the most area. That ordering is a small kindness to the
    painter: it lets them block in the largest areas first, which is how people
    are taught to work anyway.
    """
    if rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError(f"expected an (H, W, 3) image, got {rgb.shape}")
    if n_colors < 2:
        raise ValueError("n_colors must be at least 2")

    h, w = rgb.shape[:2]
    lab = rgb_to_lab_array(rgb)              # (H, W, 3) float64
    flat = lab.reshape(-1, 3)

    n_pixels = flat.shape[0]
    # Never ask for more clusters than there are distinct pixels to cluster.
    k = int(min(n_colors, n_pixels))

    rng = np.random.default_rng(seed)
    if n_pixels > sample_size:
        idx = rng.choice(n_pixels, size=sample_size, replace=False)
        fit_data = flat[idx]
    else:
        fit_data = flat

    kmeans = KMeans(n_clusters=k, n_init=4, random_state=seed)
    kmeans.fit(fit_data)
    labels = kmeans.predict(flat).astype(np.int32)
    centers = np.asarray(kmeans.cluster_centers_, dtype=np.float64)

    counts = np.bincount(labels, minlength=k).astype(np.int64)

    # Drop any empty cluster, then sort by frequency descending.
    keep = np.flatnonzero(counts > 0)
    order = keep[np.argsort(-counts[keep], kind="stable")]

    remap = np.full(k, -1, dtype=np.int32)
    remap[order] = np.arange(order.size, dtype=np.int32)
    labels = remap[labels]

    palette_lab = centers[order]
    counts = counts[order]
    palette_rgb = lab_to_rgb_array(palette_lab, as_uint8=True)

    return Quantization(
        labels=labels.reshape(h, w),
        palette_lab=palette_lab,
        palette_rgb=palette_rgb,
        counts=counts,
    )
