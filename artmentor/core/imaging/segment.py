"""Turn a quantized image into paintable regions.

The gap between "technically produces a pattern" and "produces a pattern a human
enjoys painting" lives almost entirely in this file. A raw quantized image has
thousands of two-pixel specks along every edge; none of them can be painted, and
printing a number in them is impossible. Two steps fix that:

1. A majority (mode) filter over the label map, which removes isolated specks
   without blurring boundaries the way a numeric filter would.
2. Merging every region below a minimum area into whichever neighbour it shares
   the most border with, tie-broken by color similarity.

Step 2 uses union-find so that a whole cascade of merges resolves correctly in
one pass — merging A into B and B into C must leave A pointing at C, not at a
region that no longer exists.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
from skimage.measure import label as _cc_label

from ..color.difference import delta_e_matrix

__all__ = ["RegionMap", "build_regions", "smooth_label_map"]


@dataclass
class RegionMap:
    """Contiguous paintable areas and the palette entry each one carries."""

    region_map: np.ndarray      # (H, W) int32, ids 1..n_regions
    region_palette: np.ndarray  # (n_regions + 1,) int32, index 0 unused
    label_map: np.ndarray       # (H, W) int32, palette index per pixel

    @property
    def n_regions(self) -> int:
        return int(self.region_palette.shape[0] - 1)


def _majority(labels: np.ndarray, n_colors: int, kernel: int) -> np.ndarray:
    """Per-pixel neighbourhood vote over label indices.

    A median filter is the usual shortcut, but medians of category indices are
    meaningless — index 7 is not "between" 6 and 8. This is a true vote, costing
    one box filter per color, which is cheap at the palette sizes used here.
    """
    if kernel <= 1:
        return labels.astype(np.int32, copy=False)
    if kernel % 2 == 0:
        kernel += 1

    h, w = labels.shape
    best_score = np.zeros((h, w), dtype=np.float32)
    best_label = np.zeros((h, w), dtype=np.int32)

    for k in range(n_colors):
        member = (labels == k).astype(np.float32)
        if not member.any():
            continue
        score = cv2.boxFilter(
            member, -1, (kernel, kernel), normalize=True, borderType=cv2.BORDER_REPLICATE
        )
        update = score > best_score
        best_score[update] = score[update]
        best_label[update] = k

    return best_label


def _edge_strength(rgb: np.ndarray, blur: int = 5) -> np.ndarray:
    """Local structure measure: blurred Sobel magnitude of a grayscale view."""
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    magnitude = cv2.magnitude(gx, gy)
    if blur > 1:
        magnitude = cv2.GaussianBlur(magnitude, (blur | 1, blur | 1), 0)
    return magnitude


def smooth_label_map(
    labels: np.ndarray,
    n_colors: int,
    kernel: int = 5,
    *,
    flat_kernel: int = 15,
    edge_source: np.ndarray | None = None,
    edge_percentile: float = 70.0,
) -> np.ndarray:
    """Despeckle a label map with a neighbourhood size that adapts to content.

    A single global kernel cannot win here. Quantizing a smooth gradient (a sky)
    leaves a boundary that wanders by several pixels, and only a wide vote
    straightens it — but a wide vote also erases small deliberate details like a
    pupil or a highlight, which are exactly the features a painter cares about.

    The two cases are distinguishable before quantization: a sky has almost no
    local gradient, a pupil has a lot. So vote over a wide neighbourhood where the
    source image is flat, and a narrow one where there is real structure.

    Falls back to a single uniform vote when no `edge_source` is supplied.
    """
    narrow = _majority(labels, n_colors, kernel)
    if edge_source is None or flat_kernel <= kernel:
        return narrow

    wide = _majority(labels, n_colors, flat_kernel)

    strength = _edge_strength(edge_source)
    threshold = float(np.percentile(strength, edge_percentile))
    is_flat = strength < threshold

    return np.where(is_flat, wide, narrow).astype(np.int32)


def _adjacency(region_map: np.ndarray, stride: int) -> dict[int, dict[int, int]]:
    """Shared-border length between every pair of 4-connected neighbouring regions."""
    left, right = region_map[:, :-1].ravel(), region_map[:, 1:].ravel()
    top, bottom = region_map[:-1, :].ravel(), region_map[1:, :].ravel()

    a = np.concatenate([left, top])
    b = np.concatenate([right, bottom])
    differing = a != b
    a, b = a[differing], b[differing]

    # Store both directions so each region can look up all of its neighbours.
    keys = np.concatenate(
        [a.astype(np.int64) * stride + b, b.astype(np.int64) * stride + a]
    )
    uniq, counts = np.unique(keys, return_counts=True)

    out: dict[int, dict[int, int]] = {}
    for key, count in zip(uniq.tolist(), counts.tolist()):
        i, j = divmod(key, stride)
        out.setdefault(i, {})[j] = count
    return out


def build_regions(
    labels: np.ndarray,
    palette_lab: np.ndarray,
    *,
    min_area: int = 80,
    max_passes: int = 6,
) -> RegionMap:
    """Build connected regions and merge away anything too small to paint."""
    region_map = _cc_label(labels, connectivity=1, background=-1).astype(np.int32)
    n_initial = int(region_map.max())
    if n_initial == 0:
        raise ValueError("label map produced no regions")

    # Palette index per region. Safe because every pixel of a region shares a label.
    region_palette = np.zeros(n_initial + 1, dtype=np.int32)
    region_palette[region_map.ravel()] = labels.ravel()

    # Perceptual distance between palette entries, for merge tie-breaking.
    color_distance = delta_e_matrix(palette_lab, palette_lab)

    parent = np.arange(n_initial + 1, dtype=np.int32)

    def find(x: int) -> int:
        root = x
        while parent[root] != root:
            root = parent[root]
        while parent[x] != root:  # path compression
            parent[x], x = root, int(parent[x])
        return int(root)

    stride = n_initial + 1

    for _ in range(max_passes):
        roots = np.array([find(i) for i in range(n_initial + 1)], dtype=np.int32)
        current = roots[region_map]
        areas = np.bincount(current.ravel(), minlength=n_initial + 1)

        small = [r for r in np.flatnonzero(areas).tolist() if r != 0 and areas[r] < min_area]
        if not small:
            break

        adjacency = _adjacency(current, stride)
        merged_any = False

        # Smallest first: a speck should be absorbed before it is used as a target.
        for region in sorted(small, key=lambda r: areas[r]):
            root = find(region)
            neighbours = adjacency.get(region)
            if not neighbours:
                continue

            candidates: dict[int, int] = {}
            for neighbour, shared in neighbours.items():
                n_root = find(neighbour)
                if n_root == root:
                    continue
                candidates[n_root] = candidates.get(n_root, 0) + shared
            if not candidates:
                continue

            own_color = region_palette[root]
            target = max(
                candidates.items(),
                key=lambda kv: (kv[1], -color_distance[own_color, region_palette[kv[0]]]),
            )[0]

            parent[root] = target
            merged_any = True

        if not merged_any:
            break

    # Compact ids to 1..N and rebuild the palette lookup.
    roots = np.array([find(i) for i in range(n_initial + 1)], dtype=np.int32)
    current = roots[region_map]

    surviving = np.unique(current)
    surviving = surviving[surviving != 0]

    remap = np.zeros(n_initial + 1, dtype=np.int32)
    remap[surviving] = np.arange(1, surviving.size + 1, dtype=np.int32)

    final_map = remap[current]
    final_palette = np.zeros(surviving.size + 1, dtype=np.int32)
    final_palette[remap[surviving]] = region_palette[surviving]

    return RegionMap(
        region_map=final_map,
        region_palette=final_palette,
        label_map=final_palette[final_map],
    )
