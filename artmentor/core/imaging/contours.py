"""Region masks -> simplified polygon outlines.

Outlines are extracted with holes preserved (RETR_CCOMP), because a region shaped
like a doughnut must not have its hole filled in when rendered. The SVG writer
relies on this and uses fill-rule="evenodd" so inner contours punch through.

Polygons are simplified with Douglas-Peucker. The epsilon is small on purpose:
too aggressive and curves visibly facet, which looks cheap on a printed pattern.
"""

from __future__ import annotations

import cv2
import numpy as np
from scipy import ndimage

__all__ = ["extract_contours", "smooth_polygon"]


def smooth_polygon(
    points: np.ndarray, *, iterations: int = 2, max_edge: float = 9.0
) -> np.ndarray:
    """Corner-cutting smoothing that spares structural corners.

    Quantizing a smooth gradient leaves long boundaries that zigzag by a few
    pixels. Plain Chaikin smoothing cleans those up beautifully and also rounds
    off every intentional corner in the image, turning a beak into a thumb.

    The two cases are separable by scale, not by angle: noise zigzags are made of
    very short edges, while a real corner joins two long ones. So a vertex is cut
    only when both of its adjacent edges are shorter than `max_edge`; otherwise it
    survives untouched.
    """
    if iterations <= 0 or points.shape[0] < 4:
        return points.astype(np.float64, copy=False)

    current = points.astype(np.float64, copy=True)
    for _ in range(iterations):
        nxt = np.roll(current, -1, axis=0)
        prv = np.roll(current, 1, axis=0)

        edge_in = np.linalg.norm(current - prv, axis=1)
        edge_out = np.linalg.norm(nxt - current, axis=1)
        cuttable = (edge_in < max_edge) & (edge_out < max_edge)

        out: list[np.ndarray] = []
        for i in range(current.shape[0]):
            if cuttable[i]:
                # Standard Chaikin: replace the vertex with two points at 1/4 and
                # 3/4 along its adjacent edges.
                out.append(0.25 * prv[i] + 0.75 * current[i])
                out.append(0.75 * current[i] + 0.25 * nxt[i])
            else:
                out.append(current[i])
        current = np.asarray(out, dtype=np.float64)
        if current.shape[0] < 4:
            break

    return current


def _find_contours_compat(mask: np.ndarray):
    """findContours returns 2 values on OpenCV 4/5 and 3 on OpenCV 3."""
    result = cv2.findContours(mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    return result[0] if len(result) == 2 else result[1]


def extract_contours(
    region_map: np.ndarray,
    n_regions: int,
    *,
    epsilon: float = 0.75,
    smoothing_iterations: int = 2,
    smoothing_max_edge: float = 9.0,
) -> dict[int, list[np.ndarray]]:
    """Return {region_id: [polygon, ...]} with polygons as (N, 2) float arrays of (x, y)."""
    boxes = ndimage.find_objects(region_map)
    out: dict[int, list[np.ndarray]] = {}

    for region_id in range(1, n_regions + 1):
        if region_id - 1 >= len(boxes):
            break
        box = boxes[region_id - 1]
        if box is None:
            continue

        mask = (region_map[box] == region_id).astype(np.uint8)
        padded = np.pad(mask, 1, mode="constant", constant_values=0)

        polygons: list[np.ndarray] = []
        for contour in _find_contours_compat(padded):
            simplified = cv2.approxPolyDP(contour, epsilon, True) if epsilon > 0 else contour
            points = simplified.reshape(-1, 2)
            if points.shape[0] < 3:
                continue  # degenerate: cannot enclose area
            # Undo the pad, then shift from crop coordinates back to image coordinates.
            offset = np.array([int(box[1].start) - 1, int(box[0].start) - 1], dtype=np.float64)
            shifted = points.astype(np.float64) + offset
            polygons.append(
                smooth_polygon(
                    shifted,
                    iterations=smoothing_iterations,
                    max_edge=smoothing_max_edge,
                )
            )

        if polygons:
            out[region_id] = polygons

    return out
