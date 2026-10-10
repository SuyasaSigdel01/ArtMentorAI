"""Where to print the number inside a region.

The centroid is the obvious choice and the wrong one: for a crescent or an
L-shape the centroid falls outside the region entirely, and the number lands in
a neighbour. The correct point is the *pole of inaccessibility* — the interior
point furthest from any edge — which is also exactly the point with the most
room around it for a legible digit.

A Euclidean distance transform gives both the point and the available radius in
one shot, and the radius then drives font size directly.
"""

from __future__ import annotations

import numpy as np
from scipy import ndimage

__all__ = ["compute_label_points"]


def compute_label_points(
    region_map: np.ndarray, n_regions: int
) -> dict[int, tuple[tuple[int, int], float]]:
    """Return {region_id: ((x, y), radius)} for every region.

    Work is done inside each region's bounding box rather than on the full image,
    which keeps this linear in total image area instead of regions x area.
    """
    boxes = ndimage.find_objects(region_map)
    results: dict[int, tuple[tuple[int, int], float]] = {}

    for region_id in range(1, n_regions + 1):
        if region_id - 1 >= len(boxes):
            break
        box = boxes[region_id - 1]
        if box is None:
            continue

        mask = region_map[box] == region_id
        # Pad so pixels touching the crop edge are correctly treated as boundary.
        padded = np.pad(mask, 1, mode="constant", constant_values=False)

        distance = ndimage.distance_transform_edt(padded)
        flat_index = int(np.argmax(distance))
        iy, ix = np.unravel_index(flat_index, distance.shape)
        radius = float(distance[iy, ix])

        x = int(ix) - 1 + int(box[1].start)
        y = int(iy) - 1 + int(box[0].start)
        results[region_id] = ((x, y), radius)

    return results
