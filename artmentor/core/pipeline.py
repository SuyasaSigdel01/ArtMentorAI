"""The pattern pipeline: photograph in, paintable numbered pattern out.

Everything is deterministic. Same image plus same params plus same seed gives the
same pattern, every time. That property is worth protecting: it is what makes the
output debuggable, cacheable, and explainable to a user who asks why their
pattern changed.

`PIPELINE_VERSION` is recorded on every Pattern. Bump it whenever a change alters
output for the same inputs, so old patterns stay attributable to the code that
produced them and regressions are visible rather than mysterious.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import cv2
import numpy as np

from .color.space import lab_to_rgb_array
from .imaging.contours import extract_contours
from .imaging.labeling import compute_label_points
from .imaging.quantize import presmooth, quantize
from .imaging.segment import build_regions, smooth_label_map
from .types import LabColor, PaletteEntry, Pattern, Region, RGBColor

__all__ = ["PIPELINE_VERSION", "PatternParams", "generate_pattern"]

PIPELINE_VERSION = "0.1.0"


@dataclass(frozen=True)
class PatternParams:
    """Tuning knobs for pattern generation.

    The defaults are tuned for a roughly A4 print at a comfortable beginner
    difficulty. `n_colors` and `min_region_area` together are effectively the
    difficulty dial: more colors and smaller regions means a harder painting.
    """

    n_colors: int = 12
    max_dimension: int = 900          # longest side, in pixels, before processing
    presmooth_diameter: int = 9       # 0 disables the bilateral filter
    presmooth_sigma_color: float = 35.0
    presmooth_sigma_space: float = 9.0
    label_smoothing: int = 5          # vote kernel near edges; 1 disables
    flat_smoothing: int = 17          # wider vote kernel in flat areas such as sky
    edge_percentile: float = 70.0     # below this gradient percentile counts as flat
    min_region_area: int = 90         # px; below this a region is merged away
    simplify_epsilon: float = 0.75    # Douglas-Peucker tolerance
    contour_smoothing: int = 2        # corner-cutting passes; 0 disables
    contour_smoothing_max_edge: float = 9.0  # only cut corners between short edges
    seed: int = 0

    def as_dict(self) -> dict:
        return asdict(self)


def _fit_within(rgb: np.ndarray, max_dimension: int) -> np.ndarray:
    """Downscale so the longest side is at most `max_dimension`. Never upscales."""
    h, w = rgb.shape[:2]
    longest = max(h, w)
    if longest <= max_dimension:
        return rgb
    scale = max_dimension / float(longest)
    new_size = (max(1, int(round(w * scale))), max(1, int(round(h * scale))))
    # INTER_AREA is the correct choice for downscaling; it averages rather than
    # point-samples, so it does not alias fine detail into false colors.
    return cv2.resize(rgb, new_size, interpolation=cv2.INTER_AREA)


def generate_pattern(image_rgb: np.ndarray, params: PatternParams | None = None) -> Pattern:
    """Convert an RGB image into a numbered, paintable pattern."""
    params = params or PatternParams()

    if image_rgb.ndim != 3 or image_rgb.shape[2] != 3:
        raise ValueError(f"expected an (H, W, 3) RGB image, got {image_rgb.shape}")

    working = _fit_within(np.ascontiguousarray(image_rgb, dtype=np.uint8), params.max_dimension)

    if params.presmooth_diameter > 0:
        working = presmooth(
            working,
            diameter=params.presmooth_diameter,
            sigma_color=params.presmooth_sigma_color,
            sigma_space=params.presmooth_sigma_space,
        )

    quantized = quantize(working, params.n_colors, seed=params.seed)

    smoothed = smooth_label_map(
        quantized.labels,
        quantized.n_colors,
        kernel=params.label_smoothing,
        flat_kernel=params.flat_smoothing,
        edge_source=working,
        edge_percentile=params.edge_percentile,
    )

    regions = build_regions(
        smoothed, quantized.palette_lab, min_area=params.min_region_area
    )

    # Merging can retire a color entirely. Drop empty palette entries and renumber
    # so the printed legend never shows a number that appears nowhere in the art.
    label_map = regions.label_map
    counts = np.bincount(label_map.ravel(), minlength=quantized.n_colors)
    keep = np.flatnonzero(counts > 0)

    remap = np.full(quantized.n_colors, -1, dtype=np.int32)
    remap[keep] = np.arange(keep.size, dtype=np.int32)

    label_map = remap[label_map]
    region_palette = remap[regions.region_palette]
    palette_lab = quantized.palette_lab[keep]
    palette_rgb = lab_to_rgb_array(palette_lab, as_uint8=True)
    counts = counts[keep]

    total_pixels = float(label_map.size)
    palette = [
        PaletteEntry(
            index=i,
            number=i + 1,
            rgb=RGBColor(int(palette_rgb[i][0]), int(palette_rgb[i][1]), int(palette_rgb[i][2])),
            lab=LabColor.from_array(palette_lab[i]),
            pixel_count=int(counts[i]),
            share=float(counts[i]) / total_pixels,
        )
        for i in range(keep.size)
    ]

    region_map = regions.region_map
    n_regions = regions.n_regions

    contours = extract_contours(
        region_map,
        n_regions,
        epsilon=params.simplify_epsilon,
        smoothing_iterations=params.contour_smoothing,
        smoothing_max_edge=params.contour_smoothing_max_edge,
    )
    label_points = compute_label_points(region_map, n_regions)

    # Centroids via weighted bincount: one pass over the image for all regions,
    # rather than one pass per region.
    ys, xs = np.indices(region_map.shape)
    flat = region_map.ravel()
    areas = np.bincount(flat, minlength=n_regions + 1).astype(np.float64)
    sum_x = np.bincount(flat, weights=xs.ravel(), minlength=n_regions + 1)
    sum_y = np.bincount(flat, weights=ys.ravel(), minlength=n_regions + 1)
    safe_areas = np.where(areas > 0, areas, 1.0)

    region_objects: list[Region] = []
    for region_id in range(1, n_regions + 1):
        point, radius = label_points.get(region_id, (None, 0.0))
        region_objects.append(
            Region(
                id=region_id,
                palette_index=int(region_palette[region_id]),
                area=int(areas[region_id]),
                centroid=(
                    float(sum_x[region_id] / safe_areas[region_id]),
                    float(sum_y[region_id] / safe_areas[region_id]),
                ),
                label_point=point,
                label_radius=float(radius),
                contours=contours.get(region_id, []),
            )
        )

    height, width = region_map.shape
    return Pattern(
        width=int(width),
        height=int(height),
        palette=palette,
        regions=region_objects,
        pipeline_version=PIPELINE_VERSION,
        params=params.as_dict(),
        label_map=label_map,
        region_map=region_map,
    )
