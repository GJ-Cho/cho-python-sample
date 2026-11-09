"""
Look up 3D points from a Zivid organized point cloud at given 2D pixel
coordinates. A `camera.capture()` frame's point cloud XYZ array and its color
image share the same (H, W) shape and are pixel-aligned 1:1 (unlike the
by2x2-binned Live 2D preview), so a pixel drawn on the captured image maps
directly to an (row, col) index here. Units are millimeters, matching Zivid's
default and the rest of this codebase (e.g. TransformationMatrix.translation).
Invalid/missing points are NaN, per Zivid convention.

"""

from typing import Optional, Tuple

import numpy as np
from nptyping import Float32, NDArray, Shape

PointCloudXYZ = NDArray[Shape["H, W, 3"], Float32]  # type: ignore


def point_at_pixel(
    point_cloud_xyz: PointCloudXYZ, pixel: Tuple[float, float], search_radius: int = 3
) -> Optional[np.ndarray]:
    """Look up the 3D point nearest to the given (col, row) pixel.

    If the exact pixel is invalid (NaN, e.g. a hole or the line was drawn just off an edge),
    searches outward in growing square rings up to `search_radius` pixels for the nearest
    valid point.

    Returns:
        The (3,) xyz point in mm, or None if no valid point was found within the radius.
    """
    height, width = point_cloud_xyz.shape[:2]
    center_col, center_row = int(round(pixel[0])), int(round(pixel[1]))
    if not (0 <= center_col < width and 0 <= center_row < height):
        return None

    best_point: Optional[np.ndarray] = None
    best_dist_sq: Optional[int] = None
    for radius in range(search_radius + 1):
        for row_offset in range(-radius, radius + 1):
            for col_offset in range(-radius, radius + 1):
                if max(abs(row_offset), abs(col_offset)) != radius:
                    continue  # only the newly-added ring at this radius; inner cells already checked
                row, col = center_row + row_offset, center_col + col_offset
                if not (0 <= row < height and 0 <= col < width):
                    continue
                candidate = point_cloud_xyz[row, col]
                if np.any(np.isnan(candidate)):
                    continue
                dist_sq = row_offset * row_offset + col_offset * col_offset
                if best_dist_sq is None or dist_sq < best_dist_sq:
                    best_point, best_dist_sq = candidate, dist_sq
        if best_point is not None:
            return np.array(best_point, dtype=np.float32)
    return None


def points_in_window(point_cloud_xyz: PointCloudXYZ, center_pixel: Tuple[float, float], window: int) -> np.ndarray:
    """Return the valid (non-NaN) 3D points in a `window` x `window` pixel neighborhood
    centered on `center_pixel` (col, row), as an (N, 3) array.
    """
    height, width = point_cloud_xyz.shape[:2]
    half = window // 2
    center_col, center_row = int(round(center_pixel[0])), int(round(center_pixel[1]))
    col_start, col_end = max(0, center_col - half), min(width, center_col + half + 1)
    row_start, row_end = max(0, center_row - half), min(height, center_row + half + 1)
    patch = point_cloud_xyz[row_start:row_end, col_start:col_end].reshape(-1, 3)
    valid = ~np.isnan(patch).any(axis=1)
    return patch[valid]
