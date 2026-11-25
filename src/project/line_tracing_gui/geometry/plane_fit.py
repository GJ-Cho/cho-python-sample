"""
Fit a local plane to a small 3D point neighborhood (least-squares via SVD)
and return its normal vector.

"""

from typing import Tuple

import numpy as np


def fit_plane(points: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Least-squares plane fit through `points` (N, 3).

    Returns:
        (centroid, normal): normal is unit length; its sign is arbitrary here
        (whichever side the SVD happens to pick) - see orient_normal_towards_camera.

    Raises:
        ValueError: if fewer than 3 points are given.
    """
    if points.shape[0] < 3:
        raise ValueError(f"Need at least 3 points to fit a plane, got {points.shape[0]}.")
    centroid = points.mean(axis=0)
    _, _, vh = np.linalg.svd(points - centroid, full_matrices=False)
    normal = vh[-1]
    return centroid, normal / np.linalg.norm(normal)


def orient_normal_towards_camera(normal: np.ndarray, point_on_plane: np.ndarray) -> np.ndarray:
    """Flip `normal` if needed so it points back towards the camera origin (0, 0, 0) in
    camera frame, i.e. away from the surface, towards the side the camera/robot approaches from.
    """
    towards_camera = -point_on_plane
    if np.dot(normal, towards_camera) < 0:
        return -normal
    return normal
