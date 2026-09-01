"""
Build robot-base-frame waypoints from a drawn 2D polyline (pixel coordinates)
and a captured point cloud, following the local surface so the gripper
approaches perpendicular to it at every point along the line.

TCP offset is intentionally NOT applied here: the robot controller's own
set_tcp already accounts for the pointed gripper's tip offset (see
robot/README.md), so these poses are handed directly to
RobotControl.move_l/move_j/move_path as-is.

Tool-frame convention (see RobotControlURRTDE's approach/retreat pattern,
which this mirrors): the tool Z axis points INTO the surface (the direction
the gripper travels to make contact), i.e. the *opposite* of the outward-
facing surface normal. Confirmed against the real robot's TCP orientation
(local +Z moves the gripper tip forward, into the surface).

Orientation smoothing: each waypoint's local plane fit is computed
independently from a small pixel neighborhood, so on a noisy/textured
surface consecutive waypoints' normals can swing by tens of degrees even a
few mm apart. Left unsmoothed, this forces the robot to crawl through
move_path's blended movel segments to respect angular speed limits. Z axes
are therefore averaged over a small window along the path before building
the final rotations.

Rotation-minimizing frame (about the tool Z axis): the gripper tip is a
point, rotationally symmetric about tool Z, so how far the tool has spun
about its own Z axis is physically irrelevant - only Z's direction (the
surface normal) matters. Locking the X axis to the line's local travel
direction at every waypoint (the obvious-seeming choice) therefore adds
*unnecessary* rotation about Z on every curve, which UR's movel/movePath
must still physically execute - on a curvy line this dominates the
required joint rotation (confirmed: a synthetic tight-spiral test needed
up to 282 deg/s of implied joint rotation with tangent-locked X axes at
20 mm/s, vs 107 deg/s with the approach below, for the same path). X/Y
axes are instead propagated waypoint-to-waypoint by the *minimal* rotation
that tracks the change in Z, matching the well-known "rotation-minimizing
frame" / parallel-transport technique used for axisymmetric tools in
5-axis machining. The line's travel direction only seeds the very first
waypoint's X axis (an arbitrary but deterministic starting choice).

"""

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
from scipy.spatial.transform import Rotation
from zividsamples.transformation_matrix import TransformationMatrix

from line_tracing_gui.geometry.pixel_to_point import PointCloudXYZ, point_at_pixel, points_in_window
from line_tracing_gui.geometry.plane_fit import fit_plane, orient_normal_towards_camera

DEFAULT_PLANE_FIT_WINDOW = 15
DEFAULT_SAMPLE_SPACING_MM = 5.0
DEFAULT_ORIENTATION_SMOOTHING_WINDOW = 5  # waypoints, must be odd-ish (centered); see _smooth_z_axes
MIN_PLANE_FIT_POINTS = 3

# Resampling is done in *pixel* arc length (see _resample_polyline), so on a tightly curved
# or looped drawn line the real 3D distance between "evenly spaced" pixel samples can shrink
# well below the nominal sample_spacing_mm. Enforcing a floor here - a fraction of the
# nominal spacing - keeps the real spacing predictable regardless of how the line curves, so
# a blend radius chosen relative to sample_spacing_mm stays valid.
MIN_SPACING_FRACTION_OF_NOMINAL = 0.5

Pixel = Tuple[float, float]


@dataclass
class WaypointBuildResult:
    waypoints: List[TransformationMatrix]  # base frame, in order along the drawn line
    skipped_pixel_count: int  # resampled line points with no usable 3D data nearby
    merged_close_waypoint_count: int  # points dropped for being too close (in 3D) to the previous one


@dataclass
class _RawWaypoint:
    point: np.ndarray  # mm, camera frame
    z_axis: np.ndarray  # unit vector, into the surface (see module docstring)
    tangent_hint: np.ndarray  # unnormalized, roughly along the line's travel direction


def _resample_polyline(points: List[Pixel], spacing: float) -> List[Pixel]:
    """Resample a polyline to roughly even arc-length spacing (in pixels)."""
    if len(points) < 2 or spacing <= 0:
        return list(points)

    resampled: List[Pixel] = [points[0]]
    distance_since_last_sample = 0.0
    for previous, current in zip(points[:-1], points[1:]):
        start = np.array(previous, dtype=float)
        end = np.array(current, dtype=float)
        segment_vector = end - start
        segment_length = float(np.linalg.norm(segment_vector))
        if segment_length == 0:
            continue
        direction = segment_vector / segment_length

        traveled = 0.0
        while distance_since_last_sample + (segment_length - traveled) >= spacing:
            traveled += spacing - distance_since_last_sample
            resampled.append(tuple(start + direction * traveled))
            distance_since_last_sample = 0.0
        distance_since_last_sample += segment_length - traveled

    if resampled[-1] != tuple(points[-1]):
        resampled.append(tuple(points[-1]))
    return resampled


def _orthonormal_frame(z_axis: np.ndarray, tangent_hint: np.ndarray) -> Rotation:
    """Build a right-handed orthonormal rotation with the given Z axis, using
    `tangent_hint` to fix the X axis as closely as possible to the line's travel direction.
    """
    tangent_component = tangent_hint - np.dot(tangent_hint, z_axis) * z_axis
    tangent_norm = np.linalg.norm(tangent_component)
    if tangent_norm < 1e-6:
        # tangent_hint is (near-)parallel to z_axis - fall back to any vector orthogonal to it.
        fallback = np.array([1.0, 0.0, 0.0]) if abs(z_axis[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
        tangent_component = fallback - np.dot(fallback, z_axis) * z_axis
        tangent_norm = np.linalg.norm(tangent_component)
    x_axis = tangent_component / tangent_norm

    y_axis = np.cross(z_axis, x_axis)
    y_axis = y_axis / np.linalg.norm(y_axis)
    x_axis = np.cross(y_axis, z_axis)  # re-orthogonalize in case z_axis wasn't exactly unit length

    return Rotation.from_matrix(np.column_stack([x_axis, y_axis, z_axis]))


def _build_rotation_minimizing_frames(z_axes: List[np.ndarray], seed_tangent_hint: np.ndarray) -> List[Rotation]:
    """Build one rotation per Z axis, propagating X/Y waypoint-to-waypoint by the minimal
    rotation that tracks the change in Z (see module docstring - the gripper tip is
    rotationally symmetric about tool Z, so this avoids the unnecessary about-Z rotation a
    per-point tangent-locked X axis would add). `seed_tangent_hint` only fixes the first
    waypoint's X axis, as an arbitrary but deterministic starting choice.
    """
    if len(z_axes) == 0:
        return []

    rotations = [_orthonormal_frame(z_axis=z_axes[0], tangent_hint=seed_tangent_hint)]
    for index in range(1, len(z_axes)):
        z_previous = z_axes[index - 1]
        z_current = z_axes[index]
        rotation_axis = np.cross(z_previous, z_current)
        axis_norm = np.linalg.norm(rotation_axis)
        angle = float(np.arccos(np.clip(np.dot(z_previous, z_current), -1.0, 1.0)))

        previous_matrix = rotations[-1].as_matrix()
        if axis_norm < 1e-8 or angle < 1e-8:
            # Z didn't change (or is exactly opposite - degenerate, keep the prior frame as-is).
            x_axis, y_axis = previous_matrix[:, 0], previous_matrix[:, 1]
        else:
            delta_rotation = Rotation.from_rotvec((rotation_axis / axis_norm) * angle)
            x_axis = delta_rotation.apply(previous_matrix[:, 0])
            y_axis = delta_rotation.apply(previous_matrix[:, 1])

        # Re-orthogonalize against z_current defensively (guards against float drift over a long path).
        x_axis = x_axis - np.dot(x_axis, z_current) * z_current
        x_axis = x_axis / np.linalg.norm(x_axis)
        y_axis = np.cross(z_current, x_axis)
        y_axis = y_axis / np.linalg.norm(y_axis)
        rotations.append(Rotation.from_matrix(np.column_stack([x_axis, y_axis, z_current])))

    return rotations


def _compute_raw_waypoint(
    point_cloud_xyz: PointCloudXYZ,
    pixel: Pixel,
    next_pixel: Optional[Pixel],
    plane_fit_window: int,
) -> Optional[_RawWaypoint]:
    point_3d = point_at_pixel(point_cloud_xyz, pixel)
    if point_3d is None:
        return None

    neighborhood = points_in_window(point_cloud_xyz, pixel, plane_fit_window)
    if neighborhood.shape[0] < MIN_PLANE_FIT_POINTS:
        return None
    _, normal = fit_plane(neighborhood)
    outward_normal = orient_normal_towards_camera(normal, point_3d)
    into_surface = -outward_normal  # see module docstring: tool Z points into the surface

    next_point_3d = point_at_pixel(point_cloud_xyz, next_pixel) if next_pixel is not None else None
    tangent_hint = (
        (next_point_3d - point_3d) if next_point_3d is not None else np.array([1.0, 0.0, 0.0], dtype=np.float32)
    )

    return _RawWaypoint(point=point_3d, z_axis=into_surface, tangent_hint=tangent_hint)


def _enforce_minimum_spacing(raw_waypoints: List[_RawWaypoint], min_spacing_mm: float) -> Tuple[List[_RawWaypoint], int]:
    """Greedily drop points whose 3D distance to the last *kept* point is under
    min_spacing_mm. Returns (kept, number dropped)."""
    if len(raw_waypoints) < 2 or min_spacing_mm <= 0:
        return raw_waypoints, 0
    kept = [raw_waypoints[0]]
    dropped = 0
    for candidate in raw_waypoints[1:]:
        distance = float(np.linalg.norm(candidate.point - kept[-1].point))
        if distance >= min_spacing_mm:
            kept.append(candidate)
        else:
            dropped += 1
    return kept, dropped


def _smooth_z_axes(raw_waypoints: List[_RawWaypoint], window: int) -> List[np.ndarray]:
    """Average each waypoint's Z axis with its neighbors along the path (see module
    docstring - independently-fitted per-point normals otherwise jitter enough to force the
    robot to crawl through move_path's blended segments)."""
    if window <= 1:
        return [w.z_axis for w in raw_waypoints]
    z_axes = [w.z_axis for w in raw_waypoints]
    count = len(z_axes)
    half = window // 2
    smoothed = []
    for i in range(count):
        start = max(0, i - half)
        end = min(count, i + half + 1)
        averaged = np.mean(z_axes[start:end], axis=0)
        norm = np.linalg.norm(averaged)
        smoothed.append(averaged / norm if norm > 1e-6 else z_axes[i])
    return smoothed


def build_waypoints(
    line_points_px: List[Pixel],
    point_cloud_xyz: PointCloudXYZ,
    hand_eye_transform: TransformationMatrix,
    eye_in_hand: bool = False,
    robot_pose: Optional[TransformationMatrix] = None,
    sample_spacing_mm: float = DEFAULT_SAMPLE_SPACING_MM,
    plane_fit_window: int = DEFAULT_PLANE_FIT_WINDOW,
    orientation_smoothing_window: int = DEFAULT_ORIENTATION_SMOOTHING_WINDOW,
) -> WaypointBuildResult:
    """Turn a drawn line (pixel coordinates on the captured 2D image) into an ordered list
    of robot-base-frame waypoints that follow the local surface under the line.

    Args:
        line_points_px: Drawn line, as (col, row) pixel coordinates, in drawing order.
        point_cloud_xyz: Organized point cloud XYZ (mm) from the same capture as the image
            the line was drawn on (pixel-aligned 1:1 - see module docstring).
        hand_eye_transform: Hand-eye calibration. For eye-to-hand (camera fixed), this maps
            camera frame directly to robot base frame. For eye-in-hand (camera on the
            gripper), it maps camera frame to the robot's end-effector/flange frame instead,
            and needs `robot_pose` to reach the base frame (see `eye_in_hand`).
        eye_in_hand: False (default) for a fixed camera - point_base = hand_eye_transform * point_camera.
            True if the camera is mounted on the robot - point_base = robot_pose * hand_eye_transform * point_camera,
            in which case `robot_pose` (the robot's pose *at the moment the frame was captured*,
            not its current pose) is required.
        robot_pose: Robot end-effector pose in base frame at capture time. Required (and only
            used) when `eye_in_hand` is True.
        sample_spacing_mm: Target spacing between waypoints, approximated in pixel space
            (an approximation - pixel-to-mm scale varies with distance to the surface).
        plane_fit_window: Pixel neighborhood size used to estimate the local surface normal.
        orientation_smoothing_window: Number of neighboring waypoints averaged together when
            smoothing the tool Z axis along the path (see module docstring). 1 disables smoothing.

    Raises:
        ValueError: if fewer than 2 line points are given, or if eye_in_hand is True but
            robot_pose was not given.
    """
    if len(line_points_px) < 2:
        raise ValueError("Need at least 2 line points to build waypoints.")
    if eye_in_hand and robot_pose is None:
        raise ValueError("eye_in_hand=True requires robot_pose (the robot pose at capture time).")

    camera_to_base_transform = (robot_pose * hand_eye_transform) if eye_in_hand else hand_eye_transform

    resampled_px = _resample_polyline(line_points_px, spacing=sample_spacing_mm)

    raw_waypoints: List[_RawWaypoint] = []
    skipped = 0
    for index, pixel in enumerate(resampled_px):
        next_pixel = resampled_px[index + 1] if index + 1 < len(resampled_px) else None
        raw_waypoint = _compute_raw_waypoint(point_cloud_xyz, pixel, next_pixel, plane_fit_window)
        if raw_waypoint is None:
            skipped += 1
            continue
        raw_waypoints.append(raw_waypoint)

    min_spacing_mm = sample_spacing_mm * MIN_SPACING_FRACTION_OF_NOMINAL
    raw_waypoints, merged_count = _enforce_minimum_spacing(raw_waypoints, min_spacing_mm)

    smoothed_z_axes = _smooth_z_axes(raw_waypoints, orientation_smoothing_window)
    seed_tangent_hint = raw_waypoints[0].tangent_hint if raw_waypoints else np.array([1.0, 0.0, 0.0])
    rotations = _build_rotation_minimizing_frames(smoothed_z_axes, seed_tangent_hint)

    waypoints: List[TransformationMatrix] = []
    for raw_waypoint, rotation in zip(raw_waypoints, rotations):
        pose_camera_frame = TransformationMatrix(rotation=rotation, translation=raw_waypoint.point)
        waypoints.append(camera_to_base_transform * pose_camera_frame)

    return WaypointBuildResult(waypoints=waypoints, skipped_pixel_count=skipped, merged_close_waypoint_count=merged_count)
