"""
Estimate the touch pose at a user-selected 2D point on a Zivid captured image.

Usage:
    - From ZDF file:  python touch_pose_estimation.py --zdf path/to/file.zdf
    - From camera:    python touch_pose_estimation.py --live

Workflow:
    1. Capture or load a ZDF frame (2D image + point cloud).
    2. Display the 2D RGB image; user clicks a touch point.
    3. Look up the 3D coordinates at the clicked pixel.
    4. Fit a local plane around the touch point using SVD.
    5. Build a 4x4 pose matrix (Z aligned with surface normal).
    6. Visualize the touch pose as a coordinate frame on the point cloud.

"""

import argparse
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np
import open3d as o3d
import zivid
from nptyping import Floating, NDArray, Shape


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _options() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--zdf", type=Path, metavar="FILE", help="Path to a ZDF file")
    source.add_argument("--live", action="store_true", help="Capture from connected camera")
    parser.add_argument(
        "--roi-radius",
        type=float,
        default=10.0,
        metavar="MM",
        help="Radius (mm) of the local region used for plane fitting (default: 10)",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Image display & point selection
# ---------------------------------------------------------------------------

def _select_touch_point(image_bgr: np.ndarray) -> Optional[Tuple[int, int]]:
    """Show the 2D image and let the user click a touch point.

    Args:
        image_bgr: OpenCV BGR image

    Returns:
        (u, v) pixel coordinate selected by the user, or None if cancelled

    """
    selected = {}

    def _on_mouse(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            selected["point"] = (x, y)
            print(f"Touch point selected: u={x}, v={y}")

    window = "Select touch point  (left-click → Enter to confirm  |  Esc to cancel)"
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window, 1280, 720)
    cv2.setMouseCallback(window, _on_mouse)

    display = image_bgr.copy()

    while True:
        frame = display.copy()
        if "point" in selected:
            u, v = selected["point"]
            cv2.circle(frame, (u, v), 8, (0, 255, 0), -1)
            cv2.circle(frame, (u, v), 12, (0, 200, 0), 2)
            cv2.putText(frame, f"({u}, {v})", (u + 15, v - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.imshow(window, frame)
        key = cv2.waitKey(20)
        if key == 13 and "point" in selected:   # Enter
            break
        if key == 27:                            # Esc
            selected.clear()
            break

    cv2.destroyAllWindows()
    return selected.get("point")


# ---------------------------------------------------------------------------
# 3D geometry helpers
# ---------------------------------------------------------------------------

def _get_3d_point(xyz: np.ndarray, u: int, v: int) -> Optional[np.ndarray]:
    """Return the 3D point at pixel (u, v).  Returns None if the point is NaN.

    Args:
        xyz: (H, W, 3) point cloud array
        u: column index (pixel x)
        v: row index    (pixel y)

    Returns:
        3D point as (3,) array, or None

    """
    point = xyz[v, u]
    if np.any(np.isnan(point)):
        print(f"Warning: No valid 3D point at pixel ({u}, {v})")
        return None
    return point


def _roi_sphere(
    xyz: np.ndarray,
    center: np.ndarray,
    radius: float,
) -> np.ndarray:
    """Mask the point cloud to a sphere of given radius around center.

    Args:
        xyz: (H, W, 3) point cloud array
        center: (3,) center point in mm
        radius: sphere radius in mm

    Returns:
        xyz array with points outside the sphere set to NaN

    """
    xyz_filtered = xyz.copy()
    dist = np.linalg.norm(xyz_filtered - center, axis=2)
    xyz_filtered[dist > radius] = np.nan
    return xyz_filtered


def _plane_fit(
    points: np.ndarray,
) -> Tuple[NDArray[Shape["3, 3"], Floating], NDArray[Shape["3"], Floating]]:
    """Fit a plane to a set of 3D points using SVD.

    Args:
        points: (H, W, 3) or (N, 3) array (NaN values are ignored)

    Returns:
        u_matrix: (3, 3) U matrix from SVD — columns are [X, Y, Z] axes of the plane
        centroid:  (3,) centroid of valid points

    """
    pts = points.reshape(-1, 3)
    pts = pts[~np.isnan(pts).any(axis=1)]
    if len(pts) < 3:
        raise RuntimeError("Not enough valid points to fit a plane (need ≥ 3)")
    centroid = pts.mean(axis=0)
    M = (pts - centroid).T @ (pts - centroid)
    u_matrix = np.linalg.svd(M)[0]
    return u_matrix, centroid


def _build_pose(
    u_matrix: np.ndarray,
    touch_point: np.ndarray,
) -> np.ndarray:
    """Build a 4x4 pose matrix with Z aligned to the surface normal.

    X axis is determined by the dominant image-plane direction,
    Y axis is the cross product of Z and X.

    Args:
        u_matrix: (3, 3) U matrix from SVD (column 2 = surface normal)
        touch_point: (3,) translation in mm

    Returns:
        pose: (4, 4) transformation matrix

    """
    z_axis = u_matrix[:, 2]                        # surface normal
    x_axis = u_matrix[:, 0]                        # dominant in-plane direction
    y_axis = np.cross(z_axis, x_axis)
    y_axis /= np.linalg.norm(y_axis)
    x_axis = np.cross(y_axis, z_axis)
    x_axis /= np.linalg.norm(x_axis)

    pose = np.eye(4)
    pose[:3, 0] = x_axis
    pose[:3, 1] = y_axis
    pose[:3, 2] = z_axis
    pose[:3, 3] = touch_point
    return pose


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------

def _make_open3d_point_cloud(xyz: np.ndarray, rgba: np.ndarray) -> o3d.geometry.PointCloud:
    pts = np.nan_to_num(xyz).reshape(-1, 3)
    rgb = rgba[:, :, :3].reshape(-1, 3) / 255.0
    pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pts))
    pcd.colors = o3d.utility.Vector3dVector(rgb)
    pcd = pcd.remove_non_finite_points(remove_nan=True, remove_infinite=True)
    return pcd


def _visualize_touch_pose(
    pcd: o3d.geometry.PointCloud,
    pose: np.ndarray,
    roi_xyz: Optional[np.ndarray] = None,
    roi_rgba: Optional[np.ndarray] = None,
) -> None:
    """Visualize the point cloud with the touch pose coordinate frame.

    Args:
        pcd: full scene point cloud
        pose: (4, 4) touch pose matrix
        roi_xyz: optional ROI point cloud for highlight
        roi_rgba: optional ROI rgba for highlight

    """
    coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=20)
    coord_frame.transform(pose)

    geometries = [pcd, coord_frame]

    if roi_xyz is not None and roi_rgba is not None:
        roi_pcd = _make_open3d_point_cloud(roi_xyz, roi_rgba)
        # paint ROI red for visibility
        roi_pcd.paint_uniform_color([1.0, 0.2, 0.2])
        geometries.append(roi_pcd)

    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name="Touch Pose Estimation", width=1600, height=900)
    for g in geometries:
        vis.add_geometry(g)
    opt = vis.get_render_option()
    opt.point_size = 1.5
    opt.background_color = [0.1, 0.1, 0.1]
    opt.show_coordinate_frame = True
    vc = vis.get_view_control()
    vc.set_front([0, 0, -1])
    vc.set_up([0, -1, 0])
    vis.run()
    vis.destroy_window()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def _main() -> None:
    user_options = _options()

    with zivid.Application() as app:
        if user_options.live:
            print("Connecting to camera...")
            with app.connect_camera() as camera:
                print("Capturing frame...")
                settings = zivid.capture_assistant.suggest_settings(
                    camera,
                    zivid.capture_assistant.SuggestSettingsParameters(
                        max_capture_time=__import__("datetime").timedelta(milliseconds=1200),
                        ambient_light_frequency=zivid.capture_assistant.SuggestSettingsParameters.AmbientLightFrequency.none,
                    ),
                )
                frame = camera.capture_2d_3d(settings)
        else:
            print(f"Loading ZDF file: {user_options.zdf}")
            frame = zivid.Frame(user_options.zdf)

        point_cloud = frame.point_cloud()
        xyz  = point_cloud.copy_data("xyz")
        rgba = point_cloud.copy_data("rgba")

    # 2D image for point selection
    rgb_image = rgba[:, :, :3]
    bgr_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)

    print("\nClick on the touch point in the image window, then press Enter.")
    touch_uv = _select_touch_point(bgr_image)
    if touch_uv is None:
        print("No point selected. Exiting.")
        return

    u, v = touch_uv
    touch_3d = _get_3d_point(xyz, u, v)
    if touch_3d is None:
        print("Selected pixel has no valid depth. Please try another point.")
        return

    print(f"\n3D touch point: X={touch_3d[0]:.2f}, Y={touch_3d[1]:.2f}, Z={touch_3d[2]:.2f} mm")

    roi_xyz = _roi_sphere(xyz, touch_3d, radius=user_options.roi_radius)
    u_matrix, centroid = _plane_fit(roi_xyz)
    print(f"Plane centroid: {centroid}")
    print(f"Surface normal (Z): {u_matrix[:, 2]}")

    pose = _build_pose(u_matrix, touch_3d)
    print(f"\nTouch pose (4x4):\n{np.round(pose, 4)}")

    pcd = _make_open3d_point_cloud(xyz, rgba)
    _visualize_touch_pose(pcd, pose, roi_xyz=roi_xyz, roi_rgba=rgba)


if __name__ == "__main__":
    _main()
