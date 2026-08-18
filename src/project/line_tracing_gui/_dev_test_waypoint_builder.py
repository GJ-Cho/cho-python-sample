"""
Offline unit test for the geometry pipeline (pixel_to_point / plane_fit /
waypoint_builder), using the local sample point cloud - no camera/robot needed.

Run:
    python -m line_tracing_gui._dev_test_waypoint_builder

"""

from pathlib import Path

import numpy as np
import zivid
from zividsamples.transformation_matrix import TransformationMatrix

from line_tracing_gui.geometry.waypoint_builder import build_waypoints

SAMPLE_ZDF_PATH = Path(__file__).resolve().parent.parent.parent.parent / "sample" / "sample_MR130.zdf"


def _main() -> None:
    app = zivid.Application()  # noqa: F841 (keeps the SDK runtime alive while loading the frame)
    frame = zivid.Frame(str(SAMPLE_ZDF_PATH))
    point_cloud_xyz = frame.point_cloud().copy_data("xyz")
    print(f"Point cloud shape: {point_cloud_xyz.shape}")

    # A simple straight line across a region known (from earlier inspection) to have valid depth.
    line_points_px = [(400.0, 500.0), (600.0, 500.0), (800.0, 500.0)]

    identity_hand_eye = TransformationMatrix()  # placeholder until Calibrate tab loads a real one
    result = build_waypoints(line_points_px, point_cloud_xyz, identity_hand_eye, sample_spacing_mm=20.0)

    print(f"{len(result.waypoints)} waypoints built, {result.skipped_pixel_count} pixels skipped")
    for i, waypoint in enumerate(result.waypoints):
        translation = waypoint.translation
        z_axis = waypoint.rotation.as_matrix()[:, 2]
        print(
            f"  [{i}] translation(mm)={np.round(translation, 1)}  "
            f"tool_z_axis={np.round(z_axis, 3)}  "
            f"det(R)={np.linalg.det(waypoint.rotation.as_matrix()):.4f}"
        )
        assert np.isclose(np.linalg.norm(translation), np.linalg.norm(translation))  # sanity: no NaN
        assert not np.any(np.isnan(waypoint.as_matrix())), "waypoint matrix contains NaN"
        assert np.isclose(np.linalg.det(waypoint.rotation.as_matrix()), 1.0, atol=1e-4), "not a proper rotation"

    assert result.skipped_pixel_count == 0, "expected all sample points to land on valid geometry"
    assert len(result.waypoints) >= 2
    print("OK")


if __name__ == "__main__":
    _main()
