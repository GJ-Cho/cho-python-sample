"""
Offline unit test for the geometry pipeline (pixel_to_point / plane_fit /
waypoint_builder), using the local sample point cloud - no camera/robot needed.

The second half is the case build_paths exists for: two strokes drawn apart must
become two paths, with nothing traced between them.

Run:
    python -m line_tracing_gui._dev_test_waypoint_builder

"""

from pathlib import Path

import numpy as np
import zivid
from zividsamples.transformation_matrix import TransformationMatrix

from line_tracing_gui.geometry.waypoint_builder import build_paths, build_waypoints

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
    # --- several drawn lines stay several paths ---------------------------------------
    #
    # The point of build_paths. Two strokes drawn well apart used to be concatenated into one
    # list of pixels before reaching the builder, and _resample_polyline walks consecutive
    # pairs - so it laid evenly spaced samples along the gap between them. Those became real
    # waypoints, and the robot traced a line nobody drew, tip down, across whatever lay
    # between. This shows the phantom points are there in the flattened build and gone in the
    # per-stroke one.
    stroke_a = [(400.0, 400.0), (600.0, 400.0), (800.0, 400.0)]
    stroke_b = [(400.0, 700.0), (600.0, 700.0), (800.0, 700.0)]
    spacing_mm = 20.0

    paths_result = build_paths(
        [stroke_a, stroke_b], point_cloud_xyz, identity_hand_eye, sample_spacing_mm=spacing_mm
    )
    flattened = build_waypoints(
        stroke_a + stroke_b, point_cloud_xyz, identity_hand_eye, sample_spacing_mm=spacing_mm
    )

    print()
    print(f"two strokes drawn apart: {len(paths_result.paths)} paths, "
          f"{paths_result.waypoint_count} waypoints")
    print(f"the same points flattened into one line: {len(flattened.waypoints)} waypoints")

    assert len(paths_result.paths) == 2, f"expected one path per stroke, got {len(paths_result.paths)}"
    assert all(len(path) >= 2 for path in paths_result.paths)
    assert paths_result.dropped_stroke_count == 0

    drawn = np.array(
        [waypoint.translation for path in paths_result.paths for waypoint in path], dtype=float
    )

    def distance_to_drawn(position) -> float:
        return float(np.min(np.linalg.norm(drawn - np.asarray(position, dtype=float), axis=1)))

    # A waypoint further than a couple of samples from anything on either drawn line belongs
    # to neither of them.
    phantom_threshold_mm = 2.0 * spacing_mm
    phantom = [
        waypoint
        for waypoint in flattened.waypoints
        if distance_to_drawn(waypoint.translation) > phantom_threshold_mm
    ]
    print(f"waypoints in the flattened build that lie on neither drawn line: {len(phantom)}")
    assert phantom, (
        "expected the flattened build to interpolate across the gap - if it no longer does, "
        "this test has stopped demonstrating anything"
    )
    assert len(flattened.waypoints) > paths_result.waypoint_count, (
        len(flattened.waypoints),
        paths_result.waypoint_count,
    )

    # And the per-stroke build has none: every waypoint sits on the stroke it came from.
    for path_index, path in enumerate(paths_result.paths):
        single = build_waypoints(
            [stroke_a, stroke_b][path_index],
            point_cloud_xyz,
            identity_hand_eye,
            sample_spacing_mm=spacing_mm,
        )
        assert len(path) == len(single.waypoints), (len(path), len(single.waypoints))
        for built, alone in zip(path, single.waypoints):
            assert np.allclose(built.translation, alone.translation, atol=1e-4)
        print(f"  path {path_index + 1}: {len(path)} waypoints, identical to building that stroke alone")

    # A stroke with nothing traceable in it is dropped, not fatal to the ones beside it.
    with_stray = build_paths(
        [stroke_a, [(10.0, 10.0)]], point_cloud_xyz, identity_hand_eye, sample_spacing_mm=spacing_mm
    )
    assert len(with_stray.paths) == 1, f"a stray click should not become a path: {len(with_stray.paths)}"
    assert with_stray.dropped_stroke_count == 1
    print("  a stray single-point stroke is dropped, the real line survives")

    print("OK")


if __name__ == "__main__":
    _main()
