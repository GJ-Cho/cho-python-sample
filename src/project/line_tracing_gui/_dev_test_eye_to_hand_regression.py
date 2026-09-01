"""
Regression guard for the eye-to-hand waypoint path - no camera/robot needed.

Eye-to-hand was validated on the real robot first, and eye-in-hand was added on
top of it afterwards. Those two share everything except which transform reaches
the base frame, so a change meant for eye-in-hand can silently move eye-to-hand
waypoints. This pins eye-to-hand down.

Two independent checks:

1. The chain is exactly the hand-eye transform and nothing else. Verified as an
   invariant rather than against stored numbers: building with a hand-eye
   transform must equal that transform applied to the build with an identity
   one. This catches any extra term creeping into camera_to_base_transform.
2. Golden values on the sample point cloud. These were taken while eye-to-hand
   was passing on the real robot, and re-checked to be bit-identical to the
   implementation at commit 85d774f, so a mismatch means the geometry moved.

Run:
    python -m line_tracing_gui._dev_test_eye_to_hand_regression

"""

from pathlib import Path

import numpy as np
import zivid
from scipy.spatial.transform import Rotation
from zividsamples.transformation_matrix import TransformationMatrix

from line_tracing_gui.geometry.waypoint_builder import build_waypoints

SAMPLE_ZDF_PATH = Path(__file__).resolve().parent.parent.parent.parent / "sample" / "sample_MR130.zdf"

# Fixed inputs - changing any of these invalidates the golden values below.
LINE_POINTS_PX = [(400.0, 500.0), (600.0, 480.0), (800.0, 520.0), (1000.0, 500.0)]
SAMPLE_SPACING_MM = 5.0
HAND_EYE = TransformationMatrix(
    rotation=Rotation.from_euler("xyz", [12, -30, 47], degrees=True),
    translation=np.array([-410.0, 265.0, 730.0], dtype=np.float32),
)

EXPECTED_WAYPOINT_COUNT = 118
EXPECTED_SKIPPED_COUNT = 0
EXPECTED_MERGED_COUNT = 5
EXPECTED_FIRST_TRANSLATION = np.array([-676.7588, -358.0847, 1591.7705])
EXPECTED_LAST_TRANSLATION = np.array([-481.2441, -132.6024, 1705.9860])

# TransformationMatrix keeps translation as float32 and round-trips through as_matrix() /
# from_matrix(), so composing the same transform two different ways lands ~1e-5 mm and
# ~1e-6 deg apart. These are set above that noise and still far below anything physical:
# 1e-3 deg over a metre of reach is 17 um.
TRANSLATION_TOLERANCE_MM = 1e-3
ROTATION_TOLERANCE_DEG = 1e-3


def _report(passed: bool, description: str, detail: str = "") -> bool:
    print(f"  [{'PASS' if passed else 'FAIL'}] {description}{('  - ' + detail) if detail else ''}")
    return passed


def _main() -> None:
    app = zivid.Application()  # noqa: F841 (keeps the SDK runtime alive while loading the frame)
    point_cloud_xyz = zivid.Frame(str(SAMPLE_ZDF_PATH)).point_cloud().copy_data("xyz")
    print(f"Point cloud {point_cloud_xyz.shape} from {SAMPLE_ZDF_PATH.name}\n")

    build_kwargs = {"eye_in_hand": False, "sample_spacing_mm": SAMPLE_SPACING_MM}
    result = build_waypoints(LINE_POINTS_PX, point_cloud_xyz, HAND_EYE, **build_kwargs)
    identity_result = build_waypoints(
        LINE_POINTS_PX, point_cloud_xyz, TransformationMatrix(), **build_kwargs
    )

    checks = []

    print("1. camera_to_base is exactly the hand-eye transform")
    same_length = len(result.waypoints) == len(identity_result.waypoints)
    checks.append(_report(same_length, "same waypoint count either way"))
    if same_length and result.waypoints:
        worst_translation = max(
            float(np.linalg.norm((HAND_EYE * camera_frame).translation - waypoint.translation))
            for waypoint, camera_frame in zip(result.waypoints, identity_result.waypoints)
        )
        worst_rotation_deg = np.degrees(
            max(
                float(((HAND_EYE * camera_frame).rotation.inv() * waypoint.rotation).magnitude())
                for waypoint, camera_frame in zip(result.waypoints, identity_result.waypoints)
            )
        )
        checks.append(
            _report(
                worst_translation < TRANSLATION_TOLERANCE_MM and worst_rotation_deg < ROTATION_TOLERANCE_DEG,
                "hand_eye * build(identity) == build(hand_eye)",
                f"worst {worst_translation:.3e} mm / {worst_rotation_deg:.3e} deg",
            )
        )

    print("\n2. golden values on the sample point cloud")
    checks.append(
        _report(
            len(result.waypoints) == EXPECTED_WAYPOINT_COUNT,
            f"waypoint count == {EXPECTED_WAYPOINT_COUNT}",
            f"got {len(result.waypoints)}",
        )
    )
    checks.append(
        _report(
            result.skipped_pixel_count == EXPECTED_SKIPPED_COUNT,
            f"skipped == {EXPECTED_SKIPPED_COUNT}",
            f"got {result.skipped_pixel_count}",
        )
    )
    checks.append(
        _report(
            result.merged_close_waypoint_count == EXPECTED_MERGED_COUNT,
            f"merged == {EXPECTED_MERGED_COUNT}",
            f"got {result.merged_close_waypoint_count}",
        )
    )
    if result.waypoints:
        for name, expected, actual in (
            ("first", EXPECTED_FIRST_TRANSLATION, result.waypoints[0].translation),
            ("last", EXPECTED_LAST_TRANSLATION, result.waypoints[-1].translation),
        ):
            deviation = float(np.linalg.norm(np.asarray(actual, dtype=float) - expected))
            checks.append(
                _report(
                    deviation < TRANSLATION_TOLERANCE_MM,
                    f"{name} waypoint translation",
                    f"off by {deviation:.3e} mm, got {np.round(actual, 4)}",
                )
            )

    print(f"\n{'ALL CHECKS PASSED' if all(checks) else 'FAILURES ABOVE'} ({sum(checks)}/{len(checks)})")
    if not all(checks):
        raise SystemExit(1)


if __name__ == "__main__":
    _main()
