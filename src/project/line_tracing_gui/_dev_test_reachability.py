"""
Offline checks for robot.reachability - no robot, no camera, no Zivid runtime.

check_path_reachability talks to the robot through five query methods only, so
the part worth testing without hardware - which steps get checked, in what
order, how verdicts and qnear chaining are handed along, what the report
concludes - can be exercised against a stand-in. What is NOT tested here is
whether UR's own IK and safety limits agree with the real cell; that is the
controller's answer to give, and only the real robot can give it.

The stand-in deliberately implements *nothing but* those five queries: if the
check ever started moving the robot, this test would fail with AttributeError
rather than quietly pass. That property is the point - a "pre-flight check"
that moves the robot would be worse than none at all.

Its kinematics are a toy (joint angles are a smooth function of position, and
qnear is ignored), which is enough because the joint numbers only ever feed two
things: the safety-limit query and the size of the step between neighbours.

Run:
    python -m line_tracing_gui._dev_test_reachability

"""

import math
from typing import List, Optional, Sequence, Tuple

import numpy as np
from zividsamples.transformation_matrix import TransformationMatrix

from line_tracing_gui.robot.reachability import (
    DEFAULT_JOINT_STEP_WARNING_DEG,
    MOTION_JOINT,
    MOTION_LINEAR,
    check_path_reachability,
)

MAX_REACH_MM = 1000.0
FLOOR_Z_MM = 0.0
JOINT_LIMIT_RAD = 2.0 * math.pi
# Comfortably over DEFAULT_JOINT_STEP_WARNING_DEG so the warning is unambiguous.
FLIP_OFFSET_DEG = 60.0


class FakeRobot:
    """The five queries check_path_reachability needs, and not one method more.

    Reach is a sphere of MAX_REACH_MM about the origin; the safety "plane" is z >= floor;
    joint limits are a symmetric range. `flip_x_range` injects a discontinuity in the joint
    solution over a span of X, standing in for IK resolving into a different arm
    configuration part-way along a path.
    """

    def __init__(
        self,
        max_reach_mm: float = MAX_REACH_MM,
        floor_z_mm: float = FLOOR_Z_MM,
        joint_limit_rad: float = JOINT_LIMIT_RAD,
        flip_x_range: Optional[Tuple[float, float]] = None,
        current_joints: Optional[List[float]] = None,
        raise_on_query: Optional[Exception] = None,
    ) -> None:
        self.max_reach_mm = max_reach_mm
        self.floor_z_mm = floor_z_mm
        self.joint_limit_rad = joint_limit_rad
        self.flip_x_range = flip_x_range
        self.current_joints = current_joints if current_joints is not None else [0.0] * 6
        self.raise_on_query = raise_on_query
        self.query_count = 0

    def _joints_for(self, pose: TransformationMatrix) -> List[float]:
        x, y, z = (float(value) for value in pose.translation)
        joints = [
            math.atan2(y, x),
            math.hypot(x, y) / self.max_reach_mm,
            z / self.max_reach_mm,
            0.0,
            math.pi / 2.0,
            0.0,
        ]
        if self.flip_x_range is not None and self.flip_x_range[0] <= x <= self.flip_x_range[1]:
            joints[3] += math.radians(FLIP_OFFSET_DEG)
        return joints

    def _tick(self) -> None:
        self.query_count += 1
        if self.raise_on_query is not None:
            raise self.raise_on_query

    def get_joint_positions(self) -> List[float]:
        self._tick()
        return list(self.current_joints)

    def has_inverse_kinematics_solution(
        self, pose: TransformationMatrix, qnear: Optional[List[float]] = None
    ) -> bool:
        self._tick()
        return bool(np.linalg.norm(np.asarray(pose.translation, dtype=float)) <= self.max_reach_mm)

    def get_inverse_kinematics(
        self, pose: TransformationMatrix, qnear: Optional[List[float]] = None
    ) -> List[float]:
        self._tick()
        return self._joints_for(pose)

    def is_pose_within_safety_limits(self, pose: TransformationMatrix) -> bool:
        self._tick()
        return bool(float(pose.translation[2]) >= self.floor_z_mm)

    def is_joints_within_safety_limits(self, joint_positions: Sequence[float]) -> bool:
        self._tick()
        return bool(all(abs(float(angle)) <= self.joint_limit_rad for angle in joint_positions))


def _pose(x: float, y: float, z: float) -> TransformationMatrix:
    return TransformationMatrix(translation=np.array([x, y, z], dtype=np.float32))


def _path(start: Tuple[float, float, float], end: Tuple[float, float, float], count: int):
    """`count` evenly spaced poses from start to end, identity rotation throughout."""
    return [
        _pose(*(np.array(start, dtype=float) + (np.array(end, dtype=float) - np.array(start, dtype=float)) * t))
        for t in np.linspace(0.0, 1.0, count)
    ]


def _report(passed: bool, description: str, detail: str = "") -> bool:
    print(f"  [{'PASS' if passed else 'FAIL'}] {description}{('  - ' + detail) if detail else ''}")
    return passed


def _main() -> None:
    checks = []

    print("1. a clean path passes, and every step of the sequence gets checked")
    waypoints = _path((300.0, 0.0, 300.0), (400.0, 0.0, 300.0), 21)
    home_joints = [0.0, -1.0, 1.0, 0.0, 1.5, 0.0]
    report = check_path_reachability(
        FakeRobot(),
        waypoints,
        approach_pose=_pose(300.0, 0.0, 330.0),
        retreat_pose=_pose(400.0, 0.0, 330.0),
        home_joints=home_joints,
    )
    checks.append(_report(report.ok, "report.ok", report.summary()))
    checks.append(_report(not report.warnings, "no warnings", f"{len(report.warnings)} found"))
    # home + approach + 21 waypoints + retreat + home back = 25
    expected_steps = 2 + len(waypoints) + 2
    checks.append(
        _report(len(report.checks) == expected_steps, f"{expected_steps} steps checked", f"got {len(report.checks)}")
    )
    labels = [check.label for check in report.checks]
    checks.append(
        _report(
            labels[:2] == ["Home", "Approach"] and labels[-2:] == ["Retreat", "Home (return)"],
            "sequence order matches execution",
            f"{labels[:2]} ... {labels[-2:]}",
        )
    )
    motions = [check.motion for check in report.checks]
    checks.append(
        _report(
            motions.count(MOTION_LINEAR) == len(waypoints) and motions.count(MOTION_JOINT) == 4,
            "waypoints are linear legs, approach/retreat/home are joint legs",
            f"{motions.count(MOTION_LINEAR)} linear / {motions.count(MOTION_JOINT)} joint",
        )
    )

    print("\n2. without a home posture the two home steps are dropped")
    report = check_path_reachability(
        FakeRobot(),
        waypoints,
        approach_pose=_pose(300.0, 0.0, 330.0),
        retreat_pose=_pose(400.0, 0.0, 330.0),
        home_joints=None,
    )
    checks.append(
        _report(len(report.checks) == len(waypoints) + 2, "approach + waypoints + retreat only", report.summary())
    )
    checks.append(_report(report.ok, "still ok", report.summary()))

    print("\n3. out-of-reach waypoints are reported, by index")
    # Runs from inside the 1000 mm sphere to outside it; the far end is unreachable.
    waypoints = _path((900.0, 0.0, 100.0), (1100.0, 0.0, 100.0), 21)
    expected_unreachable = [
        index
        for index, waypoint in enumerate(waypoints)
        if float(np.linalg.norm(np.asarray(waypoint.translation, dtype=float))) > MAX_REACH_MM
    ]
    report = check_path_reachability(
        FakeRobot(),
        waypoints,
        approach_pose=_pose(900.0, 0.0, 130.0),
        retreat_pose=_pose(500.0, 0.0, 130.0),
    )
    checks.append(_report(not report.ok, "report is not ok", report.summary()))
    checks.append(
        _report(
            report.failed_waypoint_indices() == expected_unreachable,
            "exactly the out-of-reach waypoints are flagged",
            f"got {report.failed_waypoint_indices()}, expected {expected_unreachable}",
        )
    )
    checks.append(
        _report(
            all("out of reach" in check.reason() for check in report.failures),
            "reason names the cause",
            report.failures[0].reason() if report.failures else "",
        )
    )
    # An unreachable pose has no joints to judge, so it must not also be blamed for
    # violating joint limits - one cause, one message.
    checks.append(
        _report(
            all(check.joints_within_safety_limits for check in report.failures),
            "unreachable poses are not double-reported as joint-limit violations",
        )
    )

    print("\n4. a waypoint below the safety plane is reported separately from reach")
    waypoints = _path((300.0, 0.0, 100.0), (400.0, 0.0, -100.0), 21)
    expected_below = [
        index for index, waypoint in enumerate(waypoints) if float(waypoint.translation[2]) < FLOOR_Z_MM
    ]
    report = check_path_reachability(
        FakeRobot(),
        waypoints,
        approach_pose=_pose(300.0, 0.0, 130.0),
        retreat_pose=_pose(400.0, 0.0, 130.0),
    )
    checks.append(
        _report(
            report.failed_waypoint_indices() == expected_below,
            "exactly the below-floor waypoints are flagged",
            f"got {report.failed_waypoint_indices()}, expected {expected_below}",
        )
    )
    checks.append(
        _report(
            all("safety plane" in check.reason() for check in report.failures),
            "reason names the safety plane, not reach",
            report.failures[0].reason() if report.failures else "",
        )
    )

    print("\n5. an arm configuration flip on a linear leg warns but does not fail")
    waypoints = _path((300.0, 0.0, 300.0), (400.0, 0.0, 300.0), 21)
    flip_start = 350.0
    robot = FakeRobot(flip_x_range=(flip_start, 1e9))
    report = check_path_reachability(
        robot,
        waypoints,
        approach_pose=_pose(300.0, 0.0, 330.0),
        retreat_pose=_pose(400.0, 0.0, 330.0),
    )
    first_flipped = next(
        index for index, waypoint in enumerate(waypoints) if float(waypoint.translation[0]) >= flip_start
    )
    checks.append(_report(not report.failures, "no failures - the poses are all reachable"))
    checks.append(
        _report(
            report.warned_waypoint_indices() == [first_flipped],
            "warning lands on the waypoint where the configuration changes",
            f"got {report.warned_waypoint_indices()}, expected [{first_flipped}]",
        )
    )
    checks.append(
        _report(report.ok, "ok stays true - a warning is not a refusal", f"ok={report.ok}"),
    )
    warned = report.warnings[0] if report.warnings else None
    checks.append(
        _report(
            warned is not None and warned.joint_step_deg is not None
            and abs(warned.joint_step_deg - FLIP_OFFSET_DEG) < 1e-6,
            f"reported step is the {FLIP_OFFSET_DEG:.0f} deg flip",
            f"got {warned.joint_step_deg:.3f} deg" if warned and warned.joint_step_deg else "none",
        )
    )

    print("\n6. the same flip on a joint leg (approach) is not warned about")
    report = check_path_reachability(
        FakeRobot(flip_x_range=(-1e9, 1e9)),  # every pose flipped -> approach jumps from the seed
        waypoints,
        approach_pose=_pose(300.0, 0.0, 330.0),
        retreat_pose=_pose(400.0, 0.0, 330.0),
    )
    checks.append(
        _report(
            not report.warnings,
            "moveJ legs may change configuration freely",
            f"{len(report.warnings)} warning(s)",
        )
    )

    print("\n7. an unreachable point does not cascade into warnings behind it")
    # The seed must stay on the last *solved* configuration, otherwise every waypoint after
    # a gap gets compared against a stale one and the report fills with noise.
    waypoints = _path((300.0, 0.0, 300.0), (400.0, 0.0, 300.0), 21)
    waypoints[10] = _pose(5000.0, 0.0, 300.0)  # far out of reach, in the middle of the path
    report = check_path_reachability(
        FakeRobot(),
        waypoints,
        approach_pose=_pose(300.0, 0.0, 330.0),
        retreat_pose=_pose(400.0, 0.0, 330.0),
    )
    checks.append(
        _report(
            report.failed_waypoint_indices() == [10],
            "only the unreachable waypoint fails",
            f"got {report.failed_waypoint_indices()}",
        )
    )
    checks.append(
        _report(not report.warnings, "and nothing behind it warns", f"{len(report.warnings)} warning(s)")
    )

    print("\n8. a home posture outside the joint limits fails, with no waypoint index")
    report = check_path_reachability(
        FakeRobot(joint_limit_rad=0.5),
        _path((300.0, 0.0, 300.0), (400.0, 0.0, 300.0), 21),
        approach_pose=_pose(300.0, 0.0, 330.0),
        retreat_pose=_pose(400.0, 0.0, 330.0),
        home_joints=[0.0, -1.0, 1.0, 0.0, 1.5, 0.0],
    )
    home_failures = [check for check in report.failures if check.label.startswith("Home")]
    checks.append(_report(len(home_failures) == 2, "both home steps fail", f"got {len(home_failures)}"))
    checks.append(
        _report(
            all(check.waypoint_index is None for check in home_failures),
            "home steps carry no waypoint index (nothing to highlight in 3D)",
        )
    )
    checks.append(
        _report(
            all("joint limits" in check.reason() for check in home_failures),
            "reason names the joint limits",
            home_failures[0].reason() if home_failures else "",
        )
    )

    print("\n9. a query blowing up mid-check is reported, not raised")
    report = check_path_reachability(
        FakeRobot(raise_on_query=RuntimeError("RTDE interface not connected.")),
        _path((300.0, 0.0, 300.0), (400.0, 0.0, 300.0), 21),
        approach_pose=_pose(300.0, 0.0, 330.0),
        retreat_pose=_pose(400.0, 0.0, 330.0),
    )
    checks.append(_report(report.error is not None, "error is captured on the report", str(report.error)))
    checks.append(_report(not report.ok, "and the report is not ok"))
    checks.append(_report("did not complete" in report.summary(), "summary says so", report.summary()))

    print("\n10. too few waypoints is refused up front")
    report = check_path_reachability(
        FakeRobot(), [_pose(300.0, 0.0, 300.0)], approach_pose=_pose(300.0, 0.0, 330.0),
        retreat_pose=_pose(300.0, 0.0, 330.0),
    )
    checks.append(_report(report.error is not None and not report.checks, "no checks run", str(report.error)))

    print("\n11. the details listing is bounded")
    waypoints = _path((1100.0, 0.0, 100.0), (1500.0, 0.0, 100.0), 60)  # all out of reach
    report = check_path_reachability(
        FakeRobot(), waypoints, approach_pose=_pose(1100.0, 0.0, 130.0), retreat_pose=_pose(1500.0, 0.0, 130.0)
    )
    detail_lines = report.details().splitlines()
    checks.append(
        _report(
            len(detail_lines) <= 13 and detail_lines[-1].startswith("...and"),
            "long failure lists are truncated with a count",
            f"{len(detail_lines)} lines, last: {detail_lines[-1] if detail_lines else ''}",
        )
    )

    print(f"\nJoint-step warning threshold in use: {DEFAULT_JOINT_STEP_WARNING_DEG:.0f} deg")
    print(f"{'ALL CHECKS PASSED' if all(checks) else 'FAILURES ABOVE'} ({sum(checks)}/{len(checks)})")
    if not all(checks):
        raise SystemExit(1)


if __name__ == "__main__":
    _main()
