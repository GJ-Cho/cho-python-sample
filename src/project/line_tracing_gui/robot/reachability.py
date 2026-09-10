"""
Pre-flight reachability check for the line tracing execution sequence.

Starting move_path and only then discovering that a waypoint is unreachable
leaves the robot standing halfway along the path with a sharp tip near the
surface - that failure mode is what this module exists to avoid. It replays the
exact sequence TracePanel executes ([home] -> approach -> waypoints -> retreat
-> [home]) against the controller's own kinematics and safety model, without
moving anything, and reports per-waypoint verdicts.

What is asked of the controller, per pose:

- ``has_inverse_kinematics_solution`` - the same call family move_j uses, so a
  pose that passes here is one move_j can actually solve.
- ``is_pose_within_safety_limits`` - the safety planes configured on the
  pendant. Independent of reachability: a pose can be perfectly solvable and
  still sit outside a configured plane.
- ``is_joints_within_safety_limits`` on the resulting joint solution - the
  configured joint ranges, which the Cartesian checks do not cover.

**qnear chaining.** Every pose is solved with the *previous* step's joint
solution as ``qnear``, mirroring how the robot actually traverses the path.
Solving each pose against the robot's current joint positions instead would let
IK pick a different arm configuration per waypoint and hide exactly the branch
flips this check should catch.

**Linear vs joint legs.** Approach, retreat and home run as moveJ, where the
controller is free to choose any joint path, so a large joint change there is
normal and not reported. The waypoints run as one blended movePath of MoveL
entries - and since movePath starts from wherever the robot already is, the
approach -> first waypoint leg is a MoveL too. Along that linear run the TCP is
constrained to a straight line, so a large joint change between neighbours means
IK flipped to a different arm configuration; the controller will then either
refuse the segment or swing through it violently. That is reported as a warning
rather than a failure, because the threshold is a heuristic and a genuinely
tight-but-valid path can trip it.

Everything resolves against the controller's currently active TCP, which is also
the frame the waypoints are expressed in (the TCP offset lives on the
controller, not in this code - see robot/README.md), so nothing is converted
here.

The robot is reached only through the five query methods listed in
`ReachabilityQueryable`, so the sequencing logic can be exercised offline
against a stand-in - see `_dev_test_reachability.py`.

"""

from dataclasses import dataclass, field
from typing import Callable, List, Optional, Sequence, Tuple

import numpy as np
from zividsamples.transformation_matrix import TransformationMatrix

# A linear segment between waypoints ~5 mm apart should move every joint by well under a
# degree. Tens of degrees means IK resolved the two ends into different arm configurations.
# Deliberately loose: this only raises a warning, and a false alarm that makes someone look
# at the 3D view costs far less than a missed configuration flip mid-trace.
DEFAULT_JOINT_STEP_WARNING_DEG = 30.0

MOTION_JOINT = "joint"  # moveJ / move_to_joints - the controller picks the joint path
MOTION_LINEAR = "linear"  # moveL / movePath entry - TCP constrained to a straight line


@dataclass
class StepCheck:
    """Verdict for one step of the execution sequence."""

    label: str  # e.g. "Home", "Approach", "Waypoint 12/87", "Retreat"
    motion: str  # MOTION_JOINT or MOTION_LINEAR
    waypoint_index: Optional[int] = None  # index into the caller's waypoint list, if it is one
    has_ik_solution: bool = True  # joints-only steps (home) need no IK
    pose_within_safety_limits: bool = True
    joints_within_safety_limits: bool = True
    joints: Optional[List[float]] = None  # solved (or given) joint angles, rad
    joint_step_deg: Optional[float] = None  # largest per-joint change from the previous step

    @property
    def failed(self) -> bool:
        return not (self.has_ik_solution and self.pose_within_safety_limits and self.joints_within_safety_limits)

    def warned(self, joint_step_warning_deg: float = DEFAULT_JOINT_STEP_WARNING_DEG) -> bool:
        return (
            not self.failed
            and self.motion == MOTION_LINEAR
            and self.joint_step_deg is not None
            and self.joint_step_deg > joint_step_warning_deg
        )

    def reason(self, joint_step_warning_deg: float = DEFAULT_JOINT_STEP_WARNING_DEG) -> str:
        """Why this step failed or warned - an empty string when it is simply fine."""
        problems = []
        if not self.has_ik_solution:
            problems.append("out of reach (no IK solution)")
        if not self.pose_within_safety_limits:
            problems.append("outside a configured safety plane")
        if not self.joints_within_safety_limits:
            problems.append("needs joint angles outside the configured joint limits")
        if problems:
            return ", ".join(problems)
        if self.warned(joint_step_warning_deg):
            assert self.joint_step_deg is not None
            return (
                f"arm configuration change of {self.joint_step_deg:.0f} deg on a straight-line "
                "segment - the robot may refuse it or swing through it"
            )
        return ""


@dataclass
class ReachabilityReport:
    checks: List[StepCheck] = field(default_factory=list)
    joint_step_warning_deg: float = DEFAULT_JOINT_STEP_WARNING_DEG
    # Set when the check could not be completed at all (e.g. the connection dropped
    # mid-check). A report with this set makes no claim about the path.
    error: Optional[str] = None

    @property
    def failures(self) -> List[StepCheck]:
        return [check for check in self.checks if check.failed]

    @property
    def warnings(self) -> List[StepCheck]:
        return [check for check in self.checks if check.warned(self.joint_step_warning_deg)]

    @property
    def ok(self) -> bool:
        return self.error is None and not self.failures

    def failed_waypoint_indices(self) -> List[int]:
        """Indices into the caller's waypoint list, for highlighting in the 3D view.
        Approach/retreat/home carry no waypoint index and are left out."""
        return [check.waypoint_index for check in self.failures if check.waypoint_index is not None]

    def warned_waypoint_indices(self) -> List[int]:
        return [check.waypoint_index for check in self.warnings if check.waypoint_index is not None]

    def summary(self) -> str:
        if self.error is not None:
            return f"Reachability check did not complete: {self.error}"
        if not self.checks:
            return "Nothing to check."
        if self.ok and not self.warnings:
            return f"All {len(self.checks)} steps reachable."
        parts = []
        if self.failures:
            parts.append(f"{len(self.failures)} unreachable")
        if self.warnings:
            parts.append(f"{len(self.warnings)} warning(s)")
        return ", ".join(parts) + f" out of {len(self.checks)} steps."

    def details(self, max_lines: int = 12) -> str:
        """Failures first, then warnings, one line each - the body of the dialog shown to
        the user. Truncated, because a path whose whole middle is out of reach would
        otherwise produce a dialog hundreds of lines tall."""
        lines = [f"- {check.label}: {check.reason(self.joint_step_warning_deg)}" for check in self.failures]
        lines += [f"- {check.label}: {check.reason(self.joint_step_warning_deg)}" for check in self.warnings]
        if len(lines) > max_lines:
            hidden = len(lines) - max_lines
            lines = lines[:max_lines] + [f"...and {hidden} more."]
        return "\n".join(lines)


def _max_joint_step_deg(previous: Optional[Sequence[float]], current: Optional[Sequence[float]]) -> Optional[float]:
    if previous is None or current is None or len(previous) != len(current):
        return None
    return float(np.max(np.abs(np.degrees(np.asarray(current, dtype=float) - np.asarray(previous, dtype=float)))))


def check_path_reachability(
    robot_control,
    waypoints: List[TransformationMatrix],
    approach_pose: TransformationMatrix,
    retreat_pose: TransformationMatrix,
    home_joints: Optional[List[float]] = None,
    joint_step_warning_deg: float = DEFAULT_JOINT_STEP_WARNING_DEG,
    progress_callback: Optional[Callable[[int, int], None]] = None,
) -> ReachabilityReport:
    """Vet the whole execution sequence without moving the robot.

    The sequence checked mirrors TracePanel's execution exactly: an optional home leg,
    approach (moveJ), the waypoints (movePath, so linear from the approach pose onward),
    retreat (moveJ), and the optional return home. Keep the two in step - a check that
    models a different sequence is worse than no check, because it invites trust it has
    not earned.

    Args:
        robot_control: Connected robot. Only these five methods are used, so a stand-in
            can be passed for offline testing: get_joint_positions,
            has_inverse_kinematics_solution, get_inverse_kinematics,
            is_pose_within_safety_limits, is_joints_within_safety_limits.
        waypoints: The traced path, base frame, in order.
        approach_pose: Standoff pose the robot moves to before the trace.
        retreat_pose: Standoff pose the robot moves to after the trace.
        home_joints: Home/start posture as joint angles (rad), or None if none is set.
        joint_step_warning_deg: Per-joint change along a linear segment above which an arm
            configuration flip is suspected.
        progress_callback: Called as (completed, total) so a caller can show progress -
            each pose costs three RTDE round-trips, so a long path takes a second or two.

    Returns:
        A ReachabilityReport. Query failures (e.g. the connection dropping mid-check) are
        captured in `report.error` rather than raised, so a caller on a worker thread can
        surface them the same way as a negative verdict.
    """
    report = ReachabilityReport(joint_step_warning_deg=joint_step_warning_deg)
    if len(waypoints) < 2:
        report.error = "There are fewer than 2 waypoints to check."
        return report

    # Steps to check, in execution order: (label, pose or None for a joints-only step,
    # motion kind, waypoint index or None).
    steps: List[Tuple[str, Optional[TransformationMatrix], str, Optional[int]]] = []
    if home_joints is not None:
        steps.append(("Home", None, MOTION_JOINT, None))
    steps.append(("Approach", approach_pose, MOTION_JOINT, None))
    total = len(waypoints)
    for index, waypoint in enumerate(waypoints):
        steps.append((f"Waypoint {index + 1}/{total}", waypoint, MOTION_LINEAR, index))
    steps.append(("Retreat", retreat_pose, MOTION_JOINT, None))
    if home_joints is not None:
        steps.append(("Home (return)", None, MOTION_JOINT, None))

    try:
        # Seed qnear with where the robot will actually stand when the sequence starts.
        previous_joints: Optional[List[float]] = (
            list(home_joints) if home_joints is not None else list(robot_control.get_joint_positions())
        )

        for completed, (label, pose, motion, waypoint_index) in enumerate(steps):
            if pose is None:  # home: moved to by joint angles, so there is no IK to solve
                assert home_joints is not None
                home = list(home_joints)
                check = StepCheck(
                    label=label,
                    motion=motion,
                    joints=home,
                    joints_within_safety_limits=robot_control.is_joints_within_safety_limits(home),
                    joint_step_deg=_max_joint_step_deg(previous_joints, home),
                )
                previous_joints = home
            else:
                has_solution = bool(robot_control.has_inverse_kinematics_solution(pose, qnear=previous_joints))
                joints = (
                    list(robot_control.get_inverse_kinematics(pose, qnear=previous_joints)) if has_solution else None
                )
                check = StepCheck(
                    label=label,
                    motion=motion,
                    waypoint_index=waypoint_index,
                    has_ik_solution=has_solution,
                    pose_within_safety_limits=bool(robot_control.is_pose_within_safety_limits(pose)),
                    # An unreachable pose has no joint solution to judge; don't report a
                    # second failure spun out of the same cause.
                    joints_within_safety_limits=(
                        bool(robot_control.is_joints_within_safety_limits(joints)) if joints is not None else True
                    ),
                    joints=joints,
                    joint_step_deg=_max_joint_step_deg(previous_joints, joints),
                )
                # Keep the last good solution as the seed: chaining from a pose that could
                # not be solved would compare every later waypoint against a stale
                # configuration, turning one unreachable point into a cascade of warnings.
                if joints is not None:
                    previous_joints = joints

            report.checks.append(check)
            if progress_callback is not None:
                progress_callback(completed + 1, len(steps))
    except Exception as ex:  # pylint: disable=broad-except
        report.error = str(ex)

    return report
