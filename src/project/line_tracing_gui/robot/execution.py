"""
Execution sequence for line tracing: [home] -> per path (approach -> trace -> retreat) -> [home].

Kept free of Qt and of the zivid module on purpose, so the stop/failure behaviour can be
exercised offline with a stub robot (see _dev_test_execution). The Line Tracing tab runs
this on a worker thread and only translates the outcome into GUI text.

Two rules this module enforces, both learned from what the robot actually does:

**Stop means stop.** Every move is issued asynchronously and then watched until the arm
comes to rest. A stopped arm looks exactly like a finished move from that point of view,
so without a flag the sequence would carry on to the retreat, the next line and home
right after the user pressed Stop. The caller sets `abort_event` and sends the stop;
this module checks the flag before issuing every move and again once the arm is at rest,
and raises ExecutionStopped instead of continuing.

**At rest is not the same as arrived.** A protective stop or an emergency stop also
brings the arm to rest. After every move the robot's own state is read back
(`robot_control.motion_health_problem()`), and anything other than a running robot in
normal/reduced safety mode is raised as a failure rather than reported as completion.

robot_control is duck-typed. Methods used: move_to_joints, move_j, move_path,
get_custom_target, is_moving, get_pose, get_joint_positions, stop,
motion_health_problem.
"""

import threading
import time
from typing import Callable, List, Optional, Sequence

from zividsamples.transformation_matrix import TransformationMatrix

STOPPED_BY_USER_MESSAGE = "Stopped by user"

# How often the worker samples the arm while a move is in progress.
DEFAULT_POLL_INTERVAL_S = 0.05


class ExecutionStopped(Exception):
    """Raised when the user pressed Stop. Not an error, but not a success either."""

    def __init__(self) -> None:
        super().__init__(STOPPED_BY_USER_MESSAGE)


def _is_set(abort_event: Optional[threading.Event]) -> bool:
    return abort_event is not None and abort_event.is_set()


def move_with_monitoring(
    robot_control,
    move_fn: Callable,
    *args,
    abort_event: Optional[threading.Event] = None,
    on_pose: Optional[Callable[[TransformationMatrix], None]] = None,
    on_joints: Optional[Callable[[List[float]], None]] = None,
    poll_interval_s: float = DEFAULT_POLL_INTERVAL_S,
    **kwargs,
) -> None:
    """Issue one move asynchronously and wait until the arm is at rest again.

    Raises ExecutionStopped if `abort_event` is set (before the move is issued, or by the
    time the arm comes to rest), and RuntimeError if the arm came to rest because the
    robot left its normal running state (protective stop, e-stop, ...).
    """
    if _is_set(abort_event):
        raise ExecutionStopped()

    kwargs["asynchronous"] = True
    move_fn(*args, **kwargs)
    # Stop may have been pressed between the check above and the move being issued; in
    # that window the stop reached the controller before the move did, so send it again.
    if _is_set(abort_event):
        robot_control.stop()

    time.sleep(poll_interval_s)
    while robot_control.is_moving():
        if on_pose is not None:
            on_pose(robot_control.get_pose().pose)
        time.sleep(poll_interval_s)
    if on_pose is not None:
        on_pose(robot_control.get_pose().pose)
    if on_joints is not None:
        on_joints(robot_control.get_joint_positions())

    if _is_set(abort_event):
        raise ExecutionStopped()
    problem = robot_control.motion_health_problem()
    if problem is not None:
        name = getattr(move_fn, "__name__", "move")
        raise RuntimeError(f"{name} did not complete: {problem}.")


def run_execution_sequence(
    robot_control,
    paths: Sequence[Sequence[TransformationMatrix]],
    approach_poses: Sequence[TransformationMatrix],
    retreat_poses: Sequence[TransformationMatrix],
    speed_m_s: float,
    acceleration: float,
    blend_m: float,
    home_joints: Optional[List[float]],
    joint_speed: float,
    joint_acceleration: float,
    abort_event: Optional[threading.Event] = None,
    on_pose: Optional[Callable[[TransformationMatrix], None]] = None,
    on_joints: Optional[Callable[[List[float]], None]] = None,
    poll_interval_s: float = DEFAULT_POLL_INTERVAL_S,
) -> str:
    """Run the whole sequence and return a completion message.

    Raises ExecutionStopped on Stop and RuntimeError on any failed or interrupted move;
    in both cases no further move is issued.
    """
    if robot_control is None:
        raise RuntimeError("Robot is not connected.")
    if not (len(paths) == len(approach_poses) == len(retreat_poses)):
        raise RuntimeError(
            f"Got {len(paths)} paths but {len(approach_poses)} approach and {len(retreat_poses)} retreat poses."
        )

    def _move(move_fn: Callable, *args, **kwargs) -> None:
        move_with_monitoring(
            robot_control,
            move_fn,
            *args,
            abort_event=abort_event,
            on_pose=on_pose,
            on_joints=on_joints,
            poll_interval_s=poll_interval_s,
            **kwargs,
        )

    if home_joints is not None:
        _move(robot_control.move_to_joints, home_joints, speed=joint_speed, acceleration=joint_acceleration)
    # Every path is approached and retreated on its own. Travelling from the end of one line
    # to the start of the next has to happen off the surface, or the tip drags across the part.
    for path, approach_pose, retreat_pose in zip(paths, approach_poses, retreat_poses):
        _move(
            robot_control.move_j,
            robot_control.get_custom_target(approach_pose),
            speed=joint_speed,
            acceleration=joint_acceleration,
        )
        _move(robot_control.move_path, list(path), speed=speed_m_s, acceleration=acceleration, blend_radius=blend_m)
        _move(
            robot_control.move_j,
            robot_control.get_custom_target(retreat_pose),
            speed=joint_speed,
            acceleration=joint_acceleration,
        )
    if home_joints is not None:
        _move(robot_control.move_to_joints, home_joints, speed=joint_speed, acceleration=joint_acceleration)

    traced = f"{len(paths)} lines" if len(paths) > 1 else "1 line"
    return f"Done - {traced}, {sum(len(path) for path in paths)} waypoints"
