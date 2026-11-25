"""
Offline checks for robot.execution - no robot, no camera, no Zivid runtime.

What is tested is the part of execution that decides *whether to issue the next move*:

- Stop pressed mid-move, between moves, or in the instant a move is being issued: no
  further move (no retreat, no next line, no home) may be issued afterwards, and the
  outcome is ExecutionStopped, not completion.
- The arm coming to rest because of a protective stop, an emergency stop or the robot
  leaving running mode: reported as a failure, and nothing further is issued.
- RobotControlURRTDE.motion_health_problem against fake RTDE interfaces, so the real
  verdicts (including "reduced safety mode is still healthy") are checked, not a copy.

The stand-in robot is a timeline: each issued move "runs" for a fixed number of
is_moving() polls. Hooks inject a Stop press or a safety stop at a chosen point in it.
What is NOT tested: how quickly the real controller decelerates on stopL, and what the
real RTDE flags read in the instant after a protective stop - only the robot can say.

Run:
    python -m line_tracing_gui._dev_test_execution

"""

import threading
from typing import Callable, List, Optional

import numpy as np
from zividsamples.transformation_matrix import TransformationMatrix

from line_tracing_gui.robot.execution import ExecutionStopped, run_execution_sequence
from line_tracing_gui.robot.robot_control_ur_rtde import RobotControlURRTDE

POLLS_PER_MOVE = 4
HOME_JOINTS = [0.0, -1.57, 1.57, -1.57, -1.57, 0.0]


class FakeTarget:
    def __init__(self, pose: TransformationMatrix) -> None:
        self.pose = pose


class FakeRobot:
    """Records every move issued. Each move keeps is_moving() True for POLLS_PER_MOVE polls,
    unless stopped (stop() or an injected safety stop) - then the arm is at rest at once.

    `on_poll(robot, move_index, poll_index)` runs on every is_moving() call while a move is
    in flight; `on_issue(robot, move_index)` runs inside the move call itself.
    """

    def __init__(
        self,
        on_poll: Optional[Callable[["FakeRobot", int, int], None]] = None,
        on_issue: Optional[Callable[["FakeRobot", int], None]] = None,
    ) -> None:
        self.on_poll = on_poll
        self.on_issue = on_issue
        self.moves: List[str] = []
        self.stop_calls = 0
        self.health_problem: Optional[str] = None
        self._remaining_polls = 0
        self._polls_in_move = 0

    def _issue(self, name: str, kwargs: dict) -> None:
        assert kwargs.get("asynchronous") is True, f"{name} was not issued asynchronously"
        self.moves.append(name)
        self._remaining_polls = POLLS_PER_MOVE
        self._polls_in_move = 0
        if self.on_issue is not None:
            self.on_issue(self, len(self.moves) - 1)

    def move_to_joints(self, joints, **kwargs) -> None:
        self._issue("home", kwargs)

    def move_j(self, target, **kwargs) -> None:
        self._issue(f"move_j z={float(target.pose.translation[2]):.0f}", kwargs)

    def move_path(self, waypoints, **kwargs) -> None:
        self._issue(f"move_path x0={float(waypoints[0].translation[0]):.0f}", kwargs)

    def get_custom_target(self, pose):
        return FakeTarget(pose)

    def is_moving(self) -> bool:
        if self._remaining_polls > 0 and self.on_poll is not None:
            self.on_poll(self, len(self.moves) - 1, self._polls_in_move)
        self._polls_in_move += 1
        if self._remaining_polls > 0:
            self._remaining_polls -= 1
        return self._remaining_polls > 0

    def halt(self) -> None:
        self._remaining_polls = 0

    def stop(self) -> None:
        self.stop_calls += 1
        self.halt()

    def get_pose(self) -> FakeTarget:
        return FakeTarget(TransformationMatrix())

    def get_joint_positions(self) -> List[float]:
        return list(HOME_JOINTS)

    def motion_health_problem(self) -> Optional[str]:
        return self.health_problem


def _pose(x: float, y: float, z: float) -> TransformationMatrix:
    return TransformationMatrix(translation=np.array([x, y, z], dtype=np.float32))


# Two lines. Approach/retreat sit at z=10, the lines at z=0; paths are told apart by x.
PATHS = [[_pose(100, 0, 0), _pose(110, 0, 0)], [_pose(200, 0, 0), _pose(210, 0, 0)]]
APPROACHES = [_pose(100, 0, 10), _pose(200, 0, 10)]
RETREATS = [_pose(110, 0, 10), _pose(210, 0, 10)]
FULL_SEQUENCE = [
    "home",
    "move_j z=10",
    "move_path x0=100",
    "move_j z=10",
    "move_j z=10",
    "move_path x0=200",
    "move_j z=10",
    "home",
]


def _run(robot: FakeRobot, abort_event: threading.Event, home_joints=HOME_JOINTS):
    """Returns ("done", message) / ("stopped", message) / ("failed", message)."""
    try:
        message = run_execution_sequence(
            robot,
            PATHS,
            APPROACHES,
            RETREATS,
            speed_m_s=0.03,
            acceleration=0.3,
            blend_m=0.001,
            home_joints=home_joints,
            joint_speed=0.2,
            joint_acceleration=0.3,
            abort_event=abort_event,
            poll_interval_s=0.0,
        )
        return "done", message
    except ExecutionStopped as ex:
        return "stopped", str(ex)
    except RuntimeError as ex:
        return "failed", str(ex)


def _press_stop(abort_event: threading.Event, robot: FakeRobot) -> None:
    # What TracePanel.on_stop_clicked does: flag first, then stop.
    abort_event.set()
    robot.stop()


def _report(passed: bool, description: str, detail: str = "") -> bool:
    print(f"  [{'PASS' if passed else 'FAIL'}] {description}{('  - ' + detail) if detail else ''}")
    return passed


class FakeRtdeControl:
    def __init__(self, connected=True, program_running=True) -> None:
        self.connected = connected
        self.program_running = program_running

    def isConnected(self) -> bool:  # noqa: N802 - mirrors ur_rtde
        return self.connected

    def isProgramRunning(self) -> bool:  # noqa: N802 - mirrors ur_rtde
        return self.program_running


class FakeRtdeReceive:
    def __init__(self, e_stopped=False, protective=False, robot_mode=7, safety_mode=1, raise_on_read=False) -> None:
        self.e_stopped = e_stopped
        self.protective = protective
        self.robot_mode = robot_mode
        self.safety_mode = safety_mode
        self.raise_on_read = raise_on_read

    def isConnected(self) -> bool:  # noqa: N802 - mirrors ur_rtde
        return True

    def isEmergencyStopped(self) -> bool:  # noqa: N802 - mirrors ur_rtde
        if self.raise_on_read:
            raise RuntimeError("socket closed")
        return self.e_stopped

    def isProtectiveStopped(self) -> bool:  # noqa: N802 - mirrors ur_rtde
        return self.protective

    def getRobotMode(self) -> int:  # noqa: N802 - mirrors ur_rtde
        return self.robot_mode

    def getSafetyMode(self) -> int:  # noqa: N802 - mirrors ur_rtde
        return self.safety_mode


def _health(control: FakeRtdeControl, receive: FakeRtdeReceive) -> Optional[str]:
    # No connect(): the interfaces are set by hand, so no socket is ever opened.
    robot = RobotControlURRTDE("0.0.0.0")
    robot.rtde_control = control
    robot.rtde_receive = receive
    return robot.motion_health_problem()


def _main() -> None:
    checks: List[bool] = []

    print("1. no stop, healthy robot: the whole sequence runs")
    robot = FakeRobot()
    outcome, message = _run(robot, threading.Event())
    checks.append(_report(outcome == "done", "completes", message))
    checks.append(_report(robot.moves == FULL_SEQUENCE, "every leg issued in order", str(robot.moves)))

    print("\n2. Stop pressed while the first line is being traced")
    abort = threading.Event()

    def _stop_mid_trace(r: FakeRobot, move_index: int, poll_index: int) -> None:
        if move_index == 2 and poll_index == 1:
            _press_stop(abort, r)

    robot = FakeRobot(on_poll=_stop_mid_trace)
    outcome, message = _run(robot, abort)
    checks.append(_report(outcome == "stopped", "reported as stopped, not done", message))
    checks.append(_report(message == "Stopped by user", "message is 'Stopped by user'", message))
    checks.append(
        _report(robot.moves == FULL_SEQUENCE[:3], "no retreat, no next line, no home", str(robot.moves))
    )

    print("\n3. Stop pressed during the approach of the second line")
    abort = threading.Event()

    def _stop_second_approach(r: FakeRobot, move_index: int, poll_index: int) -> None:
        if move_index == 4 and poll_index == 0:
            _press_stop(abort, r)

    robot = FakeRobot(on_poll=_stop_second_approach)
    outcome, _ = _run(robot, abort)
    checks.append(_report(outcome == "stopped", "reported as stopped"))
    checks.append(_report(robot.moves == FULL_SEQUENCE[:5], "second line never traced", str(robot.moves)))

    print("\n4. Stop pressed while the move call itself is in progress (race window)")
    abort = threading.Event()

    def _stop_while_issuing(r: FakeRobot, move_index: int) -> None:
        if move_index == 1:
            abort.set()  # flag set, but the stop command has not reached the controller yet

    robot = FakeRobot(on_issue=_stop_while_issuing)
    outcome, _ = _run(robot, abort)
    checks.append(_report(outcome == "stopped", "reported as stopped"))
    checks.append(_report(robot.stop_calls >= 1, "stop re-sent after the move was issued", f"{robot.stop_calls} call(s)"))
    checks.append(_report(robot.moves == FULL_SEQUENCE[:2], "nothing issued after it", str(robot.moves)))

    print("\n5. Stop already pressed before the sequence starts")
    abort = threading.Event()
    abort.set()
    robot = FakeRobot()
    outcome, _ = _run(robot, abort)
    checks.append(_report(outcome == "stopped" and robot.moves == [], "no move issued at all", str(robot.moves)))

    print("\n6. protective stop during the second line")

    def _protective_stop(r: FakeRobot, move_index: int, poll_index: int) -> None:
        if move_index == 5 and poll_index == 1:
            r.health_problem = "the robot is protective stopped - clear it on the pendant before retrying"
            r.halt()

    robot = FakeRobot(on_poll=_protective_stop)
    outcome, message = _run(robot, threading.Event())
    checks.append(_report(outcome == "failed", "reported as a failure, not completion", message))
    checks.append(_report("protective" in message and "move_path" in message, "names the move and the cause"))
    checks.append(_report(robot.moves == FULL_SEQUENCE[:6], "no retreat or home after it", str(robot.moves)))

    print("\n7. emergency stop on the way home")

    def _e_stop(r: FakeRobot, move_index: int, poll_index: int) -> None:
        if move_index == 7 and poll_index == 0:
            r.health_problem = "the robot is emergency stopped"
            r.halt()

    robot = FakeRobot(on_poll=_e_stop)
    outcome, message = _run(robot, threading.Event())
    checks.append(_report(outcome == "failed" and "emergency" in message, "reported as a failure", message))

    print("\n8. RobotControlURRTDE.motion_health_problem against fake RTDE interfaces")
    cases = [
        ("healthy (running, normal)", FakeRtdeControl(), FakeRtdeReceive(), None),
        ("reduced safety mode still counts as healthy", FakeRtdeControl(), FakeRtdeReceive(safety_mode=2), None),
        ("protective stop", FakeRtdeControl(), FakeRtdeReceive(protective=True, safety_mode=3), "protective"),
        ("emergency stop", FakeRtdeControl(), FakeRtdeReceive(e_stopped=True, safety_mode=7), "emergency"),
        ("robot not in running mode", FakeRtdeControl(), FakeRtdeReceive(robot_mode=5), "not running"),
        ("safeguard stop (safety mode 5)", FakeRtdeControl(), FakeRtdeReceive(safety_mode=5), "safeguard"),
        ("control script died", FakeRtdeControl(program_running=False), FakeRtdeReceive(), "control script"),
        ("RTDE dropped", FakeRtdeControl(connected=False), FakeRtdeReceive(), "dropped"),
        ("state unreadable counts as a problem", FakeRtdeControl(), FakeRtdeReceive(raise_on_read=True), "could not"),
    ]
    for description, control, receive, expected in cases:
        problem = _health(control, receive)
        passed = problem is None if expected is None else (problem is not None and expected in problem)
        checks.append(_report(passed, description, str(problem)))

    print(f"\n{'ALL CHECKS PASSED' if all(checks) else 'FAILURES ABOVE'} ({sum(checks)}/{len(checks)})")
    if not all(checks):
        raise SystemExit(1)


if __name__ == "__main__":
    _main()
