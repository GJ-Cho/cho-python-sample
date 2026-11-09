"""
RobotControl implementation for UR3e using the official ur_rtde library
(rtde_control.RTDEControlInterface / rtde_receive.RTDEReceiveInterface).

No custom URScript needs to run on the teach pendant - the only prerequisite
is enabling Remote Control mode on the robot (e-Series: top-right icon on
the pendant -> select "Remote Control", or Installation -> General ->
Remote Control). RTDEControlInterface's constructor connects immediately
and will raise if Remote Control mode isn't enabled.

"""

from typing import List, Optional

import numpy as np
import rtde_control
import rtde_receive
from scipy.spatial.transform import Rotation
from zividsamples.gui.robot.robot_control import RobotControl, RobotTarget
from zividsamples.transformation_matrix import TransformationMatrix

# Conservative defaults for first tests with a pointed gripper. Override per-call
# once behavior has been verified at low speed.
DEFAULT_SPEED = 0.05  # m/s
DEFAULT_ACCELERATION = 0.2  # m/s^2
DEFAULT_BLEND_RADIUS = 0.0  # m
DEFAULT_JOINT_SPEED = 0.2  # rad/s
DEFAULT_JOINT_ACCELERATION = 0.3  # rad/s^2

IS_MOVING_SPEED_THRESHOLD = 1e-3  # m/s and rad/s, on the combined 6-vector norm

# Numeric robot/safety mode values reported by RTDE, from Universal Robots' RTDE guide
# ("Real-Time Data Exchange (RTDE) Guide", robot_mode / safety_mode fields). Only used to
# turn a bare integer into readable text in describe_motion_failure(); an unrecognized
# value degrades to "unknown (<n>)" rather than failing.
ROBOT_MODE_NAMES = {
    -1: "no controller",
    0: "disconnected",
    1: "confirm safety",
    2: "booting",
    3: "power off",
    4: "power on",
    5: "idle",
    6: "backdrive",
    7: "running",
    8: "updating firmware",
}
SAFETY_MODE_NAMES = {
    1: "normal",
    2: "reduced",
    3: "protective stop",
    4: "recovery",
    5: "safeguard stop",
    6: "system emergency stop",
    7: "robot emergency stop",
    8: "violation",
    9: "fault",
    10: "validate joint id",
    11: "undefined safety mode",
    12: "automatic mode safeguard stop",
    13: "system three position enabling stop",
}


class SimpleRobotConfiguration:
    """Minimal duck-typed stand-in for zividsamples' RobotConfiguration.

    RobotControl.__init__ only stores whatever is passed as robot_configuration;
    RobotControlURRTDE only ever reads `.ip_addr` off of it. Using this instead of
    zividsamples.gui.wizard.robot_configuration.RobotConfiguration avoids reading
    from / writing to that class's QSettings("Zivid", "HandEyeGUI") group, which
    belongs to a different application.
    """

    def __init__(self, ip_addr: str):
        self.ip_addr = ip_addr


def _pose_to_rtde(pose: TransformationMatrix) -> List[float]:
    """TransformationMatrix (mm) -> UR native pose [x, y, z, rx, ry, rz] (m, rotation vector)."""
    translation_m = (pose.translation / 1000.0).tolist()
    rotvec = pose.rotation.as_rotvec(degrees=False).tolist()
    return translation_m + rotvec


def _rtde_pose_to_transformation_matrix(rtde_pose: List[float]) -> TransformationMatrix:
    return TransformationMatrix(
        translation=np.array(rtde_pose[:3], dtype=np.float32) * 1000.0,
        rotation=Rotation.from_rotvec(rtde_pose[3:6], degrees=False),
    )


class RobotControlURRTDE(RobotControl):
    def __init__(self, ip_addr: str):
        super().__init__(SimpleRobotConfiguration(ip_addr))
        self.rtde_control: Optional[rtde_control.RTDEControlInterface] = None
        self.rtde_receive: Optional[rtde_receive.RTDEReceiveInterface] = None

    def connect(self) -> None:
        self.rtde_control = rtde_control.RTDEControlInterface(self.robot_configuration.ip_addr)
        self.rtde_receive = rtde_receive.RTDEReceiveInterface(self.robot_configuration.ip_addr)

    def disconnect(self) -> None:
        if self.rtde_control is not None:
            self.rtde_control.disconnect()
        if self.rtde_receive is not None:
            self.rtde_receive.disconnect()
        self.rtde_control = None
        self.rtde_receive = None

    def _require_connected(self) -> None:
        if self.rtde_control is None or self.rtde_receive is None:
            raise RuntimeError("RTDE interface not connected.")

    def get_pose(self) -> RobotTarget:
        self._require_connected()
        assert self.rtde_receive is not None
        return RobotTarget(
            name="Current TCP Pose",
            pose=_rtde_pose_to_transformation_matrix(self.rtde_receive.getActualTCPPose()),
        )

    def get_flange_pose(self) -> RobotTarget:
        """Pose of the tool flange (the 6th axis face) in base frame.

        get_pose() returns getActualTCPPose(), which already has the controller's TCP
        offset applied: base_T_tcp = base_T_flange * flange_T_tcp. Taking that offset back
        out gives the flange, which is what a hand-eye calibration performed against the
        flange pairs with - see CalibrationPanel's pose reference.
        """
        self._require_connected()
        return RobotTarget(
            name="Current Flange Pose",
            pose=self.get_pose().pose * self.get_tcp_offset().inv(),
        )

    def is_moving(self) -> bool:
        self._require_connected()
        assert self.rtde_receive is not None
        speed = np.array(self.rtde_receive.getActualTCPSpeed())
        return bool(np.linalg.norm(speed) > IS_MOVING_SPEED_THRESHOLD)

    def get_joint_positions(self) -> List[float]:
        """Current joint angles (rad), in UR's base-to-tool joint order [j1..j6]."""
        self._require_connected()
        assert self.rtde_receive is not None
        return list(self.rtde_receive.getActualQ())

    def get_tcp_offset(self) -> TransformationMatrix:
        """Active TCP offset (flange -> tool tip), as configured on the controller
        (PolyScope Installation -> TCP Configuration, or via set_tcp_offset below)."""
        self._require_connected()
        assert self.rtde_control is not None
        return _rtde_pose_to_transformation_matrix(self.rtde_control.getTCPOffset())

    def set_tcp_offset(self, tcp_offset: TransformationMatrix) -> None:
        self._require_connected()
        assert self.rtde_control is not None
        success = self.rtde_control.setTcp(_pose_to_rtde(tcp_offset))
        if not success:
            raise RuntimeError("setTcp failed.")

    # --- Kinematics / safety queries (no motion) ------------------------------------------
    #
    # Used by robot.reachability to vet a whole path before any of it is executed. All three
    # resolve against the controller's currently active TCP - the same frame the waypoints are
    # expressed in (the TCP offset lives on the controller, not in this code; see robot/README.md),
    # so no offset conversion is needed here.

    def get_inverse_kinematics(
        self, pose: TransformationMatrix, qnear: Optional[List[float]] = None
    ) -> List[float]:
        """Joint angles (rad) that put the TCP at `pose`. Prefer passing `qnear` (the joint
        positions the robot will actually be in when it goes there) - IK otherwise resolves
        against the *current* joint positions, which can pick a different arm configuration
        than the one the robot will really be in mid-path.

        Note this returns a solution even when the pose is out of reach; call
        has_inverse_kinematics_solution() first to find that out.
        """
        self._require_connected()
        assert self.rtde_control is not None
        return list(self.rtde_control.getInverseKinematics(_pose_to_rtde(pose), list(qnear) if qnear else []))

    def has_inverse_kinematics_solution(
        self, pose: TransformationMatrix, qnear: Optional[List[float]] = None
    ) -> bool:
        """Whether the arm can reach `pose` at all. Same call family as get_inverse_kinematics
        (and therefore as move_j), so a pose that passes here is one move_j can solve."""
        self._require_connected()
        assert self.rtde_control is not None
        return bool(
            self.rtde_control.getInverseKinematicsHasSolution(_pose_to_rtde(pose), list(qnear) if qnear else [])
        )

    def is_pose_within_safety_limits(self, pose: TransformationMatrix) -> bool:
        """Whether `pose` respects the safety planes/boundaries configured on the pendant.
        Independent of reachability - a pose can be perfectly solvable and still sit outside
        a configured safety plane."""
        self._require_connected()
        assert self.rtde_control is not None
        return bool(self.rtde_control.isPoseWithinSafetyLimits(_pose_to_rtde(pose)))

    def is_joints_within_safety_limits(self, joint_positions: List[float]) -> bool:
        """Whether the given joint angles (rad) respect the configured joint range limits."""
        self._require_connected()
        assert self.rtde_control is not None
        return bool(self.rtde_control.isJointsWithinSafetyLimits(list(joint_positions)))

    def is_emergency_stopped(self) -> bool:
        self._require_connected()
        assert self.rtde_receive is not None
        return bool(self.rtde_receive.isEmergencyStopped())

    def is_protective_stopped(self) -> bool:
        self._require_connected()
        assert self.rtde_receive is not None
        return bool(self.rtde_receive.isProtectiveStopped())

    def is_rtde_connected(self) -> bool:
        """Whether the underlying RTDE sockets report as connected. Does NOT mean the robot
        can actually move - after an e-stop/protective-stop, RTDE stays connected (pose
        readouts keep working) while move_l/move_path fail. Use is_emergency_stopped() /
        is_protective_stopped() to detect that.
        """
        return (
            self.rtde_control is not None
            and self.rtde_control.isConnected()
            and self.rtde_receive is not None
            and self.rtde_receive.isConnected()
        )

    def describe_motion_failure(self) -> str:
        """Why a move just returned False, as a short human-readable clause.

        moveL/moveJ/movePath only report a bare boolean, so without this every failure
        reads the same regardless of cause. The checks are ordered by how often they are
        the real cause: a dead RTDE control script (isProgramRunning() False) is the top
        candidate - it is what a `moveJ (to joints) failed` on an otherwise healthy robot
        has looked like here - followed by the stops, then the raw modes.

        Never raises: this runs on the failure path, where a second exception would hide
        the original one.
        """
        try:
            self._require_connected()
            assert self.rtde_control is not None and self.rtde_receive is not None
            if not self.rtde_control.isConnected() or not self.rtde_receive.isConnected():
                return "the RTDE connection has dropped"
            if not self.rtde_control.isProgramRunning():
                return (
                    "the RTDE control script is not running on the controller - reconnect, and check "
                    "that the robot is still in Remote Control mode"
                )
            if self.rtde_receive.isEmergencyStopped():
                return "the robot is emergency stopped"
            if self.rtde_receive.isProtectiveStopped():
                return "the robot is protective stopped - clear it on the pendant before retrying"
            robot_mode = int(self.rtde_receive.getRobotMode())
            safety_mode = int(self.rtde_receive.getSafetyMode())
            robot_mode_name = ROBOT_MODE_NAMES.get(robot_mode, f"unknown ({robot_mode})")
            safety_mode_name = SAFETY_MODE_NAMES.get(safety_mode, f"unknown ({safety_mode})")
            if robot_mode != 7:  # 7 = running; anything else cannot execute motion
                return f"the robot is not running (robot mode: {robot_mode_name}, safety mode: {safety_mode_name})"
            if safety_mode != 1:  # 1 = normal
                return f"the safety mode is {safety_mode_name}, not normal"
            return (
                f"the robot reports itself healthy (robot mode: {robot_mode_name}, safety mode: "
                f"{safety_mode_name}) - the target was most likely unreachable or outside a safety limit"
            )
        except Exception as ex:  # pylint: disable=broad-except
            return f"the cause could not be read back from the robot ({ex})"

    def _motion_failed(self, what: str) -> RuntimeError:
        return RuntimeError(f"{what} failed: {self.describe_motion_failure()}.")

    def is_home(self) -> bool:
        # Line tracing has no fixed "home" target; not meaningful here.
        return False

    def get_custom_target(self, custom_pose: TransformationMatrix) -> RobotTarget:
        return RobotTarget(name="Custom Target", pose=custom_pose)

    def get_safe_waypoint(self) -> RobotTarget:
        raise NotImplementedError(
            "There is no station-defined safe waypoint for line tracing; "
            "build one explicitly (e.g. current pose raised along tool Z) and use get_custom_target()."
        )

    def get_target_by_id(self, target_id: int) -> RobotTarget:
        raise NotImplementedError("Station target lists are not used by RobotControlURRTDE.")

    def get_number_of_regular_targets(self) -> int:
        return 0

    def move_l(
        self,
        target: RobotTarget,
        speed: float = DEFAULT_SPEED,
        acceleration: float = DEFAULT_ACCELERATION,
        asynchronous: bool = False,
    ) -> None:
        self._require_connected()
        assert self.rtde_control is not None
        success = self.rtde_control.moveL(_pose_to_rtde(target.pose), speed, acceleration, asynchronous)
        if not success:
            raise self._motion_failed("moveL")

    def move_j(
        self,
        target: RobotTarget,
        speed: float = DEFAULT_JOINT_SPEED,
        acceleration: float = DEFAULT_JOINT_ACCELERATION,
        asynchronous: bool = False,
    ) -> None:
        self._require_connected()
        assert self.rtde_control is not None and self.rtde_receive is not None
        current_q = self.rtde_receive.getActualQ()
        q = self.rtde_control.getInverseKinematics(_pose_to_rtde(target.pose), current_q)
        success = self.rtde_control.moveJ(q, speed, acceleration, asynchronous)
        if not success:
            raise self._motion_failed("moveJ")

    def move_to_joints(
        self,
        joint_positions: List[float],
        speed: float = DEFAULT_JOINT_SPEED,
        acceleration: float = DEFAULT_JOINT_ACCELERATION,
        asynchronous: bool = False,
    ) -> None:
        """moveJ straight to given joint angles (rad) - no IK, no Cartesian target. Used for
        the home/start posture, which is captured and stored as joint angles precisely so
        moving back to it doesn't depend on IK picking the same solution branch each time.
        """
        self._require_connected()
        assert self.rtde_control is not None
        success = self.rtde_control.moveJ(list(joint_positions), speed, acceleration, asynchronous)
        if not success:
            raise self._motion_failed("moveJ (to joints)")

    def move_path(
        self,
        waypoints: List[TransformationMatrix],
        speed: float = DEFAULT_SPEED,
        acceleration: float = DEFAULT_ACCELERATION,
        blend_radius: float = DEFAULT_BLEND_RADIUS,
        asynchronous: bool = False,
    ) -> None:
        """Move through all waypoints (base frame) as one blended motion - used for line
        tracing (see geometry.waypoint_builder). Unlike calling move_l() once per waypoint,
        the controller blends between segments instead of decelerating to a stop at each one.
        """
        self._require_connected()
        assert self.rtde_control is not None
        if len(waypoints) == 0:
            return
        path = rtde_control.Path()
        last_index = len(waypoints) - 1
        for index, waypoint in enumerate(waypoints):
            # The final waypoint can't blend into a next segment that doesn't exist.
            entry_blend_radius = 0.0 if index == last_index else blend_radius
            # Confirmed against ur_rtde's actual source (RTDEControlInterface's PathEntry
            # script generation): param_[6]=velocity, param_[7]=acceleration, param_[8]=blend.
            # (velocity, acceleration) - NOT (acceleration, velocity).
            parameters = _pose_to_rtde(waypoint) + [speed, acceleration, entry_blend_radius]
            path.addEntry(
                rtde_control.PathEntry(rtde_control.PathEntry.MoveL, rtde_control.PathEntry.PositionTcpPose, parameters)
            )
        success = self.rtde_control.movePath(path, asynchronous)
        if not success:
            raise self._motion_failed("movePath")

    def stop(self, deceleration: float = 10.0) -> None:
        if self.rtde_control is not None:
            self.rtde_control.stopL(deceleration)


if __name__ == "__main__":
    import sys

    if len(sys.argv) != 2:
        print("Usage: python -m line_tracing_gui.robot.robot_control_ur_rtde <robot_ip>")
        sys.exit(1)

    robot = RobotControlURRTDE(sys.argv[1])
    print(f"Connecting to robot on {robot.robot_configuration.ip_addr}...")
    robot.connect()
    print("Connected. Reading current pose...")
    start_target = robot.get_pose()
    print(f"Current pose (mm, rotvec rad): {start_target.pose.translation} {start_target.pose.rotation.as_rotvec()}")

    offset_target = robot.get_custom_target(
        TransformationMatrix(
            rotation=start_target.pose.rotation,
            translation=start_target.pose.translation + np.array([10.0, 0.0, 0.0], dtype=np.float32),
        )
    )
    print("Moving +10 mm in X (moveL, low speed)...")
    robot.move_l(offset_target, speed=0.02, acceleration=0.1)
    end_target = robot.get_pose()
    delta = end_target.pose.translation - start_target.pose.translation
    print(f"Delta (mm): {delta}")

    print("Moving back...")
    robot.move_l(start_target, speed=0.02, acceleration=0.1)
    robot.disconnect()
    print("Done.")