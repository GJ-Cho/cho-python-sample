"""
RobotControl implementation for UR3e using the official ur_rtde library
(rtde_control.RTDEControlInterface / rtde_receive.RTDEReceiveInterface).

No custom URScript needs to run on the teach pendant - the only prerequisite
is enabling Remote Control mode on the robot (e-Series: top-right icon on
the pendant -> select "Remote Control", or Installation -> General ->
Remote Control). RTDEControlInterface's constructor connects immediately
and will raise if Remote Control mode isn't enabled.

"""

from typing import List, Optional, Tuple

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
_ZERO_RTDE_POSE = [0.0] * 6  # identity pose, for asking FK for the bare flange


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

        Uses get_tcp_offset(), the command-side offset, so it is only right once that
        matches the controller - run sync_command_tcp_offset first if unsure. The
        getForwardKinematics route would not need that, but it costs an extra control-script
        round trip and this is called on every eye-in-hand capture.
        """
        self._require_connected()
        return RobotTarget(name="Current Flange Pose", pose=self.get_pose().pose * self.get_tcp_offset().inv())

    def get_controller_tcp_offset(self) -> TransformationMatrix:
        """The TCP offset the controller actually has active, from measurements alone.

        Asks the calibrated kinematics for the flange with an explicitly zero TCP, so the
        answer holds whatever the command side happens to carry, then reads the offset off
        as flange^-1 * actual TCP pose. This costs a control-script round trip
        (getForwardKinematics), so it is called deliberately - never on every pose read.
        """
        self._require_connected()
        assert self.rtde_control is not None and self.rtde_receive is not None
        flange_rtde_pose = self.rtde_control.getForwardKinematics(self.rtde_receive.getActualQ(), _ZERO_RTDE_POSE)
        flange_pose = _rtde_pose_to_transformation_matrix(flange_rtde_pose)
        return flange_pose.inv() * self.get_pose().pose

    def sync_command_tcp_offset(self) -> Tuple[TransformationMatrix, TransformationMatrix]:
        """Point the command side's TCP at the one the controller has active.

        moveL, movePath and getInverseKinematics all place the *command* side's TCP
        (get_tcp_offset) on the pose they are given, and RTDEControlInterface starts with
        its own - not necessarily the pendant's installation TCP. When the two disagree,
        commanded poses land the flange where the tool tip was meant to go.

        Returns (offset before, offset after) so callers can report a correction.
        """
        self._require_connected()
        assert self.rtde_control is not None
        before = self.get_tcp_offset()
        after = self.get_controller_tcp_offset()
        if not self.rtde_control.setTcp(_pose_to_rtde(after)):
            raise RuntimeError("setTcp failed while syncing the command-side TCP offset.")
        return before, after

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
        """TCP offset the *command* side uses: what moveL/movePath/IK put on a given pose.

        This is RTDEControlInterface's own offset, which starts out independent of the
        pendant's installation TCP - compare against get_controller_tcp_offset, and use
        sync_command_tcp_offset to make them agree.
        """
        self._require_connected()
        assert self.rtde_control is not None
        return _rtde_pose_to_transformation_matrix(self.rtde_control.getTCPOffset())

    def set_tcp_offset(self, tcp_offset: TransformationMatrix) -> None:
        self._require_connected()
        assert self.rtde_control is not None
        success = self.rtde_control.setTcp(_pose_to_rtde(tcp_offset))
        if not success:
            raise RuntimeError("setTcp failed.")

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
            raise RuntimeError("moveL failed (robot may have refused or stopped the motion).")

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
            raise RuntimeError("moveJ failed (robot may have refused/stopped the motion, or no IK solution).")

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
            raise RuntimeError("moveJ (to joints) failed (robot may have refused or stopped the motion).")

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
            raise RuntimeError("movePath failed (robot may have refused or stopped the motion).")

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