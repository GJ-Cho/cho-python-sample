"""
Script to verify UR robot movement via RTDE (Real-Time Data Exchange).

Reads the current TCP pose, calculates a target pose offset by +10 mm in X, Y, Z,
sends the target to the robot via RTDE registers, and prints before/after poses
along with actual vs. expected movement error.

Requirements:
    - universal_robots_communication_file.xml in the same directory as this script
    - ur_comm_test.urp loaded and running on the UR robot teach pendant

RTDE guide:
    https://www.universal-robots.com/how-tos-and-faqs/how-to/ur-how-tos/real-time-data-exchange-rtde-guide-22229/
"""

import argparse
import time
from pathlib import Path
from typing import Tuple

import numpy as np
from rtde import rtde, rtde_config

_MOVE_OFFSET_MM = 10.0


def _options() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ip", required=True, help="IP address of the UR robot")
    return parser.parse_args()


def _connect(host: str) -> Tuple[rtde.RTDE, rtde.serialize.DataObject]:
    """Connect to UR robot via RTDE and configure input/output registers.

    Args:
        host: Robot IP address

    Returns:
        con: Active RTDE connection
        robot_input: Input register data object

    Raises:
        RuntimeError: If protocol negotiation, output setup, or synchronization fails

    """
    conf = rtde_config.ConfigFile(Path(__file__).parent / "universal_robots_communication_file.xml")
    output_names, output_types = conf.get_recipe("out")
    input_names, input_types = conf.get_recipe("in")

    con = rtde.RTDE(host, 30004)
    con.connect()

    if not con.negotiate_protocol_version():
        raise RuntimeError("RTDE protocol version mismatch")
    if not con.send_output_setup(output_names, output_types, frequency=200):
        raise RuntimeError("Unable to configure RTDE output")

    robot_input = con.send_input_setup(input_names, input_types)

    if not con.send_start():
        raise RuntimeError("Unable to start RTDE synchronization")

    print("RTDE connection established.\n")
    return con, robot_input


def _read_state(con: rtde.RTDE) -> rtde.serialize:
    """Receive current robot state from RTDE output registers.

    Args:
        con: Active RTDE connection

    Returns:
        Current robot state

    """
    state = con.receive()
    assert state is not None, "Failed to receive robot state"
    return state


def _move_status(state) -> int:
    """Read move completion status from output_int_register_24. Returns -1 when complete."""
    return state.output_int_register_24


def _send_command(
    con: rtde.RTDE,
    robot_input: rtde.serialize.DataObject,
    target_pose_m: list,
    pc_ready: bool = False,
    move_confirmed: bool = False,
) -> None:
    """Write target pose and control signals to RTDE input registers.

    Args:
        con: Active RTDE connection
        robot_input: Input register data object
        target_pose_m: Target TCP pose [x, y, z, rx, ry, rz] in meters/radians
        pc_ready: Signal to robot that PC has set the target pose (input_bit_register_65)
        move_confirmed: Signal to robot that PC confirmed move completion (input_bit_register_64)

    """
    robot_input.input_bit_register_64 = int(move_confirmed)
    robot_input.input_bit_register_65 = int(pc_ready)
    robot_input.input_double_register_24 = float(target_pose_m[0])
    robot_input.input_double_register_25 = float(target_pose_m[1])
    robot_input.input_double_register_26 = float(target_pose_m[2])
    robot_input.input_double_register_27 = float(target_pose_m[3])
    robot_input.input_double_register_28 = float(target_pose_m[4])
    robot_input.input_double_register_29 = float(target_pose_m[5])
    con.send(robot_input)


def _to_mm(pose_m: np.ndarray) -> np.ndarray:
    """Convert RTDE TCP pose translation from meters to millimeters."""
    result = pose_m.copy()
    result[:3] *= 1000.0
    return result


def _print_pose(label: str, pose_mm: np.ndarray) -> None:
    x, y, z, rx, ry, rz = pose_mm
    print(label)
    print(f"  X={x:.3f} mm   Y={y:.3f} mm   Z={z:.3f} mm")
    print(f"  Rx={rx:.6f} rad   Ry={ry:.6f} rad   Rz={rz:.6f} rad\n")


def _run(con: rtde.RTDE, robot_input: rtde.serialize.DataObject) -> None:
    """Read current pose, move robot by +10 mm in X/Y/Z, verify result.

    Args:
        con: Active RTDE connection
        robot_input: Input register data object

    """
    offset_m = _MOVE_OFFSET_MM / 1000.0

    # Read and print current pose
    current_pose = np.array(_read_state(con).actual_TCP_pose, dtype=float)
    _print_pose("Current TCP pose:", _to_mm(current_pose))

    # Calculate target: current + [+10 mm, +10 mm, +10 mm, 0, 0, 0]
    target_pose = current_pose.copy()
    target_pose[0] += offset_m
    target_pose[1] += offset_m
    target_pose[2] += offset_m
    _print_pose(f"Target TCP pose (current + {_MOVE_OFFSET_MM:.0f} mm in X, Y, Z):", _to_mm(target_pose))

    # Send target pose and signal robot to start moving
    _send_command(con, robot_input, target_pose.tolist(), pc_ready=True)

    # Wait for move to complete (move_status == -1)
    print("Waiting for robot movement to complete...")
    while True:
        state = _read_state(con)
        if _move_status(state) == -1:
            break

    # Read and print final pose
    final_pose = np.array(state.actual_TCP_pose, dtype=float)
    _print_pose("Final TCP pose:", _to_mm(final_pose))

    # Movement verification: expected vs actual delta
    expected_delta = np.array([_MOVE_OFFSET_MM, _MOVE_OFFSET_MM, _MOVE_OFFSET_MM])
    actual_delta = (_to_mm(final_pose) - _to_mm(current_pose))[:3]
    error = actual_delta - expected_delta
    print("Movement verification:")
    print(f"  Expected delta XYZ : {expected_delta} mm")
    print(f"  Actual delta   XYZ : {actual_delta.round(3)} mm")
    print(f"  Error          XYZ : {error.round(3)} mm\n")

    # Signal move_confirmed then disconnect
    _send_command(con, robot_input, target_pose.tolist(), move_confirmed=True)
    time.sleep(0.5)
    _send_command(con, robot_input, target_pose.tolist(), move_confirmed=False)
    time.sleep(0.5)
    con.send_pause()
    con.disconnect()
    print("RTDE connection closed.")


def _main() -> None:
    con, robot_input = _connect(_options().ip)
    _run(con, robot_input)


if __name__ == "__main__":
    _main()
