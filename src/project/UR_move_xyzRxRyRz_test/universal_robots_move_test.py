"""
Script to generate a dataset and perform hand-eye calibration using a Universal Robot UR5e robot.
The script communicates with the robot through Real-Time Data Exchange (RTDE) interface.
More information about RTDE:
https://www.universal-robots.com/how-tos-and-faqs/how-to/ur-how-tos/real-time-data-exchange-rtde-guide-22229/

The entire sample consist of two additional files:
    - universal_robots_hand_eye_script.urp: Robot program script that moves between different poses.
    - robot_communication_file.xml: communication set-up file.

Running the sample requires that you have universal_robots_hand_eye_script.urp on your UR5e robot,
and robot_communication_file.xml in the same repo as this sample. Each robot pose
must be modified to your scene. This is done in universal_robots_hand_eye_script.urp on the robot.

Further explanation of this sample is found in our knowledge base:
https://support.zivid.com/latest/academy/applications/hand-eye/ur5-robot-%2B-python-generate-dataset-and-perform-hand-eye-calibration.html

"""

import argparse
import datetime
import time
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np
import zivid
from rtde import rtde, rtde_config
from scipy.spatial.transform import Rotation


def _options() -> argparse.Namespace:
    """Function for taking in arguments from user.

    Returns:
        Arguments from user

    """
    parser = argparse.ArgumentParser(description=__doc__)
    mode_group = parser.add_mutually_exclusive_group(required=True)
    mode_group.add_argument("--eih", "--eye-in-hand", action="store_true", help="eye-in-hand calibration")
    mode_group.add_argument("--eth", "--eye-to-hand", action="store_true", help="eye-to-hand calibration")
    parser.add_argument("--ip", required=True, help="IP address to robot")

    return parser.parse_args()


def _write_robot_state(
    con: rtde.RTDE,
    input_data: rtde.serialize.DataObject,
    finish_capture: bool = False,
    camera_ready: bool = False,
    x: float = 0.4801234567,
    y: float = -0.50001,
    z: float = 0.4401111111,
    rx: float = 3,
    ry: float = 0.0666,
    rz: float = -0.234,
) -> None:
    """Write to robot I/O registers.

    Args:
        con: Connection between computer and robot
        input_data: Input package containing the specific input data registers
        finish_capture: Boolean value to robot_state that q_r scene capture is finished
        camera_ready: Boolean value to robot_state that camera is ready to capture images

    """
    input_data.input_bit_register_64 = int(finish_capture)
    input_data.input_bit_register_65 = int(camera_ready)
    input_data.input_double_register_24 = float(x)
    input_data.input_double_register_25 = float(y)
    input_data.input_double_register_26 = float(z)
    input_data.input_double_register_27 = float(rx)
    input_data.input_double_register_28 = float(ry)
    input_data.input_double_register_29 = float(rz)

    con.send(input_data)


def _initialize_robot_sync(host: str) -> Tuple[rtde.RTDE, rtde.serialize.DataObject]:
    """Set up communication with UR robot.

    Args:
        host: IP address

    Returns:
        con: Connection to robot
        robot_input_data: Package containing the specific input data registers

    Raises:
        RuntimeError: If protocol do not match
        RuntimeError: If script is unable to configure output
        RuntimeError: If synchronization is not possible

    """
    conf = rtde_config.ConfigFile(Path(Path.cwd() / "universal_robots_communication_file.xml"))
    output_names, output_types = conf.get_recipe("out")
    input_names, input_types = conf.get_recipe("in")

    # port 30004 is reserved for rtde
    con = rtde.RTDE(host, 30004)
    con.connect()

    # To ensure that the application is compatible with further versions of UR controller
    if not con.negotiate_protocol_version():
        raise RuntimeError("Protocol do not match")

    if not con.send_output_setup(output_names, output_types, frequency=200):
        raise RuntimeError("Unable to configure output")

    robot_input_data = con.send_input_setup(input_names, input_types)

    if not con.send_start():
        raise RuntimeError("Unable to start synchronization")

    print("Communication initialization completed. \n")

    return con, robot_input_data


def _read_robot_state(con: rtde.RTDE) -> rtde.serialize:
    """Receive robot output recipe.

    Args:
        con: Connection between computer and robot

    Returns:
        robot_state: Robot state

    """
    robot_state = con.receive()

    assert robot_state is not None, "Not able to receive robot_state"

    return robot_state


def _image_count(robot_state) -> int:
    """Read robot output register 24.

    Args:
        robot_state: Robot state

    Returns:
        Number of captured images

    """
    return robot_state.output_int_register_24


def _ready_for_capture(robot_state) -> bool:
    """Read robot output register 64.

    Args:
        robot_state: Robot state

    Returns:
        Boolean value that states if camera is ready to capture

    """
    return robot_state.output_bit_register_64


def _comm_test(app: zivid.Application, con: rtde.RTDE, input_data: rtde.serialize.DataObject) -> Path:
    """Generate dataset based on predefined robot poses.

    Args:
        app: Zivid application instance
        con: Connection between computer and robot
        input_data: Input package containing the specific input data registers

    Returns:
        ?? : Save_dir to where dataset is saved

    """

    # Signal robot that camera is ready
    ready_to_capture = True
    _write_robot_state(con, input_data, finish_capture=False, camera_ready=ready_to_capture)

    robot_state = _read_robot_state(con)
    pose = robot_state.actual_TCP_pose

    print("start robot pose: ",pose)

    print("Please check Robot variable x,y,z,rx,ry,rz values!")

    while _image_count(robot_state) != -1:
            robot_state = _read_robot_state(con)


    robot_state = _read_robot_state(con)
    pose = robot_state.actual_TCP_pose

    print("moved robot pose: ",pose)

    _write_robot_state(con, input_data, finish_capture=False, camera_ready=False)
    time.sleep(1.0)
    con.send_pause()
    con.disconnect()

    return


def _main() -> None:
    app = zivid.Application()
    user_options = _options()

    robot_ip_address = user_options.ip
    con, input_data = _initialize_robot_sync(robot_ip_address)
    con.send_start()

    _comm_test(app, con, input_data)


if __name__ == "__main__":
    _main()
