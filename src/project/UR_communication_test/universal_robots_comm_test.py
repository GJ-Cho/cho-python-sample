"""
Test code : communication between UR robot and PC(Python script).
The script communicates with the robot through Real-Time Data Exchange (RTDE) interface.

The entire sample consist of two additional files:
    - universal_robots_hand_eye_script.urp: Robot program script that moves between different poses.
    - robot_communication_file.xml: communication set-up file.

"""

import zivid # :) 
import argparse
import datetime
import time
from pathlib import Path
from typing import List, Tuple
import matplotlib.pyplot as plt
import keyboard
import pandas as pd

import numpy as np
from rtde import rtde, rtde_config
from scipy.spatial.transform import Rotation

IP_ROBOT = "192.168.56.101"  # Replace with your robot's IP address
SAVE_DIR = Path("dataset")  # Replace with your desired save directory

def _write_robot_state(
    con: rtde.RTDE,
    input_data: rtde.serialize.DataObject,
    finish_test: bool = False,
    ready_to_record: bool = False,
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
        finish_test: Boolean value to robot_state that q_r scene capture is finished
        ready_to_record: Boolean value to robot_state that ready to record
        x: Robot pose x coordinate
        y: Robot pose y coordinate
        z: Robot pose z coordinate
        rx: Robot pose rotation x coordinate (Radians)
        ry: Robot pose rotation y coordinate
        rz: Robot pose rotation z coordinate

    """
    input_data.input_bit_register_64 = bool(finish_test) 
    input_data.input_bit_register_65 = bool(ready_to_record)
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


def _loop_count(robot_state) -> int:
    """Read robot output register 24.

    Args:
        robot_state: Robot state "output_int_register_24"

    Returns:
        Number of loops that robot has done
        if Loop count is -1, it means that the robot stop moving

    """
    return robot_state.output_int_register_24


def _ready_for_test(robot_state) -> bool:
    """Read robot output register 64.

    Args:
        robot_state: Robot state "output_bit_register_64"

    Returns:
        Boolean value that states if camera is ready to test

    """
    return robot_state.output_bit_register_64


def _comm_test(con: rtde.RTDE, input_data: rtde.serialize.DataObject) -> None:
    """Generate dataset based on predefined robot poses.

    Args:
        con: Connection between computer and robot
        input_data: Input package containing the specific input data registers

    Returns:
        ?? : Save_dir to where dataset is saved Path ?

    """
    # Communication : Robot current pose check
    robot_state = _read_robot_state(con)
    pose = robot_state.actual_TCP_pose
    print("start robot pose: ",pose)

    # Signal robot that camera is ready
    _write_robot_state(con, input_data, finish_test=False, ready_to_record=True)
    time.sleep(1.0)

    # Wait for robot to be ready for test
    print("Waiting for robot to be ready for test...\n")
    while _ready_for_test(robot_state) == False:
        robot_state = _read_robot_state(con)

    print("______________________________________________________________________\n")
    print("Robot is ready to test!\n")
    print("Test Start!\n")
    print("______________________________________________________________________\n\n")
    
    # Plt setup
    plt.ion()
    fig, axs = plt.subplots(6, 1, figsize=(12, 10), sharex=True)
    lines = []

    for i in range(6):
        axs[i].set_ylabel(f'Joint {i+1}')
        axs[i].grid(True)
        line, = axs[i].plot([], [], label=f'Joint {i+1}')
        lines.append(line)

    axs[-1].set_xlabel('Time (s)')
    plt.tight_layout()

    # Add text for loop count
    loop_count_text = fig.text(0.95, 0.95, f'Loop Count: {_loop_count(robot_state)}', 
                              ha='right', va='top', fontsize=10, 
                              bbox=dict(facecolor='white', alpha=0.8))

    log_data = []
    timestamps = []
    joint_data = [[] for _ in range(6)]

    print("if you want to stop, press 's' key.")

    try:
        # main loop
        start_time = time.time()
        while _loop_count(robot_state) != -1:
            # Read robot state
            robot_state = _read_robot_state(con)

            # Update loop count text
            loop_count_text.set_text(f'Loop Count: {_loop_count(robot_state)}')

            now = time.time()
            elapsed = now - start_time

            actual_q = robot_state.actual_q
            actual_q_deg = np.rad2deg(actual_q)  # Convert radians to degrees
            actual_TCP_pose = robot_state.actual_TCP_pose
            timestamps.append(elapsed)

            # Save data each Axis : Actual joint position
            for i in range(6):
                joint_data[i].append(actual_q_deg[i])  # Use degrees instead of radians

            # Update plot
            for i in range(6):
                lines[i].set_data(timestamps, joint_data[i])
                axs[i].relim()
                axs[i].autoscale_view()

            plt.pause(0.01)
            # Save log data (time and joint positions)
            log_entry = {
                'time': elapsed,
                'loop_count': _loop_count(robot_state)  # Add loop count to log data
            }
            for i in range(6):
                log_entry[f'joint_{i+1}'] = actual_q_deg[i]  # Use degrees instead of radians
            log_data.append(log_entry)

            # if press 's' key, stop data collection
            if keyboard.is_pressed('s'):
                print("Sending signal to robot to stop data collection...")
                # 예시: 디지털 출력 0번을 HIGH로 설정
                _write_robot_state(con, input_data, finish_test=True, ready_to_record=False)
                break
    
    except KeyboardInterrupt:
        print("Ctrl+C pressed. Stopping data collection...")

    df = pd.DataFrame(log_data)
    df.to_csv("ur_rtde_joint_log.csv", index=False)
    print("CSV file saved : ur_rtde_joint_log.csv")
    
    _write_robot_state(con, input_data, finish_test=False, ready_to_record=False)
    
    time.sleep(1.0)
    
    con.send_pause()
    con.disconnect()

    plt.show(block=True)  # Keep the plot window open
    return


def _main() -> None:
    
    robot_ip_address = IP_ROBOT
    con, input_data = _initialize_robot_sync(robot_ip_address)
    con.send_start()

    _comm_test( con, input_data)

    


if __name__ == "__main__":
    _main()
