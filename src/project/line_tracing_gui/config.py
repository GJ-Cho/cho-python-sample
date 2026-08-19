"""
Persistent settings for the Line Tracing GUI, following the same
QSettings("Zivid", "<AppName>") convention used by zividsamples (e.g. HandEyeGUI).

"""

from pathlib import Path
from typing import List, Optional

from PyQt5.QtCore import QSettings

ORG_NAME = "Zivid"
APP_NAME = "LineTracingGUI"

DEFAULT_ROBOT_IP = "192.168.1.10"


class AppConfig:
    def __init__(self) -> None:
        self.settings = QSettings(ORG_NAME, APP_NAME)

    def robot_ip(self) -> str:
        return str(self.settings.value("robot/ip", DEFAULT_ROBOT_IP))

    def set_robot_ip(self, ip: str) -> None:
        self.settings.setValue("robot/ip", ip)

    def hand_eye_transform_path(self) -> Optional[Path]:
        value = self.settings.value("calibration/hand_eye_transform_path", "")
        return Path(value) if value else None

    def set_hand_eye_transform_path(self, path: Path) -> None:
        self.settings.setValue("calibration/hand_eye_transform_path", str(path))

    def capture_pose_path(self) -> Optional[Path]:
        """Robot pose at capture time, needed only for eye-in-hand (see CalibrationPanel)."""
        value = self.settings.value("calibration/capture_pose_path", "")
        return Path(value) if value else None

    def set_capture_pose_path(self, path: Path) -> None:
        self.settings.setValue("calibration/capture_pose_path", str(path))

    def eye_in_hand(self) -> bool:
        return self.settings.value("calibration/eye_in_hand", False, type=bool)

    def set_eye_in_hand(self, eye_in_hand: bool) -> None:
        self.settings.setValue("calibration/eye_in_hand", eye_in_hand)

    def home_joints(self) -> Optional[List[float]]:
        """Home/start posture, as joint angles (rad). None if never set."""
        value = self.settings.value("robot/home_joints", "")
        if not value:
            return None
        return [float(v) for v in str(value).split(",")]

    def set_home_joints(self, joint_positions: List[float]) -> None:
        self.settings.setValue("robot/home_joints", ",".join(str(v) for v in joint_positions))