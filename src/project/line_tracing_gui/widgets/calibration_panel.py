"""
Hand-eye calibration panel: load a hand-eye transform YAML (produced by
Zivid's Hand-Eye GUI or CLI samples), display it, and choose whether the
camera is mounted eye-in-hand or eye-to-hand (this changes how
geometry.waypoint_builder turns a camera-frame point into a robot-base-frame
target - see its docstring).

Eye-to-hand needs nothing but that transform, so the panel is just the one
section. Eye-in-hand additionally needs the robot pose at the moment the frame
was captured - point_base = robot_pose * hand_eye_transform * point_camera - so a
second section appears for it, and is hidden again for eye-to-hand.

That capture pose can be read straight off a connected robot or loaded from a
YAML file. Reading it off the robot is only correct while the robot is still
standing where it stood when the frame was captured; move it first and the
waypoints come out transformed by the difference.

No TCP offset UI here - the robot controller's own set_tcp handles the
gripper tip offset (see robot/README.md), not this app.

"""

from pathlib import Path
from typing import Optional

from PyQt5.QtCore import pyqtSignal
from PyQt5.QtWidgets import (
    QButtonGroup,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QRadioButton,
    QVBoxLayout,
    QWidget,
)
from zividsamples.gui.widgets.pose_widget import PoseWidget, PoseWidgetDisplayMode
from zividsamples.save_load_matrix import load_and_assert_affine_matrix
from zividsamples.transformation_matrix import TransformationMatrix

from line_tracing_gui.config import AppConfig
from line_tracing_gui.theme import TEXT_MUTED
from line_tracing_gui.widgets.robot_connection_widget import RobotConnectionWidget

READ_FROM_ROBOT_SOURCE = "(read from the robot)"


class CalibrationPanel(QWidget):
    hand_eye_transform_updated = pyqtSignal(TransformationMatrix)
    eye_in_hand_changed = pyqtSignal(bool)

    def __init__(
        self,
        robot_connection_widget: RobotConnectionWidget,
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self.config = AppConfig()
        self.robot_connection_widget = robot_connection_widget
        self.eye_in_hand = self.config.eye_in_hand()
        self.capture_pose_is_set = False

        self.hand_eye_group_box = self._build_hand_eye_group_box()
        self.capture_pose_group_box = self._build_capture_pose_group_box()
        self.capture_pose_group_box.setVisible(self.eye_in_hand)

        layout = QVBoxLayout()
        layout.addWidget(self.hand_eye_group_box)
        layout.addWidget(self.capture_pose_group_box)
        layout.addStretch(1)  # sections pack at the top instead of spreading down the tab
        self.setLayout(layout)

        saved_hand_eye_path = self.config.hand_eye_transform_path()
        if saved_hand_eye_path is not None and saved_hand_eye_path.exists():
            self._load_hand_eye_from_path(saved_hand_eye_path)

        saved_capture_pose_path = self.config.capture_pose_path()
        if saved_capture_pose_path is not None and saved_capture_pose_path.exists():
            self._load_capture_pose_from_path(saved_capture_pose_path)

    # --- Hand-eye transform ----------------------------------------------------------------

    def _build_hand_eye_group_box(self) -> QGroupBox:
        self.eye_to_hand_radio = QRadioButton("Eye-To-Hand (camera stationary)")
        self.eye_in_hand_radio = QRadioButton("Eye-In-Hand (camera on flange)")
        self.mount_button_group = QButtonGroup(self)
        self.mount_button_group.addButton(self.eye_to_hand_radio)
        self.mount_button_group.addButton(self.eye_in_hand_radio)
        (self.eye_in_hand_radio if self.eye_in_hand else self.eye_to_hand_radio).setChecked(True)
        self.eye_to_hand_radio.toggled.connect(self.on_mount_type_toggled)

        self.hand_eye_path_field = QLineEdit()
        self.hand_eye_path_field.setReadOnly(True)
        self.hand_eye_path_field.setPlaceholderText("(no file loaded)")
        self.load_hand_eye_button = QPushButton("Load...")
        self.load_hand_eye_button.clicked.connect(self.on_load_hand_eye_clicked)

        self.hand_eye_pose_widget = PoseWidget.HandEye(
            eye_in_hand=self.eye_in_hand, display_mode=PoseWidgetDisplayMode.Basic
        )

        mount_row = QHBoxLayout()
        mount_row.addWidget(self.eye_to_hand_radio)
        mount_row.addWidget(self.eye_in_hand_radio)
        mount_row.addStretch(1)

        path_row = QHBoxLayout()
        path_row.addWidget(self.hand_eye_path_field, stretch=1)
        path_row.addWidget(self.load_hand_eye_button)

        form_layout = QFormLayout()
        form_layout.addRow("Camera Mount", mount_row)
        form_layout.addRow("Hand-Eye YAML", path_row)

        group_box = QGroupBox("Hand-Eye Calibration")
        group_layout = QVBoxLayout()
        group_layout.setSpacing(10)
        group_layout.addLayout(form_layout)
        group_layout.addWidget(self.hand_eye_pose_widget)
        group_box.setLayout(group_layout)
        return group_box

    def _load_hand_eye_from_path(self, path: Path) -> None:
        try:
            matrix = load_and_assert_affine_matrix(path)
        except RuntimeError as ex:
            QMessageBox.warning(self, "Hand-Eye Calibration", f"Failed to load {path}:\n{ex}")
            return
        transformation_matrix = TransformationMatrix.from_matrix(matrix)
        self.hand_eye_pose_widget.set_transformation_matrix(transformation_matrix)
        self.hand_eye_path_field.setText(str(path))
        self.config.set_hand_eye_transform_path(path)
        self.hand_eye_transform_updated.emit(transformation_matrix)

    def on_load_hand_eye_clicked(self) -> None:
        saved_path = self.config.hand_eye_transform_path()
        start_dir = str(saved_path.parent) if saved_path is not None else str(Path.home())
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Select Hand-Eye Transform YAML", start_dir, "YAML files (*.yaml *.yml)"
        )
        if not file_path:
            return
        self._load_hand_eye_from_path(Path(file_path))

    # --- Robot capture pose (eye-in-hand only) ----------------------------------------------

    def _build_capture_pose_group_box(self) -> QGroupBox:
        hint_label = QLabel(
            "The robot's end-effector pose in base frame at the moment the frame was captured. "
            "Reading it from the robot is only valid while the robot still stands where it stood "
            "for the capture."
        )
        hint_label.setWordWrap(True)
        hint_label.setStyleSheet(f"color: {TEXT_MUTED};")

        self.capture_pose_source_field = QLineEdit()
        self.capture_pose_source_field.setReadOnly(True)
        self.capture_pose_source_field.setPlaceholderText("(not set)")
        self.load_capture_pose_button = QPushButton("Load...")
        self.load_capture_pose_button.clicked.connect(self.on_load_capture_pose_clicked)
        self.read_capture_pose_button = QPushButton("Read From Robot")
        self.read_capture_pose_button.clicked.connect(self.on_read_capture_pose_clicked)

        self.capture_pose_widget = PoseWidget.Robot(eye_in_hand=True, display_mode=PoseWidgetDisplayMode.Basic)

        source_row = QHBoxLayout()
        source_row.addWidget(self.capture_pose_source_field, stretch=1)
        source_row.addWidget(self.load_capture_pose_button)
        source_row.addWidget(self.read_capture_pose_button)

        form_layout = QFormLayout()
        form_layout.addRow("Capture Pose", source_row)

        group_box = QGroupBox("Robot Capture Pose")
        group_layout = QVBoxLayout()
        group_layout.setSpacing(10)
        group_layout.addWidget(hint_label)
        group_layout.addLayout(form_layout)
        group_layout.addWidget(self.capture_pose_widget)
        group_box.setLayout(group_layout)
        return group_box

    def _set_capture_pose(self, pose: TransformationMatrix, source: str) -> None:
        self.capture_pose_widget.set_transformation_matrix(pose)
        self.capture_pose_source_field.setText(source)
        self.capture_pose_is_set = True

    def _load_capture_pose_from_path(self, path: Path) -> None:
        try:
            matrix = load_and_assert_affine_matrix(path)
        except RuntimeError as ex:
            QMessageBox.warning(self, "Robot Capture Pose", f"Failed to load {path}:\n{ex}")
            return
        self._set_capture_pose(TransformationMatrix.from_matrix(matrix), str(path))
        self.config.set_capture_pose_path(path)

    def on_load_capture_pose_clicked(self) -> None:
        saved_path = self.config.capture_pose_path()
        start_dir = str(saved_path.parent) if saved_path is not None else str(Path.home())
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Select Robot Capture Pose YAML", start_dir, "YAML files (*.yaml *.yml)"
        )
        if not file_path:
            return
        self._load_capture_pose_from_path(Path(file_path))

    def on_read_capture_pose_clicked(self) -> None:
        robot_control = self.robot_connection_widget.robot_control
        if robot_control is None:
            QMessageBox.warning(self, "Robot Capture Pose", "Robot is not connected.")
            return
        try:
            target = robot_control.get_pose()
        except Exception as ex:  # pylint: disable=broad-except
            QMessageBox.warning(self, "Robot Capture Pose", f"Failed to read the pose: {ex}")
            return
        self._set_capture_pose(target.pose, READ_FROM_ROBOT_SOURCE)

    # --- Shared ------------------------------------------------------------------------------

    def on_mount_type_toggled(self, eye_to_hand_checked: bool) -> None:
        self.eye_in_hand = not eye_to_hand_checked
        self.config.set_eye_in_hand(self.eye_in_hand)
        self.hand_eye_pose_widget.on_eye_in_hand_toggled(self.eye_in_hand)
        self.capture_pose_group_box.setVisible(self.eye_in_hand)
        self.eye_in_hand_changed.emit(self.eye_in_hand)

    def get_hand_eye_transform(self) -> TransformationMatrix:
        return self.hand_eye_pose_widget.get_transformation_matrix()

    def get_eye_in_hand(self) -> bool:
        return self.eye_in_hand

    def get_capture_pose(self) -> Optional[TransformationMatrix]:
        """The robot pose at capture time, or None when eye-to-hand or nothing is set yet.

        geometry.waypoint_builder.build_waypoints needs this as `robot_pose` whenever
        eye_in_hand is True.
        """
        if not self.eye_in_hand or not self.capture_pose_is_set:
            return None
        return self.capture_pose_widget.get_transformation_matrix()
