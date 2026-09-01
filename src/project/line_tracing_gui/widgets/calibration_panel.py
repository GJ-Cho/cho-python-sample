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

That capture pose always comes from reading the robot - there is no file for it.
The robot stands still while the frame is taken, so its pose at that moment is
the capture pose, and TracePanel reads it on every capture (see
record_capture_pose). Read From Robot repeats that on demand, which is only
correct while the robot still stands where it stood for the capture. The value
doubles as a position to send the robot back to (Move To Capture Pose).

The robot only ever reports getActualTCPPose(), i.e. the pose with the
controller's tool offset applied - and this app deliberately keeps a non-zero one
for the pointed gripper (see PLAN.md). So which frame the capture pose ends up in
is decided by Pose Reference, and it has to be the frame the hand-eye YAML is
expressed in, or the middle frame does not cancel in
base_T_touch = capture_pose * hand_eye * camera_T_point:

    Flange : hand_eye = flange_T_camera -> capture pose = base_T_flange
             (get_flange_pose takes the active TCP offset back out)
    TCP    : hand_eye = tcp_T_camera    -> capture pose = base_T_tcp
             (getActualTCPPose used as-is)

Zivid's own samples calibrate against the flange, so Flange is the default and
what this rig uses. Either way this only matters for eye-in-hand - eye-to-hand
never multiplies by a robot pose at all.

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

from line_tracing_gui.config import POSE_REFERENCE_FLANGE, POSE_REFERENCE_TCP, AppConfig
from line_tracing_gui.theme import TEXT_MUTED
from line_tracing_gui.widgets.robot_connection_widget import RobotConnectionWidget

RECORDED_AT_CAPTURE_SOURCE = "(recorded at capture)"
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

        # Zivid's own touch-test sample commands the robot in flange poses and takes the TCP
        # out in software, so the robot poses a hand-eye calibration registers are flange
        # poses and its result is flange_T_camera. Calibrating with a TCP active instead
        # yields tcp_T_camera, which Pose Reference below can still accommodate - but this
        # says up front which one the panel expects.
        self.eye_in_hand_note = QLabel(
            "Eye-In-Hand: the YAML should be the camera pose relative to the FLANGE "
            "(calibrate with the TCP zeroed). Pose Reference below must match it."
        )
        self.eye_in_hand_note.setWordWrap(True)
        self.eye_in_hand_note.setStyleSheet(f"color: {TEXT_MUTED};")
        self.eye_in_hand_note.setVisible(self.eye_in_hand)

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
        group_layout.addWidget(self.eye_in_hand_note)
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
        # Kept to one line on purpose; the conditions and caveats are on the buttons' tooltips
        # and in this module's docstring rather than crowding the panel.
        hint_label = QLabel(
            "Robot pose at capture time, in the frame Pose Reference selects - read off the robot "
            "on every Eye-In-Hand capture. Move To Capture Pose sends the robot back there."
        )
        hint_label.setWordWrap(True)
        hint_label.setStyleSheet(f"color: {TEXT_MUTED};")

        # The robot only ever reports getActualTCPPose(), so which frame the capture pose ends
        # up in is decided here - and it has to be the frame the hand-eye YAML is expressed in,
        # otherwise the middle frame does not cancel in
        # base_T_touch = capture_pose * hand_eye * camera_T_point.
        reference_explanation_label = QLabel(
            "Pose Reference - which frame the hand-eye YAML is expressed in:\n"
            "    Flange:  hand_eye = flange_T_camera  →  capture pose = base_T_flange"
            "   (the active TCP offset is taken back out)\n"
            "    TCP:     hand_eye = tcp_T_camera     →  capture pose = base_T_tcp"
            "      (what the robot reports, used as-is)"
        )
        reference_explanation_label.setWordWrap(True)
        reference_explanation_label.setStyleSheet(f"color: {TEXT_MUTED};")

        self.flange_reference_radio = QRadioButton("Flange (6th axis face, TCP removed)")
        self.tcp_reference_radio = QRadioButton("TCP (as configured on the controller)")
        self.pose_reference_button_group = QButtonGroup(self)
        self.pose_reference_button_group.addButton(self.flange_reference_radio)
        self.pose_reference_button_group.addButton(self.tcp_reference_radio)
        is_flange = self.config.hand_eye_pose_reference() == POSE_REFERENCE_FLANGE
        (self.flange_reference_radio if is_flange else self.tcp_reference_radio).setChecked(True)
        self.flange_reference_radio.toggled.connect(self.on_pose_reference_toggled)

        self.capture_pose_source_field = QLineEdit()
        self.capture_pose_source_field.setReadOnly(True)
        self.capture_pose_source_field.setPlaceholderText("(not set)")
        self.read_capture_pose_button = QPushButton("Read From Robot")
        self.read_capture_pose_button.setToolTip(
            "Read the robot's current pose as the capture pose.\n"
            "Only valid while the robot still stands where it stood for the capture -\n"
            "move it first and every waypoint comes out transformed by the difference.\n"
            "Captures do this on their own; this is for re-reading it by hand."
        )
        self.read_capture_pose_button.clicked.connect(self.on_read_capture_pose_clicked)
        self.move_to_capture_pose_button = QPushButton("Move To Capture Pose")
        self.move_to_capture_pose_button.setToolTip(
            "Send the robot back to the recorded capture pose (moveJ, so not a straight line).\n"
            "Asks for confirmation with the target coordinates first."
        )
        self.move_to_capture_pose_button.setEnabled(False)
        self.move_to_capture_pose_button.clicked.connect(self.on_move_to_capture_pose_clicked)

        self.capture_pose_widget = PoseWidget.Robot(eye_in_hand=True, display_mode=PoseWidgetDisplayMode.Basic)

        source_row = QHBoxLayout()
        source_row.addWidget(self.capture_pose_source_field, stretch=1)
        source_row.addWidget(self.read_capture_pose_button)
        source_row.addWidget(self.move_to_capture_pose_button)

        reference_row = QHBoxLayout()
        reference_row.addWidget(self.flange_reference_radio)
        reference_row.addWidget(self.tcp_reference_radio)
        reference_row.addStretch(1)

        form_layout = QFormLayout()
        form_layout.addRow("Pose Reference", reference_row)
        form_layout.addRow("Capture Pose", source_row)

        group_box = QGroupBox("Robot Capture Pose")
        group_layout = QVBoxLayout()
        group_layout.setSpacing(10)
        group_layout.addWidget(hint_label)
        group_layout.addWidget(reference_explanation_label)
        group_layout.addLayout(form_layout)
        group_layout.addWidget(self.capture_pose_widget)
        group_box.setLayout(group_layout)
        return group_box

    def _set_capture_pose(self, pose: TransformationMatrix, source: str) -> None:
        self.capture_pose_widget.set_transformation_matrix(pose)
        self.capture_pose_source_field.setText(source)
        self.capture_pose_is_set = True
        self.move_to_capture_pose_button.setEnabled(True)


    def record_capture_pose(self, source: str = RECORDED_AT_CAPTURE_SOURCE) -> Optional[str]:
        """Record the robot's current pose as the capture pose; returns an error, or None.

        TracePanel calls this from on_capture_clicked: the robot stands still while the frame
        is taken, so its pose right then *is* the capture pose. Recording it there means
        nothing has to be entered by hand, and it doubles as a position to send the robot
        back to later (see on_move_to_capture_pose_clicked).
        """
        robot_control = self.robot_connection_widget.robot_control
        if robot_control is None:
            return "Robot is not connected."
        try:
            # get_pose() is getActualTCPPose(), i.e. the TCP pose. get_flange_pose() takes the
            # controller's TCP offset back out - see get_pose_reference().
            if self.get_pose_reference() == POSE_REFERENCE_FLANGE:
                target = robot_control.get_flange_pose()
            else:
                target = robot_control.get_pose()
        except Exception as ex:  # pylint: disable=broad-except
            return f"Failed to read the pose: {ex}"
        self._set_capture_pose(target.pose, source)
        return None

    def on_read_capture_pose_clicked(self) -> None:
        error_message = self.record_capture_pose(READ_FROM_ROBOT_SOURCE)
        if error_message is not None:
            QMessageBox.warning(self, "Robot Capture Pose", error_message)

    def on_move_to_capture_pose_clicked(self) -> None:
        pose = self.get_capture_pose()
        if pose is None:
            return
        robot_control = self.robot_connection_widget.robot_control
        if robot_control is None:
            QMessageBox.warning(self, "Move To Capture Pose", "Robot is not connected.")
            return
        # move_j takes a *TCP* target, so a pose stored against the flange has to have the
        # TCP offset put back on - otherwise the robot lands one TCP offset short.
        try:
            if self.get_pose_reference() == POSE_REFERENCE_FLANGE:
                pose = pose * robot_control.get_tcp_offset()
        except Exception as ex:  # pylint: disable=broad-except
            QMessageBox.warning(self, "Move To Capture Pose", f"Failed to read the TCP offset: {ex}")
            return
        translation = pose.translation
        confirm = QMessageBox.question(
            self,
            "Move To Capture Pose",
            "The robot will move (moveJ) to the recorded capture pose:\n"
            f"X={translation[0]:.1f} Y={translation[1]:.1f} Z={translation[2]:.1f} mm\n\n"
            "Make sure the path there is clear. Continue?",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No,
        )
        if confirm != QMessageBox.Yes:
            return
        success, message = self.robot_connection_widget.move_to_pose(pose)
        if success:
            QMessageBox.information(self, "Move To Capture Pose", message)
        else:
            QMessageBox.warning(self, "Move To Capture Pose", message)

    def get_pose_reference(self) -> str:
        """Which robot pose the hand-eye calibration was performed against.

        POSE_REFERENCE_FLANGE means the calibration used the bare 6th-axis face, so the TCP
        offset has to come back out of what the robot reports; POSE_REFERENCE_TCP means it
        used the pose with the TCP applied, which is what the robot reports directly.
        """
        return POSE_REFERENCE_FLANGE if self.flange_reference_radio.isChecked() else POSE_REFERENCE_TCP

    def on_pose_reference_toggled(self, _flange_checked: bool) -> None:
        self.config.set_hand_eye_pose_reference(self.get_pose_reference())
        # Anything already recorded is in the other convention, so it is now wrong by one TCP
        # offset. Drop it rather than leave a plausible-looking but stale pose on screen.
        if self.capture_pose_is_set:
            self._clear_capture_pose()

    def _clear_capture_pose(self) -> None:
        self.capture_pose_widget.set_transformation_matrix(TransformationMatrix())
        self.capture_pose_source_field.clear()
        self.capture_pose_is_set = False
        self.move_to_capture_pose_button.setEnabled(False)

    # --- Shared ------------------------------------------------------------------------------

    def on_mount_type_toggled(self, eye_to_hand_checked: bool) -> None:
        self.eye_in_hand = not eye_to_hand_checked
        self.config.set_eye_in_hand(self.eye_in_hand)
        self.hand_eye_pose_widget.on_eye_in_hand_toggled(self.eye_in_hand)
        self.eye_in_hand_note.setVisible(self.eye_in_hand)
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
