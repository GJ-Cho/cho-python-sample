"""
Hand-eye calibration panel: load a hand-eye transform YAML (produced by
Zivid's Hand-Eye GUI or CLI samples), display it, and choose whether the
camera is mounted eye-in-hand or eye-to-hand (this changes how
geometry.waypoint_builder turns a camera-frame point into a robot-base-frame
target - see its docstring).

No TCP offset UI here - the robot controller's own set_tcp handles the
gripper tip offset (see robot_program/README.md), not this app.

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


class CalibrationPanel(QWidget):
    hand_eye_transform_updated = pyqtSignal(TransformationMatrix)
    eye_in_hand_changed = pyqtSignal(bool)

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.config = AppConfig()
        self.eye_in_hand = self.config.eye_in_hand()

        self.eye_to_hand_radio = QRadioButton("Eye-To-Hand (camera stationary)")
        self.eye_in_hand_radio = QRadioButton("Eye-In-Hand (camera on flange)")
        self.mount_button_group = QButtonGroup(self)
        self.mount_button_group.addButton(self.eye_to_hand_radio)
        self.mount_button_group.addButton(self.eye_in_hand_radio)
        (self.eye_in_hand_radio if self.eye_in_hand else self.eye_to_hand_radio).setChecked(True)
        self.eye_to_hand_radio.toggled.connect(self.on_mount_type_toggled)

        self.eye_in_hand_note = QLabel(
            "⚠ Eye-In-Hand additionally needs the robot pose at capture time (handled in the Line Tracing tab)."
        )
        self.eye_in_hand_note.setWordWrap(True)

        self.path_field = QLineEdit()
        self.path_field.setReadOnly(True)
        self.path_field.setPlaceholderText("(no file loaded)")
        self.load_button = QPushButton("Load...")
        self.load_button.clicked.connect(self.on_load_clicked)

        self.hand_eye_pose_widget = PoseWidget.HandEye(
            eye_in_hand=self.eye_in_hand, display_mode=PoseWidgetDisplayMode.Basic
        )

        group_box = QGroupBox("Hand-Eye Calibration")
        group_layout = QVBoxLayout()
        group_layout.setSpacing(10)

        form_layout = QFormLayout()
        mount_row = QHBoxLayout()
        mount_row.addWidget(self.eye_to_hand_radio)
        mount_row.addWidget(self.eye_in_hand_radio)
        mount_row.addStretch(1)
        form_layout.addRow("Camera Mount", mount_row)

        path_row = QHBoxLayout()
        path_row.addWidget(self.path_field, stretch=1)
        path_row.addWidget(self.load_button)
        form_layout.addRow("Hand-Eye YAML", path_row)

        group_layout.addLayout(form_layout)
        group_layout.addWidget(self.eye_in_hand_note)
        group_layout.addWidget(self.hand_eye_pose_widget)
        group_layout.addStretch(1)  # keeps the rows packed at the top rather than spread out
        group_box.setLayout(group_layout)

        layout = QVBoxLayout()
        layout.addWidget(group_box)
        self.setLayout(layout)

        self._update_eye_in_hand_note()

        saved_path = self.config.hand_eye_transform_path()
        if saved_path is not None and saved_path.exists():
            self._load_from_path(saved_path)

    def _update_eye_in_hand_note(self) -> None:
        self.eye_in_hand_note.setVisible(self.eye_in_hand)

    def on_mount_type_toggled(self, eye_to_hand_checked: bool) -> None:
        self.eye_in_hand = not eye_to_hand_checked
        self.config.set_eye_in_hand(self.eye_in_hand)
        self.hand_eye_pose_widget.on_eye_in_hand_toggled(self.eye_in_hand)
        self._update_eye_in_hand_note()
        self.eye_in_hand_changed.emit(self.eye_in_hand)

    def _load_from_path(self, path: Path) -> None:
        try:
            matrix = load_and_assert_affine_matrix(path)
        except RuntimeError as ex:
            QMessageBox.warning(self, "Calibration", f"Failed to load {path}:\n{ex}")
            return
        transformation_matrix = TransformationMatrix.from_matrix(matrix)
        self.hand_eye_pose_widget.set_transformation_matrix(transformation_matrix)
        self.path_field.setText(str(path))
        self.config.set_hand_eye_transform_path(path)
        self.hand_eye_transform_updated.emit(transformation_matrix)

    def on_load_clicked(self) -> None:
        saved_path = self.config.hand_eye_transform_path()
        start_dir = str(saved_path.parent) if saved_path is not None else str(Path.home())
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Select Hand-Eye Transform YAML", start_dir, "YAML files (*.yaml *.yml)"
        )
        if not file_path:
            return
        self._load_from_path(Path(file_path))

    def get_hand_eye_transform(self) -> TransformationMatrix:
        return self.hand_eye_pose_widget.get_transformation_matrix()

    def get_eye_in_hand(self) -> bool:
        return self.eye_in_hand
