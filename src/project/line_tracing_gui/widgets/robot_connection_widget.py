"""
UR3e connection panel for the Line Tracing GUI.

Not a reuse of zividsamples.gui.robot.robot_control_widget.RobotControlWidget:
that widget is tightly coupled to RoboDK's station-target-list concept
(get_target_by_id / get_number_of_regular_targets), which line tracing has no
use for. This widget follows the same visual/interaction pattern (IP field,
Connect button with color-coded status, background thread for blocking
moves) but talks directly to RobotControlURRTDE.

"""

import queue
import threading
from typing import Optional

import numpy as np
from PyQt5.QtCore import QTimer, pyqtSignal
from PyQt5.QtWidgets import (
    QApplication,
    QDoubleSpinBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
    QWidget,
)
from scipy.spatial.transform import Rotation
from zividsamples.gui.robot.robot_control import RobotTarget
from zividsamples.transformation_matrix import TransformationMatrix

from line_tracing_gui.config import AppConfig
from line_tracing_gui.robot.robot_control_ur_rtde import RobotControlURRTDE
from line_tracing_gui.theme import (
    BUSY_BUTTON_STYLE,
    STATUS_DANGER_STYLE,
    STATUS_OK_STYLE,
    STATUS_WARNING_STYLE,
)
from line_tracing_gui.widgets.spin_box_stepper import SpinBoxStepper

POSE_POLL_INTERVAL_MS = 300
TEST_MOVE_OFFSET_MM = 10.0
TEST_MOVE_VELOCITY = 0.02  # m/s, deliberately slow for a first verification move
TEST_MOVE_ACCELERATION = 0.1  # m/s^2
TCP_FIELD_WIDTH_PX = 76


class RobotConnectionWidget(QWidget):
    robot_connected = pyqtSignal(bool)
    actual_pose_updated = pyqtSignal(RobotTarget)
    actual_joints_updated = pyqtSignal(list)  # 6 joint angles, radians

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.config = AppConfig()
        self.robot_control: Optional[RobotControlURRTDE] = None
        self.connected = False
        self.result_queue: Optional[queue.Queue] = None

        self.ip_input = QLineEdit(self.config.robot_ip())
        self.connect_button = QPushButton("Connect")
        self.connect_button.setCheckable(True)
        self.reconnect_button = QPushButton("Reconnect")
        self.pose_label = QLabel("(not connected)")
        self.status_label = QLabel("(not connected)")
        self.test_move_button = QPushButton(f"Round-Trip Test (+{TEST_MOVE_OFFSET_MM:.0f} mm X, slow)")
        self.test_move_button.setEnabled(False)

        group_box = QGroupBox("Robot (UR3e)")
        group_layout = QVBoxLayout()
        ip_form = QFormLayout()
        ip_form.addRow("Robot IP", self.ip_input)
        group_layout.addLayout(ip_form)
        buttons_layout = QHBoxLayout()
        buttons_layout.addWidget(self.connect_button)
        buttons_layout.addWidget(self.reconnect_button)
        buttons_layout.addWidget(self.test_move_button)
        buttons_layout.addStretch(1)  # buttons keep their natural width instead of filling the row
        group_layout.addLayout(buttons_layout)
        group_layout.addWidget(self.status_label)
        group_layout.addWidget(self.pose_label)
        group_box.setLayout(group_layout)

        layout = QVBoxLayout()
        layout.addWidget(group_box)
        layout.addWidget(self._build_tcp_offset_panel())
        # Keeps both sections top-aligned now that this widget sits in a tall column
        # of its own on the Connect tab, rather than stacked under the camera panel.
        layout.addStretch(1)
        self.setLayout(layout)

        self.connect_button.clicked.connect(self.on_connect_clicked)
        self.reconnect_button.clicked.connect(self.on_reconnect_clicked)
        self.test_move_button.clicked.connect(self.on_test_move_clicked)

        self.pose_timer = QTimer(self)
        self.pose_timer.setInterval(POSE_POLL_INTERVAL_MS)
        self.pose_timer.timeout.connect(self.on_poll_pose)

    def _build_tcp_offset_panel(self) -> QWidget:
        self.tcp_x_spinbox = QDoubleSpinBox()
        self.tcp_y_spinbox = QDoubleSpinBox()
        self.tcp_z_spinbox = QDoubleSpinBox()
        for spinbox in (self.tcp_x_spinbox, self.tcp_y_spinbox, self.tcp_z_spinbox):
            spinbox.setRange(-1000.0, 1000.0)
            spinbox.setSuffix(" mm")

        self.tcp_rx_spinbox = QDoubleSpinBox()
        self.tcp_ry_spinbox = QDoubleSpinBox()
        self.tcp_rz_spinbox = QDoubleSpinBox()
        for spinbox in (self.tcp_rx_spinbox, self.tcp_ry_spinbox, self.tcp_rz_spinbox):
            spinbox.setRange(-180.0, 180.0)
            spinbox.setSuffix(" deg")

        self.load_tcp_button = QPushButton("Load")
        self.load_tcp_button.clicked.connect(self.on_load_tcp_clicked)
        self.apply_tcp_button = QPushButton("Apply")
        self.apply_tcp_button.clicked.connect(self.on_apply_tcp_clicked)

        # Narrower fields than the default: three steppers have to fit side by side in
        # the Connect tab's right-hand column.
        translation_row = QHBoxLayout()
        for spinbox in (self.tcp_x_spinbox, self.tcp_y_spinbox, self.tcp_z_spinbox):
            translation_row.addWidget(SpinBoxStepper(spinbox, field_width_px=TCP_FIELD_WIDTH_PX))
        translation_row.addStretch(1)

        rotation_row = QHBoxLayout()
        for spinbox in (self.tcp_rx_spinbox, self.tcp_ry_spinbox, self.tcp_rz_spinbox):
            rotation_row.addWidget(SpinBoxStepper(spinbox, field_width_px=TCP_FIELD_WIDTH_PX))
        rotation_row.addStretch(1)

        buttons_row = QHBoxLayout()
        buttons_row.addWidget(self.load_tcp_button)
        buttons_row.addWidget(self.apply_tcp_button)
        buttons_row.addStretch(1)

        form = QFormLayout()
        form.addRow("Position (X, Y, Z)", translation_row)
        form.addRow("Rotation (Rx, Ry, Rz)", rotation_row)

        group_box = QGroupBox("TCP Offset (robot controller set_tcp)")
        group_layout = QVBoxLayout()
        group_layout.addLayout(form)
        group_layout.addLayout(buttons_row)
        group_box.setLayout(group_layout)
        return group_box

    def on_load_tcp_clicked(self) -> None:
        if self.robot_control is None:
            QMessageBox.warning(self, "TCP Offset", "Robot is not connected.")
            return
        try:
            tcp_offset = self.robot_control.get_tcp_offset()
        except Exception as ex:  # pylint: disable=broad-except
            QMessageBox.warning(self, "TCP Offset", f"Failed to read: {ex}")
            return
        translation = tcp_offset.translation
        rotation_deg = tcp_offset.rotation.as_rotvec(degrees=True)
        self.tcp_x_spinbox.setValue(float(translation[0]))
        self.tcp_y_spinbox.setValue(float(translation[1]))
        self.tcp_z_spinbox.setValue(float(translation[2]))
        self.tcp_rx_spinbox.setValue(float(rotation_deg[0]))
        self.tcp_ry_spinbox.setValue(float(rotation_deg[1]))
        self.tcp_rz_spinbox.setValue(float(rotation_deg[2]))

    def on_apply_tcp_clicked(self) -> None:
        if self.robot_control is None:
            QMessageBox.warning(self, "TCP Offset", "Robot is not connected.")
            return
        translation = np.array(
            [self.tcp_x_spinbox.value(), self.tcp_y_spinbox.value(), self.tcp_z_spinbox.value()], dtype=np.float32
        )
        rotation = Rotation.from_rotvec(
            [self.tcp_rx_spinbox.value(), self.tcp_ry_spinbox.value(), self.tcp_rz_spinbox.value()], degrees=True
        )
        confirm = QMessageBox.question(
            self,
            "Apply TCP Offset",
            f"Set the TCP offset to X={translation[0]:.1f} Y={translation[1]:.1f} Z={translation[2]:.1f} mm.\n"
            "This affects every subsequent move and waypoint calculation. Continue?",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No,
        )
        if confirm != QMessageBox.Yes:
            return
        try:
            self.robot_control.set_tcp_offset(TransformationMatrix(rotation=rotation, translation=translation))
            QMessageBox.information(self, "TCP Offset", "Applied.")
        except Exception as ex:  # pylint: disable=broad-except
            QMessageBox.warning(self, "TCP Offset", f"Failed to apply: {ex}")

    def on_connect_clicked(self) -> None:
        if self.connected:
            self.disconnect_robot()
            return
        ip_addr = self.ip_input.text().strip()
        self.connect_button.setChecked(True)
        self.connect_button.setStyleSheet(BUSY_BUTTON_STYLE)
        self.connect_button.setText("Connecting...")
        QApplication.processEvents()
        try:
            self.robot_control = RobotControlURRTDE(ip_addr)
            self.robot_control.connect()
            self.config.set_robot_ip(ip_addr)
            self.connected = True
            # Connected: the button stays checked, which the theme styles as the active
            # state - clear the transient busy color so that rule applies again.
            self.connect_button.setStyleSheet("")
            self.connect_button.setText("Disconnect")
            self.test_move_button.setEnabled(True)
            self.on_load_tcp_clicked()
            self.pose_timer.start()
            self.robot_connected.emit(True)
        except Exception as ex:  # pylint: disable=broad-except
            QMessageBox.warning(self, "Robot Control", f"Failed to connect: {ex}")
            self.robot_control = None
            self.connect_button.setChecked(False)
            self.connect_button.setStyleSheet("")
            self.connect_button.setText("Connect")

    def disconnect_robot(self) -> None:
        self.pose_timer.stop()
        if self.robot_control is not None:
            self.robot_control.disconnect()
        self.robot_control = None
        self.connected = False
        self.connect_button.setChecked(False)
        self.connect_button.setStyleSheet("")
        self.connect_button.setText("Connect")
        self.test_move_button.setEnabled(False)
        self.pose_label.setText("(not connected)")
        self.status_label.setText("(not connected)")
        self.status_label.setStyleSheet("")
        self.robot_connected.emit(False)

    def on_reconnect_clicked(self) -> None:
        """Recreate the RTDE interfaces from scratch. RTDE pose readouts keep working through
        an e-stop/protective-stop (that's why the pose display can look fine while moves fail),
        but the safest recovery once the fault is cleared on the pendant is a fresh connection
        rather than assuming the old interface objects are still in a good state."""
        if self.connected:
            self.disconnect_robot()
        self.on_connect_clicked()

    def pause_polling(self) -> None:
        """Stop the periodic TCP-pose poll. Callers that run their own blocking
        move (e.g. TracePanel's line execution) on this same RobotControlURRTDE
        must pause polling first - two threads calling rtde_control/rtde_receive
        at once is not safe (see on_test_move_clicked for the same precaution)."""
        self.pose_timer.stop()

    def resume_polling(self) -> None:
        if self.connected:
            self.pose_timer.start()

    def on_poll_pose(self) -> None:
        if self.robot_control is None:
            return
        try:
            target = self.robot_control.get_pose()
            translation = target.pose.translation
            self.pose_label.setText(
                f"TCP (mm): X={translation[0]:.1f}  Y={translation[1]:.1f}  Z={translation[2]:.1f}"
            )
            self._update_status_label()
            self.actual_pose_updated.emit(target)
            self.actual_joints_updated.emit(self.robot_control.get_joint_positions())
        except Exception as ex:  # pylint: disable=broad-except
            QMessageBox.warning(self, "Robot Control", f"Lost connection: {ex}")
            self.disconnect_robot()

    def _update_status_label(self) -> None:
        # RTDE pose readouts (get_pose above) keep working through an e-stop/protective-stop -
        # that alone does NOT mean move_l/move_path will succeed. Check the actual safety
        # state so a failed move has an obvious explanation instead of a generic error.
        assert self.robot_control is not None
        if self.robot_control.is_emergency_stopped():
            self.status_label.setText("⚠ Emergency stop - clear on the pendant, then press 'Reconnect'")
            self.status_label.setStyleSheet(STATUS_DANGER_STYLE)
        elif self.robot_control.is_protective_stopped():
            self.status_label.setText("⚠ Protective stop - clear on the pendant, then press 'Reconnect'")
            self.status_label.setStyleSheet(STATUS_WARNING_STYLE)
        else:
            self.status_label.setText("Normal (ready to move)")
            self.status_label.setStyleSheet(STATUS_OK_STYLE)

    def _run_test_move(self) -> None:
        assert self.robot_control is not None
        assert self.result_queue is not None
        try:
            start_target = self.robot_control.get_pose()
            offset_pose = TransformationMatrix(
                rotation=start_target.pose.rotation,
                translation=start_target.pose.translation.copy(),
            )
            offset_pose.translation[0] += TEST_MOVE_OFFSET_MM
            self.robot_control.move_l(
                self.robot_control.get_custom_target(offset_pose),
                speed=TEST_MOVE_VELOCITY,
                acceleration=TEST_MOVE_ACCELERATION,
            )
            end_target = self.robot_control.get_pose()
            self.robot_control.move_l(
                self.robot_control.get_custom_target(start_target.pose),
                speed=TEST_MOVE_VELOCITY,
                acceleration=TEST_MOVE_ACCELERATION,
            )
            delta = end_target.pose.translation - start_target.pose.translation
            self.result_queue.put((True, delta))
        except Exception as ex:  # pylint: disable=broad-except
            self.result_queue.put((False, str(ex)))

    def on_test_move_clicked(self) -> None:
        if self.robot_control is None:
            return
        self.test_move_button.setEnabled(False)
        self.test_move_button.setStyleSheet(BUSY_BUTTON_STYLE)
        QApplication.processEvents()
        # The pose-poll timer and the move thread would otherwise call send()/receive()
        # on the same RTDE socket from two threads at once - pause polling for the move.
        self.pause_polling()
        self.result_queue = queue.Queue()
        move_thread = threading.Thread(target=self._run_test_move)
        move_thread.start()
        while move_thread.is_alive():
            QApplication.processEvents()
        success, result = self.result_queue.get()
        self.resume_polling()
        self.test_move_button.setStyleSheet("")
        self.test_move_button.setEnabled(True)
        if success:
            QMessageBox.information(
                self,
                "Round-Trip Test Complete",
                f"Measured displacement (mm): {result}\n(expected: [{TEST_MOVE_OFFSET_MM:.0f}, 0, 0])",
            )
        else:
            QMessageBox.warning(self, "Round-Trip Test Failed", str(result))