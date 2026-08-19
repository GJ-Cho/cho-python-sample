"""
Line Tracing tab: capture a 2D+3D frame, draw a line on it, build/preview
the resulting robot waypoints, and run them on the robot.

Layout is a horizontal split: 2D capture/draw on the left (with its own
buttons above it), 3D point cloud/waypoint preview on the right (with its
own controls above it) - both visible at once. A third section below covers
robot execution: an optional home/start posture, approach/retreat, speed/
acceleration/blend, run/stop, and live TCP+joint monitoring.

Execution sequence when run: [home] -> approach (moveJ) -> waypoints
(move_path, blended moveL) -> retreat (moveJ) -> [home]. The home leg only
happens if a home posture has been captured (see on_set_home_clicked).

"""

import queue
import threading
import time
from typing import List, Optional

import numpy as np
import zivid
from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import (
    QApplication,
    QDoubleSpinBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QSplitter,
    QVBoxLayout,
    QWidget,
)
from zividsamples.transformation_matrix import TransformationMatrix

from line_tracing_gui.config import AppConfig
from line_tracing_gui.geometry.waypoint_builder import DEFAULT_SAMPLE_SPACING_MM, build_waypoints
from line_tracing_gui.robot.robot_control_ur_rtde import DEFAULT_JOINT_ACCELERATION, DEFAULT_JOINT_SPEED
from line_tracing_gui.theme import mark_as_accent, mark_as_danger
from line_tracing_gui.widgets.calibration_panel import CalibrationPanel
from line_tracing_gui.widgets.camera_panel import CameraPanel
from line_tracing_gui.widgets.drawable_image_viewer import DrawableImageViewer
from line_tracing_gui.widgets.pointcloud_viewer_widget import PointCloudViewerWidget
from line_tracing_gui.widgets.robot_connection_widget import RobotConnectionWidget
from line_tracing_gui.widgets.spin_box_stepper import SpinBoxStepper

SPACING_MIN_MM = 1.0
SPACING_MAX_MM = 50.0

APPROACH_DEFAULT_MM = 30.0
APPROACH_MIN_MM = 0.0
APPROACH_MAX_MM = 150.0

EXEC_SPEED_DEFAULT_MM_S = 20.0
EXEC_SPEED_MIN_MM_S = 1.0
EXEC_SPEED_MAX_MM_S = 200.0

EXEC_ACCELERATION_DEFAULT = 0.2
EXEC_ACCELERATION_MIN = 0.05
EXEC_ACCELERATION_MAX = 1.0

EXEC_BLEND_DEFAULT_MM = 1.0
EXEC_BLEND_MIN_MM = 0.0
EXEC_BLEND_MAX_MM = 20.0

MONITOR_POLL_INTERVAL_S = 0.05


class TracePanel(QWidget):
    # Emitted from the execution worker thread (Qt auto-queues delivery to this widget's
    # thread) so the live TCP/joint monitors and the 3D "current position" marker can
    # update while a move is in progress.
    live_pose_updated = pyqtSignal(object)  # TransformationMatrix, base frame
    live_joints_updated = pyqtSignal(list)  # radians

    def __init__(
        self,
        camera_panel: CameraPanel,
        calibration_panel: CalibrationPanel,
        robot_connection_widget: RobotConnectionWidget,
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self.camera_panel = camera_panel
        self.calibration_panel = calibration_panel
        self.robot_connection_widget = robot_connection_widget
        self.config = AppConfig()
        self.point_cloud_xyz: Optional[np.ndarray] = None
        self.point_cloud_rgb: Optional[np.ndarray] = None
        self.waypoints: List[TransformationMatrix] = []  # base frame, set by on_generate_waypoints_clicked

        splitter = QSplitter(Qt.Horizontal)
        splitter.addWidget(self._build_2d_panel())
        splitter.addWidget(self._build_3d_panel())
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 1)

        layout = QVBoxLayout()
        layout.addWidget(splitter)
        layout.addWidget(self._build_execution_panel())
        self.setLayout(layout)

        self.robot_connection_widget.robot_connected.connect(self._update_execute_button_state)
        self.robot_connection_widget.actual_pose_updated.connect(lambda target: self._on_pose_updated(target.pose))
        self.robot_connection_widget.actual_joints_updated.connect(self._on_joints_updated)
        self.live_pose_updated.connect(self._on_pose_updated)
        self.live_joints_updated.connect(self._on_joints_updated)

        self._update_home_status_label()

    def _build_2d_panel(self) -> QWidget:
        self.capture_button = QPushButton("Capture (2D+3D)")
        mark_as_accent(self.capture_button)
        self.capture_button.clicked.connect(self.on_capture_clicked)

        self.draw_mode_button = QPushButton("Draw Line")
        self.draw_mode_button.setCheckable(True)
        self.draw_mode_button.setEnabled(False)
        self.draw_mode_button.toggled.connect(self.on_draw_mode_toggled)

        self.clear_button = QPushButton("Clear")
        self.clear_button.setEnabled(False)
        self.clear_button.clicked.connect(self.on_clear_clicked)

        self.undo_button = QPushButton("Undo")
        self.undo_button.setEnabled(False)
        self.undo_button.clicked.connect(self.on_undo_clicked)

        self.status_label_2d = QLabel("Connect a camera and capture.")

        self.image_viewer = DrawableImageViewer()
        self.image_viewer.setMinimumSize(400, 400)

        buttons_layout = QHBoxLayout()
        buttons_layout.addWidget(self.capture_button)
        buttons_layout.addWidget(self.draw_mode_button)
        buttons_layout.addWidget(self.clear_button)
        buttons_layout.addWidget(self.undo_button)
        buttons_layout.addStretch(1)  # buttons keep their natural width instead of filling the row

        group_box = QGroupBox("2D Capture / Draw Line")
        group_layout = QVBoxLayout()
        group_layout.addLayout(buttons_layout)
        group_layout.addWidget(self.status_label_2d)
        group_layout.addWidget(self.image_viewer)
        group_box.setLayout(group_layout)
        return group_box

    def _build_3d_panel(self) -> QWidget:
        self.spacing_spinbox = QDoubleSpinBox()
        self.spacing_spinbox.setRange(SPACING_MIN_MM, SPACING_MAX_MM)
        self.spacing_spinbox.setValue(DEFAULT_SAMPLE_SPACING_MM)
        self.spacing_spinbox.setSuffix(" mm")
        self.spacing_spinbox.setToolTip(
            "Target spacing between waypoints (approximated in pixel space - the actual mm spacing\n"
            "varies with the distance to the surface)"
        )

        self.generate_waypoints_button = QPushButton("Generate Waypoints")
        mark_as_accent(self.generate_waypoints_button)
        self.generate_waypoints_button.setEnabled(False)
        self.generate_waypoints_button.clicked.connect(self.on_generate_waypoints_clicked)

        self.status_label_3d = QLabel("Capture, then draw a line - the point cloud shows up here.")

        self.pointcloud_viewer = PointCloudViewerWidget()
        self.pointcloud_viewer.setMinimumSize(400, 400)

        controls_form = QFormLayout()
        spacing_row = QHBoxLayout()
        spacing_row.addWidget(SpinBoxStepper(self.spacing_spinbox))
        spacing_row.addWidget(self.generate_waypoints_button)
        spacing_row.addStretch(1)
        controls_form.addRow("Waypoint Spacing", spacing_row)

        group_box = QGroupBox("3D Point Cloud / Waypoints (yellow dot = current robot position)")
        group_layout = QVBoxLayout()
        group_layout.addLayout(controls_form)
        group_layout.addWidget(self.status_label_3d)
        group_layout.addWidget(self.pointcloud_viewer)
        group_box.setLayout(group_layout)
        return group_box

    def _build_execution_panel(self) -> QWidget:
        self.set_home_button = QPushButton("Set Current As Home")
        self.set_home_button.clicked.connect(self.on_set_home_clicked)
        self.move_home_button = QPushButton("Move To Home")
        self.move_home_button.clicked.connect(self.on_move_home_clicked)
        self.home_status_label = QLabel("")

        self.approach_spinbox = QDoubleSpinBox()
        self.approach_spinbox.setRange(APPROACH_MIN_MM, APPROACH_MAX_MM)
        self.approach_spinbox.setValue(APPROACH_DEFAULT_MM)
        self.approach_spinbox.setSuffix(" mm")
        self.approach_spinbox.setToolTip(
            "Standoff distance from the surface where the gripper starts its approach and ends its\n"
            "retreat, at the first/last point of the line (along the waypoint's local -Z, i.e. the\n"
            "direction opposite to where the tip points)"
        )

        self.speed_spinbox = QDoubleSpinBox()
        self.speed_spinbox.setRange(EXEC_SPEED_MIN_MM_S, EXEC_SPEED_MAX_MM_S)
        self.speed_spinbox.setValue(EXEC_SPEED_DEFAULT_MM_S)
        self.speed_spinbox.setSuffix(" mm/s")
        self.speed_spinbox.setToolTip(
            "Speed for the waypoint-following leg (move_path). Approach, retreat and home moves\n"
            "use the moveJ defaults."
        )

        self.acceleration_spinbox = QDoubleSpinBox()
        self.acceleration_spinbox.setRange(EXEC_ACCELERATION_MIN, EXEC_ACCELERATION_MAX)
        self.acceleration_spinbox.setValue(EXEC_ACCELERATION_DEFAULT)
        self.acceleration_spinbox.setSuffix(" m/s²")

        self.blend_spinbox = QDoubleSpinBox()
        self.blend_spinbox.setRange(EXEC_BLEND_MIN_MM, EXEC_BLEND_MAX_MM)
        self.blend_spinbox.setValue(EXEC_BLEND_DEFAULT_MM)
        self.blend_spinbox.setSuffix(" mm")
        self.blend_spinbox.setToolTip(
            "Execution is refused if this exceeds half the spacing between adjacent waypoints -\n"
            "overlapping blend zones make the robot skip intermediate points and speed up abruptly."
        )

        self.execute_button = QPushButton("Execute")
        mark_as_accent(self.execute_button)
        self.execute_button.setEnabled(False)
        self.execute_button.clicked.connect(self.on_execute_clicked)

        self.stop_button = QPushButton("Stop")
        mark_as_danger(self.stop_button)
        for button in (self.execute_button, self.stop_button):
            button.setMinimumWidth(150)  # wide enough to read as the panel's main action
        self.stop_button.setEnabled(False)
        self.stop_button.clicked.connect(self.on_stop_clicked)

        self.execution_status_label = QLabel("")

        self.tcp_monitor_label = QLabel("TCP: -")
        self.joints_monitor_label = QLabel("Joints: -")

        home_row = QHBoxLayout()
        home_row.addWidget(self.set_home_button)
        home_row.addWidget(self.move_home_button)
        home_row.addWidget(self.home_status_label)
        home_row.addStretch(1)

        # One setting per row (label right next to its own field, nothing stretched apart) -
        # a single wide row with 4 label+field pairs left big, confusing gaps between each
        # label and its field once the window is wide.
        form = QFormLayout()
        form.setFieldGrowthPolicy(QFormLayout.FieldsStayAtSizeHint)
        form.addRow("Home / Start Posture", home_row)
        form.addRow("Approach / Retreat", SpinBoxStepper(self.approach_spinbox))
        form.addRow("Speed", SpinBoxStepper(self.speed_spinbox))
        form.addRow("Acceleration", SpinBoxStepper(self.acceleration_spinbox))
        form.addRow("Blend", SpinBoxStepper(self.blend_spinbox))

        # Execute/Stop sit to the right of the settings form, in the wide empty space that
        # would otherwise be left blank next to the narrow (FieldsStayAtSizeHint) fields.
        buttons_column = QVBoxLayout()
        buttons_column.addWidget(self.execute_button)
        buttons_column.addWidget(self.stop_button)
        buttons_column.addWidget(self.execution_status_label)
        buttons_column.addStretch(1)

        form_and_buttons_row = QHBoxLayout()
        form_and_buttons_row.addLayout(form, 0)
        form_and_buttons_row.addSpacing(40)
        form_and_buttons_row.addLayout(buttons_column, 0)
        form_and_buttons_row.addStretch(1)

        group_box = QGroupBox("Robot Execution")
        group_layout = QVBoxLayout()
        group_layout.addLayout(form_and_buttons_row)
        group_layout.addWidget(self.tcp_monitor_label)
        group_layout.addWidget(self.joints_monitor_label)
        group_box.setLayout(group_layout)
        return group_box

    # --- 2D capture / draw ---------------------------------------------------------------

    def _capture_settings(self) -> zivid.Settings:
        settings = zivid.Settings()
        settings.acquisitions.append(zivid.Settings.Acquisition())
        return settings

    def on_capture_clicked(self) -> None:
        camera = self.camera_panel.camera
        if camera is None or not camera.state.connected:
            QMessageBox.warning(self, "Capture", "Connect a camera first.")
            return

        self.camera_panel.stop_live_preview()
        self.camera_panel.update_capture_button_state()

        try:
            with camera.capture(self._capture_settings()) as frame:
                point_cloud = frame.point_cloud()
                self.point_cloud_xyz = point_cloud.copy_data("xyz")
                rgba = point_cloud.copy_data("rgba_srgb")
        except RuntimeError as ex:
            QMessageBox.critical(self, "Capture Failed", str(ex))
            return

        self.point_cloud_rgb = rgba
        self.waypoints = []

        qimage = QImage(rgba.data, rgba.shape[1], rgba.shape[0], QImage.Format_RGBA8888)
        self.image_viewer.set_pixmap(QPixmap.fromImage(qimage), reset_zoom=True)
        self.pointcloud_viewer.show_point_cloud(self.point_cloud_xyz, self.point_cloud_rgb)

        self.draw_mode_button.setEnabled(True)
        self.clear_button.setEnabled(True)
        self.undo_button.setEnabled(True)
        self.generate_waypoints_button.setEnabled(True)
        self.status_label_2d.setText(f"Captured ({rgba.shape[1]}x{rgba.shape[0]}) - draw a line.")
        self.status_label_3d.setText("Point cloud shown - draw a line, then generate waypoints.")
        self._update_execute_button_state()

    def on_draw_mode_toggled(self, checked: bool) -> None:
        self.image_viewer.set_draw_mode(checked)
        self.draw_mode_button.setText("Stop Drawing" if checked else "Draw Line")

    def on_clear_clicked(self) -> None:
        self.image_viewer.clear_line()

    def on_undo_clicked(self) -> None:
        self.image_viewer.undo_last_stroke()

    # --- Waypoint generation --------------------------------------------------------------

    def on_generate_waypoints_clicked(self) -> None:
        if self.point_cloud_xyz is None:
            QMessageBox.warning(self, "Generate Waypoints", "Capture first.")
            return
        line_points_px = self.image_viewer.get_line_points()
        if len(line_points_px) < 2:
            QMessageBox.warning(self, "Generate Waypoints", "Draw a line first.")
            return

        hand_eye_transform = self.calibration_panel.get_hand_eye_transform()
        eye_in_hand = self.calibration_panel.get_eye_in_hand()
        capture_pose = self.calibration_panel.get_capture_pose()
        if eye_in_hand and capture_pose is None:
            QMessageBox.warning(
                self,
                "Generate Waypoints",
                "Eye-In-Hand needs the robot pose at capture time.\n"
                "Set it under Robot Capture Pose in the Calibration tab - either load a YAML, or\n"
                "read it off the robot while it still stands where it stood for the capture.",
            )
            return

        try:
            result = build_waypoints(
                line_points_px,
                self.point_cloud_xyz,
                hand_eye_transform,
                eye_in_hand=eye_in_hand,
                robot_pose=capture_pose,
                sample_spacing_mm=self.spacing_spinbox.value(),
            )
        except ValueError as ex:
            QMessageBox.warning(self, "Generate Waypoints", str(ex))
            return

        if len(result.waypoints) < 2:
            QMessageBox.warning(
                self,
                "Generate Waypoints",
                f"Only {len(result.waypoints)} valid waypoint(s) (skipped {result.skipped_pixel_count}).\n"
                "Check that the line lies over valid 3D data.",
            )
            return

        self.waypoints = result.waypoints
        self.status_label_3d.setText(
            f"Generated {len(result.waypoints)} waypoints "
            f"(skipped {result.skipped_pixel_count}, merged {result.merged_close_waypoint_count} too close together)"
        )
        self._auto_adjust_blend_spinbox()

        # The 3D view is drawn in camera frame (matching the point cloud); undo the
        # camera-to-base transform to get each waypoint back from base frame for display.
        camera_to_base_transform = self._camera_to_base_transform()
        camera_frame_waypoints = [camera_to_base_transform.inv() * waypoint for waypoint in self.waypoints]
        self.pointcloud_viewer.show_waypoints(camera_frame_waypoints)
        self._update_execute_button_state()

    def _camera_to_base_transform(self) -> TransformationMatrix:
        """Camera frame -> robot base frame, matching what build_waypoints applies.

        Eye-to-hand is the hand-eye transform alone; eye-in-hand additionally goes
        through the robot's pose at capture time. Falls back to the hand-eye transform
        when that pose is not set, so the 3D preview still draws something sensible.
        """
        hand_eye_transform = self.calibration_panel.get_hand_eye_transform()
        capture_pose = self.calibration_panel.get_capture_pose()
        if capture_pose is None:
            return hand_eye_transform
        return capture_pose * hand_eye_transform

    def _update_execute_button_state(self) -> None:
        has_waypoints = len(self.waypoints) >= 2
        self.execute_button.setEnabled(has_waypoints and self.robot_connection_widget.connected)

    def _auto_adjust_blend_spinbox(self) -> None:
        """Set the blend spinbox to a value that's safely valid for the just-generated
        waypoints, so the user isn't stuck manually retuning it against the "Blend Radius
        Too Large" check in on_execute_clicked every time the line's curvature changes the
        real 3D spacing (build_waypoints already floors that spacing, but the safe blend
        value still depends on it, so it's set here rather than as a fixed default)."""
        min_spacing = self._min_waypoint_spacing_mm()
        if not np.isfinite(min_spacing):
            return
        safe_blend = min_spacing * 0.4  # comfortably under the spacing/2 hard limit
        self.blend_spinbox.setValue(max(EXEC_BLEND_MIN_MM, min(EXEC_BLEND_MAX_MM, safe_blend)))

    # --- Live TCP/joint monitoring + 3D "current position" marker ------------------------

    def _on_pose_updated(self, pose: TransformationMatrix) -> None:
        translation = pose.translation
        rotation_deg = pose.rotation.as_rotvec(degrees=True)
        self.tcp_monitor_label.setText(
            "TCP (mm / deg, rotation vector): "
            f"X={translation[0]:.1f} Y={translation[1]:.1f} Z={translation[2]:.1f}   "
            f"Rx={rotation_deg[0]:.1f} Ry={rotation_deg[1]:.1f} Rz={rotation_deg[2]:.1f}"
        )
        camera_frame_position = (self._camera_to_base_transform().inv() * pose).translation
        self.pointcloud_viewer.show_current_position(camera_frame_position)

    def _on_joints_updated(self, joint_positions: List[float]) -> None:
        degrees = np.degrees(joint_positions)
        joint_text = "  ".join(f"J{i + 1}={value:.1f}°" for i, value in enumerate(degrees))
        self.joints_monitor_label.setText(f"Joints (deg): {joint_text}")

    # --- Home / start posture --------------------------------------------------------------

    def _update_home_status_label(self) -> None:
        home_joints = self.config.home_joints()
        if home_joints is None:
            self.home_status_label.setText("No home set")
        else:
            degrees = np.degrees(home_joints)
            self.home_status_label.setText("Home: [" + ", ".join(f"{v:.1f}°" for v in degrees) + "]")

    def on_set_home_clicked(self) -> None:
        robot_control = self.robot_connection_widget.robot_control
        if robot_control is None:
            QMessageBox.warning(self, "Set Home", "Robot is not connected.")
            return
        self.config.set_home_joints(robot_control.get_joint_positions())
        self._update_home_status_label()

    def on_move_home_clicked(self) -> None:
        robot_control = self.robot_connection_widget.robot_control
        if robot_control is None:
            QMessageBox.warning(self, "Move To Home", "Robot is not connected.")
            return
        home_joints = self.config.home_joints()
        if home_joints is None:
            QMessageBox.warning(self, "Move To Home", "No home position has been set.")
            return

        self.move_home_button.setEnabled(False)
        self.execution_status_label.setText("Moving to home...")
        self.robot_connection_widget.pause_polling()
        QApplication.processEvents()

        result_queue: "queue.Queue" = queue.Queue()

        def _run() -> None:
            try:
                self._move_with_monitoring(
                    robot_control, robot_control.move_to_joints, home_joints,
                    speed=DEFAULT_JOINT_SPEED, acceleration=DEFAULT_JOINT_ACCELERATION,
                )
                result_queue.put((True, "Moved to the home position."))
            except Exception as ex:  # pylint: disable=broad-except
                result_queue.put((False, str(ex)))

        thread = threading.Thread(target=_run)
        thread.start()
        while thread.is_alive():
            QApplication.processEvents()
        success, message = result_queue.get()

        self.robot_connection_widget.resume_polling()
        self.move_home_button.setEnabled(True)
        if success:
            self.execution_status_label.setText("Done")
            QMessageBox.information(self, "Move To Home Complete", message)
        else:
            self.execution_status_label.setText(f"Failed: {message}")
            QMessageBox.warning(self, "Move To Home Failed", message)

    # --- Execution -------------------------------------------------------------------------

    def _approach_offset(self) -> TransformationMatrix:
        # Tool Z points into the surface (see geometry.waypoint_builder docstring) - moving
        # along local -Z backs the tool away from the surface, towards the camera side.
        return TransformationMatrix(translation=np.array([0.0, 0.0, -self.approach_spinbox.value()], dtype=np.float32))

    def _move_with_monitoring(self, robot_control, move_fn, *args, **kwargs) -> None:
        """Run one move asynchronously and poll pose/joints until it finishes, emitting
        live_pose_updated/live_joints_updated along the way (see class docstring)."""
        kwargs["asynchronous"] = True
        move_fn(*args, **kwargs)
        time.sleep(MONITOR_POLL_INTERVAL_S)
        while robot_control.is_moving():
            self.live_pose_updated.emit(robot_control.get_pose().pose)
            time.sleep(MONITOR_POLL_INTERVAL_S)
        self.live_pose_updated.emit(robot_control.get_pose().pose)
        self.live_joints_updated.emit(robot_control.get_joint_positions())

    def _run_execution(self, result_queue: "queue.Queue") -> None:
        robot_control = self.robot_connection_widget.robot_control
        assert robot_control is not None
        try:
            approach_offset = self._approach_offset()
            speed_m_s = self.speed_spinbox.value() / 1000.0
            acceleration = self.acceleration_spinbox.value()
            blend_m = self.blend_spinbox.value() / 1000.0
            home_joints = self.config.home_joints()

            approach_pose = self.waypoints[0] * approach_offset
            retreat_pose = self.waypoints[-1] * approach_offset

            if home_joints is not None:
                self._move_with_monitoring(
                    robot_control, robot_control.move_to_joints, home_joints,
                    speed=DEFAULT_JOINT_SPEED, acceleration=DEFAULT_JOINT_ACCELERATION,
                )
            self._move_with_monitoring(
                robot_control, robot_control.move_j, robot_control.get_custom_target(approach_pose),
                speed=DEFAULT_JOINT_SPEED, acceleration=DEFAULT_JOINT_ACCELERATION,
            )
            self._move_with_monitoring(
                robot_control, robot_control.move_path, self.waypoints,
                speed=speed_m_s, acceleration=acceleration, blend_radius=blend_m,
            )
            self._move_with_monitoring(
                robot_control, robot_control.move_j, robot_control.get_custom_target(retreat_pose),
                speed=DEFAULT_JOINT_SPEED, acceleration=DEFAULT_JOINT_ACCELERATION,
            )
            if home_joints is not None:
                self._move_with_monitoring(
                    robot_control, robot_control.move_to_joints, home_joints,
                    speed=DEFAULT_JOINT_SPEED, acceleration=DEFAULT_JOINT_ACCELERATION,
                )
            result_queue.put((True, f"Done - {len(self.waypoints)} waypoints"))
        except Exception as ex:  # pylint: disable=broad-except
            result_queue.put((False, str(ex)))

    def _min_waypoint_spacing_mm(self) -> float:
        if len(self.waypoints) < 2:
            return float("inf")
        return min(
            float(np.linalg.norm(b.translation - a.translation))
            for a, b in zip(self.waypoints[:-1], self.waypoints[1:])
        )

    def on_execute_clicked(self) -> None:
        if len(self.waypoints) < 2:
            return
        robot_control = self.robot_connection_widget.robot_control
        if robot_control is None:
            QMessageBox.warning(self, "Execute", "Robot is not connected.")
            return

        # A blend radius that overlaps into the neighboring segment (i.e. more than half the
        # gap to the next waypoint) confuses the controller's blended-path planner - this is
        # the likely cause of "speeds up partway through and rushes to the end" behavior.
        min_spacing = self._min_waypoint_spacing_mm()
        blend_mm = self.blend_spinbox.value()
        if blend_mm > min_spacing / 2:
            QMessageBox.warning(
                self,
                "Blend Radius Too Large",
                f"The blend radius ({blend_mm:.1f} mm) exceeds half the minimum waypoint spacing "
                f"({min_spacing:.1f} mm).\n"
                "Running like this makes adjacent blend zones overlap, so the robot skips\n"
                "intermediate waypoints, speeds up abruptly and rushes to the end.\n\n"
                f"Reduce the blend radius to {min_spacing / 2:.1f} mm or less, or regenerate the "
                "waypoints with a wider spacing.",
            )
            return

        has_home = self.config.home_joints() is not None
        sequence = (
            "home → approach (moveJ) → line tracing → retreat (moveJ) → home"
            if has_home
            else "approach (moveJ) → line tracing → retreat (moveJ)  (no home set)"
        )
        confirm = QMessageBox.question(
            self,
            "Confirm Execution",
            f"Sequence: {sequence}\n"
            f"{len(self.waypoints)} waypoints, speed {self.speed_spinbox.value():.0f} mm/s, "
            f"acceleration {self.acceleration_spinbox.value():.2f} m/s², "
            f"approach/retreat {self.approach_spinbox.value():.0f} mm\n\n"
            "The gripper tip is sharp - keep clear of the robot and be ready to hit the "
            "emergency stop.\n"
            "Continue?",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No,
        )
        if confirm != QMessageBox.Yes:
            return

        self.execute_button.setEnabled(False)
        self.stop_button.setEnabled(True)
        self.execution_status_label.setText("Running...")
        self.robot_connection_widget.pause_polling()
        QApplication.processEvents()

        result_queue: "queue.Queue" = queue.Queue()
        thread = threading.Thread(target=self._run_execution, args=(result_queue,))
        thread.start()
        while thread.is_alive():
            QApplication.processEvents()
        success, message = result_queue.get()

        self.robot_connection_widget.resume_polling()
        self.stop_button.setEnabled(False)
        self._update_execute_button_state()
        if success:
            self.execution_status_label.setText("Done")
            QMessageBox.information(self, "Execution Complete", message)
        else:
            self.execution_status_label.setText(f"Failed: {message}")
            QMessageBox.warning(self, "Execution Failed", message)

    def on_stop_clicked(self) -> None:
        robot_control = self.robot_connection_widget.robot_control
        if robot_control is not None:
            robot_control.stop()
        self.execution_status_label.setText("Stop requested")

    def get_point_cloud_xyz(self) -> Optional[np.ndarray]:
        return self.point_cloud_xyz

    def get_line_points_px(self):
        return self.image_viewer.get_line_points()

    def get_waypoints(self) -> List[TransformationMatrix]:
        return self.waypoints