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

from line_tracing_gui.config import POSE_REFERENCE_FLANGE, AppConfig
from line_tracing_gui.geometry.projection import (
    VERDICT_OK,
    VERDICT_UNCHECKED,
    VERDICT_UNREACHABLE,
    VERDICT_WARNING,
    build_projector_image,
)
from line_tracing_gui.geometry.waypoint_builder import DEFAULT_SAMPLE_SPACING_MM, build_paths
from line_tracing_gui.robot.reachability import ReachabilityReport, check_paths_reachability
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

# Just enough to lift the tip clear of the surface between strokes. Tracing letters means
# a great many of these hops, so every millimetre is paid for on every one of them.
APPROACH_DEFAULT_MM = 10.0
APPROACH_MIN_MM = 0.0
APPROACH_MAX_MM = 150.0

EXEC_SPEED_DEFAULT_MM_S = 30.0
EXEC_SPEED_MIN_MM_S = 1.0
EXEC_SPEED_MAX_MM_S = 200.0

EXEC_ACCELERATION_DEFAULT = 0.3
EXEC_ACCELERATION_MIN = 0.05
EXEC_ACCELERATION_MAX = 1.0

EXEC_BLEND_DEFAULT_MM = 1.0
EXEC_BLEND_MIN_MM = 0.0
EXEC_BLEND_MAX_MM = 20.0

MONITOR_POLL_INTERVAL_S = 0.05

# How far the robot may sit from the capture pose before the projection is treated as stale.
# Comfortably above the arm's own repeatability and pose-read noise, and far below anything
# that would be visible in the projected line.
PROJECTION_POSE_TOLERANCE_MM = 1.0
PROJECTION_POSE_TOLERANCE_DEG = 0.5


class TracePanel(QWidget):
    # Emitted from the execution worker thread (Qt auto-queues delivery to this widget's
    # thread) so the live TCP/joint monitors and the 3D "current position" marker can
    # update while a move is in progress.
    live_pose_updated = pyqtSignal(object)  # TransformationMatrix, base frame
    live_joints_updated = pyqtSignal(list)  # radians
    # Emitted from the reachability worker thread: (steps completed, total steps).
    reachability_progress = pyqtSignal(int, int)

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
        # One list of base-frame waypoints per drawn stroke, in drawing order. Strokes stay
        # separate all the way through - see geometry.waypoint_builder.build_paths.
        self.paths: List[List[TransformationMatrix]] = []
        # Verdicts for exactly the waypoints above; dropped whenever they are regenerated so
        # the 3D highlight and the projector colours can never describe a stale path.
        self.reachability_report: Optional[ReachabilityReport] = None
        # zivid.projection.ProjectedImage while the projector is on. Projection also stops by
        # itself when a capture starts on the same camera, so this is cleared there too.
        self._projected_image = None

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
        self.reachability_progress.connect(self._on_reachability_progress)
        # Losing the camera kills the projection with it, and re-gaining one makes Project
        # available again - both only show up through this signal.
        self.camera_panel.camera_connected.connect(self._on_camera_connected)

        self._update_home_status_label()

    def _build_2d_panel(self) -> QWidget:
        self.capture_button = QPushButton("Capture (2D+3D)")
        mark_as_accent(self.capture_button)
        self.capture_button.clicked.connect(self.on_capture_clicked)

        self.draw_mode_button = QPushButton("Draw Lines")
        self.draw_mode_button.setCheckable(True)
        self.draw_mode_button.setEnabled(False)
        self.draw_mode_button.setToolTip(
            "Draw on the captured image. Every press-drag-release is one line, and each one\n"
            "becomes its own robot path - the robot lifts off the surface between them rather\n"
            "than tracing from the end of one to the start of the next."
        )
        self.draw_mode_button.toggled.connect(self.on_draw_mode_toggled)

        self.clear_button = QPushButton("Clear")
        self.clear_button.setEnabled(False)
        self.clear_button.setToolTip("Remove every line drawn so far.")
        self.clear_button.clicked.connect(self.on_clear_clicked)

        self.undo_button = QPushButton("Undo")
        self.undo_button.setEnabled(False)
        self.undo_button.setToolTip("Remove the line drawn most recently.")
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

        group_box = QGroupBox("2D Capture / Draw Lines")
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
        self.generate_waypoints_button.setToolTip(
            "Turn every drawn line into its own robot path. Lines are never joined up:\n"
            "resampling across the gap between two of them would put waypoints on surface\n"
            "nobody drew on."
        )
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
            "retreat, at the first/last point of every line (along the waypoint's local -Z, i.e. the\n"
            "direction opposite to where the tip points).\n\n"
            "This is also how far the tip lifts between one line and the next, so with many short\n"
            "lines it is paid for many times over - keep it just clear of the surface."
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

        self.check_reachability_button = QPushButton("Check Reachability")
        self.check_reachability_button.setEnabled(False)
        self.check_reachability_button.setToolTip(
            "Ask the controller whether every pose in the sequence (home, approach, each waypoint,\n"
            "retreat) can actually be reached and stays inside the configured safety limits -\n"
            "without moving the robot. Execute runs this check by itself too; this button is for\n"
            "trying a line out before committing to it."
        )
        self.check_reachability_button.clicked.connect(self.on_check_reachability_clicked)

        self.project_button = QPushButton("Project")
        self.project_button.setCheckable(True)
        self.project_button.setEnabled(False)
        self.project_button.setToolTip(
            "Draw the waypoint line onto the real surface with the camera's projector, so the\n"
            "path can be checked against the physical part before the robot moves. This is the only\n"
            "check that catches a wrong camera-to-base chain - the 3D preview maps the waypoints\n"
            "back with the same transform used to build them, so the error cancels out there."
        )
        self.project_button.toggled.connect(self.on_project_toggled)

        self.execute_button = QPushButton("Execute")
        mark_as_accent(self.execute_button)
        self.execute_button.setEnabled(False)
        self.execute_button.clicked.connect(self.on_execute_clicked)

        self.stop_button = QPushButton("Stop")
        mark_as_danger(self.stop_button)
        for button in (self.check_reachability_button, self.project_button, self.execute_button, self.stop_button):
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
        buttons_column.addWidget(self.check_reachability_button)
        buttons_column.addWidget(self.project_button)
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

        # A capture on this camera stops any projection at the SDK level, so let go of the
        # handle here rather than leave the button latched over a projector that is already off.
        self._stop_projection()
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
        self.paths = []

        # Eye-in-hand turns a camera-frame point into a base-frame one through the robot's
        # pose at capture time, and the robot stood still for the capture we just took - so
        # its pose right now is that pose. Recording it here means nothing has to be entered
        # by hand, and the Calibration tab can send the robot back to it later.
        if self.calibration_panel.get_eye_in_hand():
            error_message = self.calibration_panel.record_capture_pose()
            if error_message is not None:
                QMessageBox.warning(
                    self,
                    "Capture",
                    f"Captured, but the robot pose at capture time could not be recorded:\n{error_message}\n\n"
                    "Eye-In-Hand needs it to generate waypoints - connect the robot and capture again,\n"
                    "or set it under Robot Capture Pose in the Calibration tab.",
                )

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

    def _on_camera_connected(self, connected: bool) -> None:
        if not connected:
            self._stop_projection()
        self._update_execute_button_state()

    def on_draw_mode_toggled(self, checked: bool) -> None:
        self.image_viewer.set_draw_mode(checked)
        self.draw_mode_button.setText("Stop Drawing" if checked else "Draw Lines")

    def on_clear_clicked(self) -> None:
        self.image_viewer.clear_line()

    def on_undo_clicked(self) -> None:
        self.image_viewer.undo_last_stroke()

    # --- Waypoint generation --------------------------------------------------------------

    def on_generate_waypoints_clicked(self) -> None:
        if self.point_cloud_xyz is None:
            QMessageBox.warning(self, "Generate Waypoints", "Capture first.")
            return
        strokes_px = self.image_viewer.get_strokes()
        if not strokes_px:
            QMessageBox.warning(self, "Generate Waypoints", "Draw at least one line first.")
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
            result = build_paths(
                strokes_px,
                self.point_cloud_xyz,
                hand_eye_transform,
                eye_in_hand=eye_in_hand,
                robot_pose=capture_pose,
                sample_spacing_mm=self.spacing_spinbox.value(),
                reference_x_axis=self._reference_tool_x_axis(),
            )
        except ValueError as ex:
            QMessageBox.warning(self, "Generate Waypoints", str(ex))
            return

        if not result.paths:
            QMessageBox.warning(
                self,
                "Generate Waypoints",
                f"None of the {len(strokes_px)} line(s) drawn produced a traceable path "
                f"(skipped {result.skipped_pixel_count} point(s)).\n"
                "Check that the lines lie over valid 3D data.",
            )
            return

        self.paths = result.paths
        # Both the reachability verdicts and the projected line describe the previous path.
        self.reachability_report = None
        self._stop_projection()
        line_count = f"{len(result.paths)} lines" if len(result.paths) > 1 else "1 line"
        dropped = (
            f", dropped {result.dropped_stroke_count} line(s) with too little 3D data"
            if result.dropped_stroke_count
            else ""
        )
        self.status_label_3d.setText(
            f"Generated {result.waypoint_count} waypoints over {line_count} "
            f"(skipped {result.skipped_pixel_count}, merged {result.merged_close_waypoint_count} too close "
            f"together{dropped}) - {self._transform_chain_description()}"
        )
        self._auto_adjust_blend_spinbox()

        # The 3D view is drawn in camera frame (matching the point cloud); undo the
        # camera-to-base transform to get each waypoint back from base frame for display.
        camera_frame_transform = self._camera_to_base_transform().inv()
        self.pointcloud_viewer.show_waypoints(
            [[camera_frame_transform * waypoint for waypoint in path] for path in self.paths]
        )
        self._update_execute_button_state()

    def _reference_tool_x_axis(self) -> Optional[np.ndarray]:
        """Base-frame direction to keep the tool X axis near, so the wrist barely turns.

        The tool X axis the robot holds at the capture pose. Every path is seeded from this
        one direction (see geometry.waypoint_builder), which matters most when tracing many
        short strokes: seeded from each stroke's own travel direction instead, the wrist
        unwinds and rewinds between them for no physical reason - the tip is rotationally
        symmetric, so the twist buys nothing.

        Returns None when there is nothing to read it from, and the builder then falls back
        to each line's travel direction.
        """
        robot_control = self.robot_connection_widget.robot_control
        pose = self.calibration_panel.get_capture_pose()
        if pose is not None:
            # get_capture_pose() is in whichever frame Pose Reference names. Tool axes are
            # the flange's turned by the controller's TCP offset, so a flange-referenced
            # pose needs that rotation put back on. On this rig the offset is a pure
            # translation and the two coincide - but that is a fact about the rig, not
            # something to build into the code.
            if self.calibration_panel.get_pose_reference() == POSE_REFERENCE_FLANGE and robot_control is not None:
                try:
                    pose = pose * robot_control.get_tcp_offset()
                except Exception:  # pylint: disable=broad-except
                    pass  # keep the flange orientation; it differs only if the offset rotates
            return np.asarray(pose.rotation.as_matrix()[:, 0], dtype=float)
        # Eye-to-hand has no capture pose; the arm's current orientation is the next best
        # thing to stay close to.
        if robot_control is not None:
            try:
                return np.asarray(robot_control.get_pose().pose.rotation.as_matrix()[:, 0], dtype=float)
            except Exception:  # pylint: disable=broad-except
                return None
        return None

    def _transform_chain_description(self) -> str:
        """Which camera -> base chain the waypoints were built through.

        Shown next to the waypoint count because a wrong Pose Reference produces waypoints
        that look perfectly fine everywhere in this GUI - the 3D preview maps them back with
        the inverse of the very transform used to build them, so the error cancels out. The
        only place it shows is the robot itself. Keeping the chain on screen makes a
        mismatch visible where the numbers already are.
        """
        if not self.calibration_panel.get_eye_in_hand():
            return "eye-to-hand: hand-eye transform only"
        return f"eye-in-hand: capture pose ({self.calibration_panel.get_pose_reference()}) x hand-eye transform"

    def _camera_to_base_transform(self) -> TransformationMatrix:
        """Camera frame -> robot base frame, matching what build_waypoints applies.

        Eye-to-hand is the hand-eye transform alone; eye-in-hand additionally goes
        through the robot's pose at capture time. Falls back to the hand-eye transform
        when that pose is not set, so the 3D preview still draws something sensible.

        The eye-in-hand test is not redundant: a capture pose recorded during an
        eye-in-hand session stays set after switching the mount type to eye-to-hand, and
        build_waypoints ignores it there. Without the test this would keep multiplying it
        in, so the 3D preview and the projection would describe a path the robot is not
        going to follow.
        """
        hand_eye_transform = self.calibration_panel.get_hand_eye_transform()
        if not self.calibration_panel.get_eye_in_hand():
            return hand_eye_transform
        capture_pose = self.calibration_panel.get_capture_pose()
        if capture_pose is None:
            return hand_eye_transform
        return capture_pose * hand_eye_transform

    def _total_waypoint_count(self) -> int:
        return sum(len(path) for path in self.paths)

    def _update_execute_button_state(self) -> None:
        # The reachability check needs exactly what execution needs - waypoints and a live
        # robot to ask - so the two buttons share one enabling rule.
        has_waypoints = bool(self.paths)
        ready = has_waypoints and self.robot_connection_widget.connected
        self.execute_button.setEnabled(ready)
        self.check_reachability_button.setEnabled(ready)
        # Projection needs the camera, not the robot: it is the camera's own projector that
        # draws, and eye-to-hand does not involve the robot's pose at all. Eye-in-hand still
        # wants a capture pose, but that is recorded at capture time, not read live.
        camera = self.camera_panel.camera
        self.project_button.setEnabled(
            has_waypoints and camera is not None and camera.state.connected
        )

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

    def _approach_retreat_poses(self) -> "tuple[List[TransformationMatrix], List[TransformationMatrix]]":
        """Standoff poses either side of every traced line, one pair per path.

        Each path gets its own pair because the robot has to lift off the surface to travel
        from one line to the next - moving across with the tip still down would drag it over
        the part. Shared by the reachability check and by execution so the check can never
        vet a different set than the one that actually runs.
        """
        approach_offset = self._approach_offset()
        approaches = [path[0] * approach_offset for path in self.paths]
        retreats = [path[-1] * approach_offset for path in self.paths]
        return approaches, retreats

    # --- Reachability check ----------------------------------------------------------------

    def _on_reachability_progress(self, completed: int, total: int) -> None:
        self.execution_status_label.setText(f"Checking... {completed}/{total}")

    def _run_reachability_check(self, result_queue: "queue.Queue", **check_kwargs) -> None:
        """Worker body. Everything it needs is passed in - it must not read widgets, which
        belong to the GUI thread. It must also always put exactly one report on the queue:
        the caller blocks on that queue, so a worker that dies silently would hang the GUI.
        """
        try:
            report = check_paths_reachability(
                progress_callback=lambda completed, total: self.reachability_progress.emit(completed, total),
                **check_kwargs,
            )
        except Exception as ex:  # pylint: disable=broad-except
            report = ReachabilityReport(error=str(ex))
        result_queue.put(report)

    def _perform_reachability_check(
        self,
        approach_poses: List[TransformationMatrix],
        retreat_poses: List[TransformationMatrix],
    ) -> Optional[ReachabilityReport]:
        """Run the check on a worker thread, keeping the UI responsive, and paint the result
        onto the 3D view. Returns None only if the robot is not connected.

        The approach/retreat poses are passed in rather than derived here so the caller can
        hand the very same objects to execution - a check that vetted a different set than the
        one that runs would invite trust it has not earned.
        """
        robot_control = self.robot_connection_widget.robot_control
        if robot_control is None:
            QMessageBox.warning(self, "Check Reachability", "Robot is not connected.")
            return None

        self.check_reachability_button.setEnabled(False)
        self.execute_button.setEnabled(False)
        self.execution_status_label.setText("Checking...")
        # Polling and the check both talk RTDE; pausing keeps the two off each other's backs
        # for the second or so the check takes, exactly as execution does.
        self.robot_connection_widget.pause_polling()
        QApplication.processEvents()

        # Read every input here, on the GUI thread, and hand them over as plain values.
        check_kwargs = {
            "robot_control": robot_control,
            "paths": [list(path) for path in self.paths],
            "approach_poses": list(approach_poses),
            "retreat_poses": list(retreat_poses),
            "home_joints": self.config.home_joints(),
        }

        result_queue: "queue.Queue" = queue.Queue()
        thread = threading.Thread(
            target=self._run_reachability_check, args=(result_queue,), kwargs=check_kwargs
        )
        thread.start()
        while thread.is_alive():
            QApplication.processEvents()
        report: ReachabilityReport = result_queue.get()

        self.robot_connection_widget.resume_polling()
        self.reachability_report = report
        self._update_execute_button_state()
        self.pointcloud_viewer.show_reachability(
            report.failed_waypoint_indices(), report.warned_waypoint_indices()
        )
        self.execution_status_label.setText(report.summary())
        return report

    def on_check_reachability_clicked(self) -> None:
        if not self.paths:
            return
        report = self._perform_reachability_check(*self._approach_retreat_poses())
        if report is None:
            return
        if report.error is not None:
            QMessageBox.warning(self, "Check Reachability", report.summary())
            return
        if report.ok and not report.warnings:
            QMessageBox.information(self, "Check Reachability", report.summary())
            return
        QMessageBox.warning(
            self,
            "Check Reachability",
            f"{report.summary()}\n\n{report.details()}\n\n"
            "Unreachable points are marked red in the 3D view, warnings amber. Redraw the line "
            "clear of them, or move the robot/camera so the surface sits better inside the "
            "working area, then generate the waypoints again.",
        )

    # --- Projector overlay -----------------------------------------------------------------

    def _waypoint_verdicts(self) -> List[List[str]]:
        """One projector colour per waypoint, from the last reachability check.

        Shaped like self.paths: one list of verdicts per path. The report is dropped whenever
        the paths are regenerated, so it either describes exactly these paths or is absent -
        there is no stale-index case to guard against here.
        """
        report = self.reachability_report
        if report is None or report.error is not None:
            return [[VERDICT_UNCHECKED] * len(path) for path in self.paths]
        unreachable = set(report.failed_waypoint_indices())
        warned = set(report.warned_waypoint_indices())
        verdicts = []
        for path_index, path in enumerate(self.paths):
            path_verdicts = []
            for waypoint_index in range(len(path)):
                key = (path_index, waypoint_index)
                if key in unreachable:
                    path_verdicts.append(VERDICT_UNREACHABLE)
                elif key in warned:
                    path_verdicts.append(VERDICT_WARNING)
                else:
                    path_verdicts.append(VERDICT_OK)
            verdicts.append(path_verdicts)
        return verdicts

    @staticmethod
    def _pose_deviation(first: TransformationMatrix, second: TransformationMatrix) -> "tuple[float, float]":
        """(millimetres, degrees) between two poses."""
        translation = float(
            np.linalg.norm(
                np.asarray(first.translation, dtype=float) - np.asarray(second.translation, dtype=float)
            )
        )
        rotation_deg = float(np.degrees((first.rotation.inv() * second.rotation).magnitude()))
        return translation, rotation_deg

    def _camera_to_base_for_projection(self) -> Optional[TransformationMatrix]:
        """Which camera -> base chain to map the waypoints back through for projection.

        Eye-to-hand has a fixed camera, so the robot's pose never enters into it. Eye-in-hand
        moves the camera with the arm: the capture pose is the honest default (it reproduces
        build_waypoints' own chain, so hand-eye cancels and only the pixel -> 3D
        correspondence is under test), but if the robot has since moved, the user is asked
        whether to go back or to project from where it stands - see geometry.projection for
        what each of those two actually proves.

        Returns None when the user cancels, or when the state needed is missing.
        """
        hand_eye_transform = self.calibration_panel.get_hand_eye_transform()
        if not self.calibration_panel.get_eye_in_hand():
            return hand_eye_transform

        capture_pose = self.calibration_panel.get_capture_pose()
        if capture_pose is None:
            QMessageBox.warning(
                self,
                "Project",
                "Eye-In-Hand needs the robot pose at capture time to know where the line is.\n"
                "Capture again, or read the pose under Robot Capture Pose in the Calibration tab.",
            )
            return None

        current_pose, error_message = self.calibration_panel.read_current_pose_in_reference_frame()
        if current_pose is None:
            # Without the current pose there is no way to tell whether the robot has moved.
            # Projecting from the capture pose is still the right guess, but say so.
            QMessageBox.warning(
                self,
                "Project",
                f"Could not read the robot's current pose ({error_message}).\n"
                "Projecting as if the robot were still standing where it was at capture time - "
                "if it has moved since, the line will land in the wrong place.",
            )
            return capture_pose * hand_eye_transform

        translation_mm, rotation_deg = self._pose_deviation(capture_pose, current_pose)
        if translation_mm <= PROJECTION_POSE_TOLERANCE_MM and rotation_deg <= PROJECTION_POSE_TOLERANCE_DEG:
            return capture_pose * hand_eye_transform

        message_box = QMessageBox(self)
        message_box.setIcon(QMessageBox.Question)
        message_box.setWindowTitle("Robot Has Moved Since The Capture")
        message_box.setText(
            f"The robot is {translation_mm:.1f} mm and {rotation_deg:.1f}° away from the pose it "
            "stood in when the frame was captured. The camera moved with it, so projecting the line "
            "as captured would put it in the wrong place."
        )
        message_box.setInformativeText(
            "Move To Capture Pose - go back first, then project. The plain check: only the "
            "pixel-to-3D correspondence is under test, so a line that lands wrong has one cause.\n\n"
            "Project From Here - re-map the line for where the robot stands now. This goes through "
            "the hand-eye transform, so a line that lands right also validates the hand-eye chain - "
            "but a line that lands wrong no longer says which of the two is at fault. Occlusion and "
            "the projector's focus are not accounted for, so keep the change of angle modest."
        )
        move_button = message_box.addButton("Move To Capture Pose", QMessageBox.AcceptRole)
        project_here_button = message_box.addButton("Project From Here", QMessageBox.ActionRole)
        message_box.addButton(QMessageBox.Cancel)
        message_box.setDefaultButton(move_button)
        message_box.exec_()

        clicked = message_box.clickedButton()
        if clicked is project_here_button:
            self.execution_status_label.setText("Projecting from the current pose (hand-eye chain included)")
            return current_pose * hand_eye_transform
        if clicked is not move_button:
            return None

        self.calibration_panel.on_move_to_capture_pose_clicked()
        # The move has its own confirmation and can be declined or fail, so re-read rather
        # than assume it happened.
        moved_pose, _ = self.calibration_panel.read_current_pose_in_reference_frame()
        if moved_pose is None:
            return None
        translation_mm, rotation_deg = self._pose_deviation(capture_pose, moved_pose)
        if translation_mm > PROJECTION_POSE_TOLERANCE_MM or rotation_deg > PROJECTION_POSE_TOLERANCE_DEG:
            QMessageBox.warning(
                self,
                "Project",
                f"The robot is still {translation_mm:.1f} mm and {rotation_deg:.1f}° from the capture "
                "pose, so nothing was projected.",
            )
            return None
        return capture_pose * hand_eye_transform

    def on_project_toggled(self, checked: bool) -> None:
        if not checked:
            self._stop_projection()
            return
        error_message = self._start_projection()
        if error_message is None:
            return
        if error_message:
            QMessageBox.warning(self, "Project", error_message)
        # Nothing is being projected, so the button must not stay latched. Block signals so
        # this does not come straight back in as an un-toggle.
        self.project_button.blockSignals(True)
        self.project_button.setChecked(False)
        self.project_button.blockSignals(False)

    def _start_projection(self) -> Optional[str]:
        """Show the waypoint line with the camera's projector.

        Returns None on success, or a message to show. An empty string means the user
        cancelled and has already been told everything they need to know.
        """
        camera = self.camera_panel.camera
        if camera is None or not camera.state.connected:
            return "Connect a camera first."
        if not self.paths:
            return "Generate waypoints first."

        camera_to_base_transform = self._camera_to_base_for_projection()
        if camera_to_base_transform is None:
            return ""

        # A capture on the same camera stops any projection, and the live preview captures
        # continuously - the two cannot both be running.
        self.camera_panel.stop_live_preview()
        self.camera_panel.update_capture_button_state()

        camera_frame_transform = camera_to_base_transform.inv()
        paths_camera_frame = [
            np.array(
                [(camera_frame_transform * waypoint).translation for waypoint in path], dtype=np.float32
            )
            for path in self.paths
        ]
        try:
            result = build_projector_image(camera, paths_camera_frame, verdicts=self._waypoint_verdicts())
            if result.drawn_waypoint_count == 0:
                return result.summary()
            self._projected_image = zivid.projection.show_image_bgra(camera, result.image_bgra)
        except Exception as ex:  # pylint: disable=broad-except
            self._projected_image = None
            return f"Projection failed: {ex}"

        self.execution_status_label.setText(result.summary())
        return None

    def _stop_projection(self) -> None:
        """Stop projecting, if we are. Safe to call at any time, including when the camera has
        already torn the projection down on its own (a capture does that)."""
        if self._projected_image is not None:
            try:
                self._projected_image.stop()
            except Exception:  # pylint: disable=broad-except
                pass  # already stopped by a capture, or the camera went away
            self._projected_image = None
        if self.project_button.isChecked():
            self.project_button.blockSignals(True)
            self.project_button.setChecked(False)
            self.project_button.blockSignals(False)

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

    def _run_execution(
        self,
        result_queue: "queue.Queue",
        robot_control,
        paths: List[List[TransformationMatrix]],
        approach_poses: List[TransformationMatrix],
        retreat_poses: List[TransformationMatrix],
        speed_m_s: float,
        acceleration: float,
        blend_m: float,
        home_joints: Optional[List[float]],
    ) -> None:
        """Worker body for the execution sequence.

        Everything it needs is passed in: Qt widgets belong to the GUI thread, and reading a
        spin box from a worker while the GUI thread spins in processEvents is what hung the
        reachability check under test. The same rule holds here.

        It must also always put exactly one result on the queue - the caller blocks on that
        queue, so a worker that dies before reporting would hang the GUI. Hence the try
        covering the whole body, argument validation included.
        """
        try:
            if robot_control is None:
                raise RuntimeError("Robot is not connected.")

            if home_joints is not None:
                self._move_with_monitoring(
                    robot_control, robot_control.move_to_joints, home_joints,
                    speed=DEFAULT_JOINT_SPEED, acceleration=DEFAULT_JOINT_ACCELERATION,
                )
            # Every path is approached and retreated on its own. Travelling from the end of
            # one line to the start of the next has to happen off the surface, or the tip
            # drags across the part on the way.
            for path, approach_pose, retreat_pose in zip(paths, approach_poses, retreat_poses):
                self._move_with_monitoring(
                    robot_control, robot_control.move_j, robot_control.get_custom_target(approach_pose),
                    speed=DEFAULT_JOINT_SPEED, acceleration=DEFAULT_JOINT_ACCELERATION,
                )
                self._move_with_monitoring(
                    robot_control, robot_control.move_path, path,
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
            traced = f"{len(paths)} lines" if len(paths) > 1 else "1 line"
            result_queue.put((True, f"Done - {traced}, {sum(len(path) for path in paths)} waypoints"))
        except Exception as ex:  # pylint: disable=broad-except
            result_queue.put((False, str(ex)))

    def _min_waypoint_spacing_mm(self) -> float:
        """Tightest gap between neighbouring waypoints anywhere in any path.

        Measured within paths only: the jump from the end of one line to the start of the
        next is not a blended segment - the robot retreats and approaches - so it has no say
        in how large a blend radius is safe.
        """
        gaps = [
            float(np.linalg.norm(b.translation - a.translation))
            for path in self.paths
            for a, b in zip(path[:-1], path[1:])
        ]
        return min(gaps) if gaps else float("inf")

    def on_execute_clicked(self) -> None:
        if not self.paths:
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

        # Vet the whole sequence before any of it runs. Failing mid-move_path leaves the
        # robot parked halfway along the line with the tip near the surface, which is both
        # the worst place to stop and the hardest to recover from - so an unreachable pose
        # blocks execution outright, while the heuristic configuration-flip warnings only
        # get surfaced in the confirmation below for the user to weigh.
        approach_poses, retreat_poses = self._approach_retreat_poses()
        report = self._perform_reachability_check(approach_poses, retreat_poses)
        if report is None:
            return
        if report.error is not None:
            QMessageBox.warning(self, "Execute", report.summary())
            return
        if report.failures:
            QMessageBox.warning(
                self,
                "Path Not Reachable",
                f"{report.summary()}\n\n{report.details()}\n\n"
                "Execution was not started. The offending waypoints are marked red in the 3D view "
                "(warnings amber) - redraw the line clear of them, or reposition the robot or the "
                "surface, then generate the waypoints again.",
            )
            return
        warning_note = (
            f"{report.summary()}\n{report.details()}\n\n"
            if report.warnings
            else ""
        )

        has_home = self.config.home_joints() is not None
        per_line = "approach (moveJ) → line tracing → retreat (moveJ)"
        if len(self.paths) > 1:
            per_line = f"[ {per_line} ] × {len(self.paths)} lines"
        sequence = f"home → {per_line} → home" if has_home else f"{per_line}  (no home set)"

        confirm = QMessageBox.question(
            self,
            "Confirm Execution",
            f"{warning_note}Sequence: {sequence}\n"
            f"{self._total_waypoint_count()} waypoints, speed {self.speed_spinbox.value():.0f} mm/s, "
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

        # The line is about to be traced for real, and on eye-in-hand the moving arm carries
        # the camera off the capture pose, so anything still projected would be misleading.
        self._stop_projection()
        self.execute_button.setEnabled(False)
        self.stop_button.setEnabled(True)
        self.execution_status_label.setText("Running...")
        self.robot_connection_widget.pause_polling()
        QApplication.processEvents()

        # Same rule as the reachability check: every widget value is read here, on the GUI
        # thread, and handed to the worker as a plain value.
        execution_kwargs = {
            "robot_control": robot_control,
            "paths": [list(path) for path in self.paths],
            "approach_poses": approach_poses,
            "retreat_poses": retreat_poses,
            "speed_m_s": self.speed_spinbox.value() / 1000.0,
            "acceleration": self.acceleration_spinbox.value(),
            "blend_m": self.blend_spinbox.value() / 1000.0,
            "home_joints": self.config.home_joints(),
        }

        result_queue: "queue.Queue" = queue.Queue()
        thread = threading.Thread(
            target=self._run_execution, args=(result_queue,), kwargs=execution_kwargs
        )
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

    def get_strokes_px(self):
        return self.image_viewer.get_strokes()

    def get_paths(self) -> List[List[TransformationMatrix]]:
        return self.paths