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
from line_tracing_gui.widgets.calibration_panel import CalibrationPanel
from line_tracing_gui.widgets.camera_panel import CameraPanel
from line_tracing_gui.widgets.drawable_image_viewer import DrawableImageViewer
from line_tracing_gui.widgets.pointcloud_viewer_widget import PointCloudViewerWidget
from line_tracing_gui.widgets.robot_connection_widget import RobotConnectionWidget

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
        self.capture_button = QPushButton("캡처 (2D+3D)")
        self.capture_button.clicked.connect(self.on_capture_clicked)

        self.draw_mode_button = QPushButton("라인 그리기")
        self.draw_mode_button.setCheckable(True)
        self.draw_mode_button.setEnabled(False)
        self.draw_mode_button.toggled.connect(self.on_draw_mode_toggled)

        self.clear_button = QPushButton("지우기")
        self.clear_button.setEnabled(False)
        self.clear_button.clicked.connect(self.on_clear_clicked)

        self.undo_button = QPushButton("실행 취소")
        self.undo_button.setEnabled(False)
        self.undo_button.clicked.connect(self.on_undo_clicked)

        self.status_label_2d = QLabel("카메라를 연결하고 캡처해주세요.")

        self.image_viewer = DrawableImageViewer()
        self.image_viewer.setMinimumSize(400, 400)

        buttons_layout = QHBoxLayout()
        buttons_layout.addWidget(self.capture_button)
        buttons_layout.addWidget(self.draw_mode_button)
        buttons_layout.addWidget(self.clear_button)
        buttons_layout.addWidget(self.undo_button)

        group_box = QGroupBox("2D 캡처 / 라인 그리기")
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
            "웨이포인트 사이의 목표 간격 (픽셀 공간에서 근사 - 표면까지 거리에 따라 실제 mm 간격은 달라질 수 있음)"
        )

        self.generate_waypoints_button = QPushButton("웨이포인트 생성")
        self.generate_waypoints_button.setEnabled(False)
        self.generate_waypoints_button.clicked.connect(self.on_generate_waypoints_clicked)

        self.status_label_3d = QLabel("캡처 후 라인을 그리면 여기에 점 구름이 표시됩니다.")

        self.pointcloud_viewer = PointCloudViewerWidget()
        self.pointcloud_viewer.setMinimumSize(400, 400)

        controls_form = QFormLayout()
        spacing_row = QHBoxLayout()
        spacing_row.addWidget(self.spacing_spinbox)
        spacing_row.addWidget(self.generate_waypoints_button)
        controls_form.addRow("웨이포인트 간격", spacing_row)

        group_box = QGroupBox("3D 포인트 클라우드 / 웨이포인트 (노란 점 = 로봇 현재 위치)")
        group_layout = QVBoxLayout()
        group_layout.addLayout(controls_form)
        group_layout.addWidget(self.status_label_3d)
        group_layout.addWidget(self.pointcloud_viewer)
        group_box.setLayout(group_layout)
        return group_box

    def _build_execution_panel(self) -> QWidget:
        self.set_home_button = QPushButton("현재 위치를 홈으로 설정")
        self.set_home_button.clicked.connect(self.on_set_home_clicked)
        self.move_home_button = QPushButton("홈으로 이동")
        self.move_home_button.clicked.connect(self.on_move_home_clicked)
        self.home_status_label = QLabel("")

        self.approach_spinbox = QDoubleSpinBox()
        self.approach_spinbox.setRange(APPROACH_MIN_MM, APPROACH_MAX_MM)
        self.approach_spinbox.setValue(APPROACH_DEFAULT_MM)
        self.approach_spinbox.setSuffix(" mm")
        self.approach_spinbox.setToolTip(
            "라인의 시작/끝 지점에서 그리퍼가 접근을 시작하고 후퇴를 마치는, 표면에서 떨어진 거리 "
            "(웨이포인트의 로컬 -Z 방향, 즉 찌르는 방향의 반대쪽)"
        )

        self.speed_spinbox = QDoubleSpinBox()
        self.speed_spinbox.setRange(EXEC_SPEED_MIN_MM_S, EXEC_SPEED_MAX_MM_S)
        self.speed_spinbox.setValue(EXEC_SPEED_DEFAULT_MM_S)
        self.speed_spinbox.setSuffix(" mm/s")
        self.speed_spinbox.setToolTip("웨이포인트를 따라가는 구간(move_path)의 속도. 접근/후퇴/홈 이동은 moveJ 기본값을 씀.")

        self.acceleration_spinbox = QDoubleSpinBox()
        self.acceleration_spinbox.setRange(EXEC_ACCELERATION_MIN, EXEC_ACCELERATION_MAX)
        self.acceleration_spinbox.setValue(EXEC_ACCELERATION_DEFAULT)
        self.acceleration_spinbox.setSuffix(" m/s²")

        self.blend_spinbox = QDoubleSpinBox()
        self.blend_spinbox.setRange(EXEC_BLEND_MIN_MM, EXEC_BLEND_MAX_MM)
        self.blend_spinbox.setValue(EXEC_BLEND_DEFAULT_MM)
        self.blend_spinbox.setSuffix(" mm")
        self.blend_spinbox.setToolTip(
            "인접 웨이포인트 간격의 절반보다 크면 실행이 거부됩니다 - 블렌드 구간이 겹치면 "
            "로봇이 중간 지점을 건너뛰고 갑자기 빨라지는 문제가 생깁니다."
        )

        self.execute_button = QPushButton("실행")
        self.execute_button.setEnabled(False)
        self.execute_button.clicked.connect(self.on_execute_clicked)

        self.stop_button = QPushButton("정지")
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
        form.addRow("홈/시작 자세", home_row)
        form.addRow("접근/후퇴", self.approach_spinbox)
        form.addRow("속도", self.speed_spinbox)
        form.addRow("가속도", self.acceleration_spinbox)
        form.addRow("블렌드", self.blend_spinbox)

        # Execute/Stop sit to the right of the settings form, in the wide empty space that
        # would otherwise be left blank next to the narrow (FieldsStayAtSizeHint) fields.
        buttons_column = QVBoxLayout()
        buttons_column.addWidget(self.execute_button)
        buttons_column.addWidget(self.stop_button)
        buttons_column.addWidget(self.execution_status_label)
        buttons_column.addStretch(1)

        form_and_buttons_row = QHBoxLayout()
        form_and_buttons_row.addLayout(form, 0)
        form_and_buttons_row.addLayout(buttons_column, 1)

        group_box = QGroupBox("로봇 실행")
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
            QMessageBox.warning(self, "캡처", "먼저 카메라를 연결해주세요.")
            return

        self.camera_panel.stop_live_preview()
        self.camera_panel.update_capture_button_state()

        try:
            with camera.capture(self._capture_settings()) as frame:
                point_cloud = frame.point_cloud()
                self.point_cloud_xyz = point_cloud.copy_data("xyz")
                rgba = point_cloud.copy_data("rgba_srgb")
        except RuntimeError as ex:
            QMessageBox.critical(self, "캡처 실패", str(ex))
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
        self.status_label_2d.setText(f"캡처 완료 ({rgba.shape[1]}x{rgba.shape[0]}) - 라인을 그려주세요.")
        self.status_label_3d.setText("점 구름 표시됨 - 라인을 그리고 웨이포인트를 생성해주세요.")
        self._update_execute_button_state()

    def on_draw_mode_toggled(self, checked: bool) -> None:
        self.image_viewer.set_draw_mode(checked)
        self.draw_mode_button.setText("그리기 중지" if checked else "라인 그리기")

    def on_clear_clicked(self) -> None:
        self.image_viewer.clear_line()

    def on_undo_clicked(self) -> None:
        self.image_viewer.undo_last_stroke()

    # --- Waypoint generation --------------------------------------------------------------

    def on_generate_waypoints_clicked(self) -> None:
        if self.point_cloud_xyz is None:
            QMessageBox.warning(self, "웨이포인트 생성", "먼저 캡처해주세요.")
            return
        line_points_px = self.image_viewer.get_line_points()
        if len(line_points_px) < 2:
            QMessageBox.warning(self, "웨이포인트 생성", "먼저 라인을 그려주세요.")
            return

        hand_eye_transform = self.calibration_panel.get_hand_eye_transform()
        eye_in_hand = self.calibration_panel.get_eye_in_hand()
        if eye_in_hand:
            QMessageBox.warning(
                self,
                "웨이포인트 생성",
                "Eye-in-hand는 캡처 시점의 로봇 pose 기록이 아직 구현되지 않았습니다. "
                "캘리브레이션 탭에서 Eye-to-hand를 선택해주세요.",
            )
            return

        try:
            result = build_waypoints(
                line_points_px,
                self.point_cloud_xyz,
                hand_eye_transform,
                eye_in_hand=eye_in_hand,
                sample_spacing_mm=self.spacing_spinbox.value(),
            )
        except ValueError as ex:
            QMessageBox.warning(self, "웨이포인트 생성", str(ex))
            return

        if len(result.waypoints) < 2:
            QMessageBox.warning(
                self,
                "웨이포인트 생성",
                f"유효한 웨이포인트가 {len(result.waypoints)}개뿐입니다 (건너뜀 {result.skipped_pixel_count}개). "
                "라인이 유효한 3D 영역 위에 있는지 확인해주세요.",
            )
            return

        self.waypoints = result.waypoints
        self.status_label_3d.setText(
            f"웨이포인트 {len(result.waypoints)}개 생성 "
            f"(건너뜀 {result.skipped_pixel_count}개, 간격 좁아서 병합 {result.merged_close_waypoint_count}개)"
        )
        self._auto_adjust_blend_spinbox()

        # The 3D view is drawn in camera frame (matching the point cloud); undo the
        # hand-eye transform to get each waypoint back from base frame for display.
        camera_frame_waypoints = [hand_eye_transform.inv() * waypoint for waypoint in self.waypoints]
        self.pointcloud_viewer.show_waypoints(camera_frame_waypoints)
        self._update_execute_button_state()

    def _update_execute_button_state(self) -> None:
        has_waypoints = len(self.waypoints) >= 2
        self.execute_button.setEnabled(has_waypoints and self.robot_connection_widget.connected)

    def _auto_adjust_blend_spinbox(self) -> None:
        """Set the blend spinbox to a value that's safely valid for the just-generated
        waypoints, so the user isn't stuck manually retuning it against the "블렌드 반경
        확인 필요" check in on_execute_clicked every time the line's curvature changes the
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
        hand_eye_transform = self.calibration_panel.get_hand_eye_transform()
        camera_frame_position = (hand_eye_transform.inv() * pose).translation
        self.pointcloud_viewer.show_current_position(camera_frame_position)

    def _on_joints_updated(self, joint_positions: List[float]) -> None:
        degrees = np.degrees(joint_positions)
        joint_text = "  ".join(f"J{i + 1}={value:.1f}°" for i, value in enumerate(degrees))
        self.joints_monitor_label.setText(f"Joints (deg): {joint_text}")

    # --- Home / start posture --------------------------------------------------------------

    def _update_home_status_label(self) -> None:
        home_joints = self.config.home_joints()
        if home_joints is None:
            self.home_status_label.setText("홈 설정 안 됨")
        else:
            degrees = np.degrees(home_joints)
            self.home_status_label.setText("홈 설정됨: [" + ", ".join(f"{v:.1f}°" for v in degrees) + "]")

    def on_set_home_clicked(self) -> None:
        robot_control = self.robot_connection_widget.robot_control
        if robot_control is None:
            QMessageBox.warning(self, "홈 설정", "로봇이 연결되어 있지 않습니다.")
            return
        self.config.set_home_joints(robot_control.get_joint_positions())
        self._update_home_status_label()

    def on_move_home_clicked(self) -> None:
        robot_control = self.robot_connection_widget.robot_control
        if robot_control is None:
            QMessageBox.warning(self, "홈으로 이동", "로봇이 연결되어 있지 않습니다.")
            return
        home_joints = self.config.home_joints()
        if home_joints is None:
            QMessageBox.warning(self, "홈으로 이동", "홈 위치가 설정되어 있지 않습니다.")
            return

        self.move_home_button.setEnabled(False)
        self.execution_status_label.setText("홈으로 이동 중...")
        self.robot_connection_widget.pause_polling()
        QApplication.processEvents()

        result_queue: "queue.Queue" = queue.Queue()

        def _run() -> None:
            try:
                self._move_with_monitoring(
                    robot_control, robot_control.move_to_joints, home_joints,
                    speed=DEFAULT_JOINT_SPEED, acceleration=DEFAULT_JOINT_ACCELERATION,
                )
                result_queue.put((True, "홈 위치로 이동했습니다."))
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
            self.execution_status_label.setText("완료")
            QMessageBox.information(self, "홈 이동 완료", message)
        else:
            self.execution_status_label.setText(f"실패: {message}")
            QMessageBox.warning(self, "홈 이동 실패", message)

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
            result_queue.put((True, f"완료 - 웨이포인트 {len(self.waypoints)}개"))
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
            QMessageBox.warning(self, "실행", "로봇이 연결되어 있지 않습니다.")
            return

        # A blend radius that overlaps into the neighboring segment (i.e. more than half the
        # gap to the next waypoint) confuses the controller's blended-path planner - this is
        # the likely cause of "speeds up partway through and rushes to the end" behavior.
        min_spacing = self._min_waypoint_spacing_mm()
        blend_mm = self.blend_spinbox.value()
        if blend_mm > min_spacing / 2:
            QMessageBox.warning(
                self,
                "블렌드 반경 확인 필요",
                f"블렌드 반경({blend_mm:.1f}mm)이 웨이포인트 최소 간격({min_spacing:.1f}mm)의 절반보다 큽니다.\n"
                "이 상태로 실행하면 인접 블렌드 구간이 겹쳐서, 중간 웨이포인트를 건너뛰고 갑자기 빨라지며 "
                "끝까지 한 번에 움직이는 문제가 생길 수 있습니다 (실제로 겪으신 증상과 일치).\n\n"
                f"블렌드 반경을 {min_spacing / 2:.1f}mm 이하로 줄이거나, 웨이포인트 간격을 넓혀서 다시 생성해주세요.",
            )
            return

        has_home = self.config.home_joints() is not None
        sequence = "홈 → 접근(moveJ) → 라인 트레이싱 → 후퇴(moveJ) → 홈" if has_home else "접근(moveJ) → 라인 트레이싱 → 후퇴(moveJ) (홈 미설정)"
        confirm = QMessageBox.question(
            self,
            "실행 확인",
            f"순서: {sequence}\n"
            f"웨이포인트 {len(self.waypoints)}개, 속도 {self.speed_spinbox.value():.0f} mm/s, "
            f"가속도 {self.acceleration_spinbox.value():.2f} m/s², 접근/후퇴 거리 {self.approach_spinbox.value():.0f} mm\n\n"
            "그리퍼가 뾰족합니다 - 로봇 주변에 사람이 없는지, 즉시 비상정지 가능한 상태인지 확인해주세요.\n"
            "계속하시겠습니까?",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No,
        )
        if confirm != QMessageBox.Yes:
            return

        self.execute_button.setEnabled(False)
        self.stop_button.setEnabled(True)
        self.execution_status_label.setText("실행 중...")
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
            self.execution_status_label.setText("완료")
            QMessageBox.information(self, "실행 완료", message)
        else:
            self.execution_status_label.setText(f"실패: {message}")
            QMessageBox.warning(self, "실행 실패", message)

    def on_stop_clicked(self) -> None:
        robot_control = self.robot_connection_widget.robot_control
        if robot_control is not None:
            robot_control.stop()
        self.execution_status_label.setText("정지 요청됨")

    def get_point_cloud_xyz(self) -> Optional[np.ndarray]:
        return self.point_cloud_xyz

    def get_line_points_px(self):
        return self.image_viewer.get_line_points()

    def get_waypoints(self) -> List[TransformationMatrix]:
        return self.waypoints