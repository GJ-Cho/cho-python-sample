"""
Calibration Board / ArUco Marker Pose Estimator  —  GUI Application

Run:
    python touch_pose_estimation_cal_board_marker.py

Workflow:
  1. Load a ZDF file  OR  connect a camera and capture.
  2. Choose detection source: Calibration Board or ArUco Marker.
  3. Choose camera config: Hand-to-Eye (Eye-to-Hand) or Hand-in-Eye (Eye-in-Hand).
  4. Load Hand-Eye calibration matrix (YAML file or manual 4×4 input).
  5. Hand-in-Eye only: provide the robot capture pose (YAML / 4×4 / xyz+rotation).
  6. Click "Detect & Estimate" → detect board/marker → transform to robot-base frame.
  7. Click "Visualize in 3D" → Open3D viewer with coordinate frame at detected object.
  8. Results printed to terminal and shown in the GUI output boxes.

Transformation formulas:
  Hand-to-Eye (Eye-to-Hand, camera fixed):
      T_robot = T_hand_eye @ T_object_in_camera

  Hand-in-Eye (Eye-in-Hand, camera on robot):
      T_robot = T_robot_capture @ T_hand_eye @ T_object_in_camera
"""

import os
import signal
import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import open3d as o3d
import zivid
from scipy.spatial.transform import Rotation
from PyQt5.QtCore import Qt, QThread, QTimer, pyqtSignal
from PyQt5.QtGui import QFont, QImage, QPixmap
from PyQt5.QtWidgets import (
    QMainWindow, QWidget, QLabel, QPushButton, QStatusBar,
    QHBoxLayout, QVBoxLayout, QFormLayout, QGroupBox,
    QComboBox, QRadioButton, QTextEdit, QButtonGroup,
    QFileDialog, QMessageBox, QSizePolicy, QScrollArea,
    QSplitter, QLineEdit,
)
from zividsamples.gui.qt_application import ZividQtApplication, ZividColors
from zividsamples.save_load_matrix import load_and_assert_affine_matrix


# ─────────────────────────────────────────────────────────────────────────────
#  Rotation format definitions
#  (label, scipy_seq_or_type, axis_labels)
#  scipy: lowercase = extrinsic (fixed axes), uppercase = intrinsic (moving axes)
# ─────────────────────────────────────────────────────────────────────────────

ROT_FMT_DEFS = [
    (
        "Rotation Vector  [rx  ry  rz]  (u, v, w)"
        "   → Universal Robots (PolyScope)",
        "rotvec",
        ("rx", "ry", "rz"),
    ),
    (
        "Quaternion  [qx  qy  qz  qw]  (q1-q4)"
        "   → ABB (RAPID)",
        "quat",
        ("qx", "qy", "qz", "qw"),
    ),
    (
        "Euler XYZ extrinsic  (고정축 X→Y→Z)"
        "   → Fanuc, Motoman/Yaskawa",
        "xyz",
        ("Rx(X)", "Ry(Y)", "Rz(Z)"),
    ),
    (
        "Euler ZYX extrinsic  (고정축 Z→Y→X)"
        "   → Epson, CRS",
        "zyx",
        ("Rz(Z)", "Ry(Y)", "Rx(X)"),
    ),
    (
        "Euler ZYX intrinsic  (이동축 Z→Y'→X''  A-B-C)"
        "   → ABB, KUKA, Nachi",
        "ZYX",
        ("A(Z)", "B(Y')", "C(X'')"),
    ),
    (
        "Euler ZYZ extrinsic  (고정축 Z→Y→Z)"
        "   → Denso",
        "zyz",
        ("W(Z)", "P(Y)", "R(Z)"),
    ),
    (
        "Euler ZYZ intrinsic  (이동축 Z→Y'→Z'')"
        "   → Doosan Robotics, Adept, Comau, Kawasaki",
        "ZYZ",
        ("α(Z)", "β(Y')", "γ(Z'')"),
    ),
    (
        "Euler ZXZ intrinsic  (이동축 Z→X'→Z'')"
        "   → CATIA, SolidWorks",
        "ZXZ",
        ("α(Z)", "β(X')", "γ(Z'')"),
    ),
    (
        "Euler XYZ intrinsic  (이동축 X→Y'→Z'')"
        "   → Stäubli (VAL3), Mecademic",
        "XYZ",
        ("α(X)", "β(Y')", "γ(Z'')"),
    ),
]


# ─────────────────────────────────────────────────────────────────────────────
#  Rotation helpers
# ─────────────────────────────────────────────────────────────────────────────

def _rot_to_display_str(rot: Rotation, fmt_idx: int, use_deg: bool) -> str:
    """Single-line rotation string for selected format and unit (GUI display)."""
    _, seq, labels = ROT_FMT_DEFS[fmt_idx]
    unit = "deg" if use_deg else "rad"

    if seq == "rotvec":
        rv = rot.as_rotvec()
        if use_deg:
            rv = np.degrees(rv)
        return (f"{labels[0]}: {rv[0]:+.6f}   "
                f"{labels[1]}: {rv[1]:+.6f}   "
                f"{labels[2]}: {rv[2]:+.6f}   [{unit}]")

    if seq == "quat":
        q = rot.as_quat()
        return (f"{labels[0]}: {q[0]:+.6f}   "
                f"{labels[1]}: {q[1]:+.6f}   "
                f"{labels[2]}: {q[2]:+.6f}   "
                f"{labels[3]}: {q[3]:+.6f}")

    a = rot.as_euler(seq, degrees=use_deg)
    return (f"{labels[0]}: {a[0]:+.6f}   "
            f"{labels[1]}: {a[1]:+.6f}   "
            f"{labels[2]}: {a[2]:+.6f}   [{unit}]")


def _pose_to_6dof_str(T: np.ndarray, fmt_idx: int, use_deg: bool) -> str:
    """Position [mm] + rotation string for GUI 6DoF display."""
    tx, ty, tz = T[:3, 3]
    rot = Rotation.from_matrix(T[:3, :3])
    pos_str = f"X: {tx:+10.3f}  Y: {ty:+10.3f}  Z: {tz:+10.3f}   [mm]"
    rot_str = _rot_to_display_str(rot, fmt_idx, use_deg)
    return f"{pos_str}\n{rot_str}"


def _pose_to_6dof_terminal_str(T: np.ndarray, fmt_idx: int) -> str:
    """Position [mm] + rotation string for terminal — always shows both rad AND deg."""
    tx, ty, tz = T[:3, 3]
    rot = Rotation.from_matrix(T[:3, :3])
    _, seq, labels = ROT_FMT_DEFS[fmt_idx]

    pos_str = f"X: {tx:+10.3f}   Y: {ty:+10.3f}   Z: {tz:+10.3f}   [mm]"

    if seq == "quat":
        q = rot.as_quat()
        rot_str = (f"{labels[0]}: {q[0]:+.6f}   {labels[1]}: {q[1]:+.6f}   "
                   f"{labels[2]}: {q[2]:+.6f}   {labels[3]}: {q[3]:+.6f}")
        return f"{pos_str}\n{rot_str}"

    if seq == "rotvec":
        rv = rot.as_rotvec()
        rv_d = np.degrees(rv)
        rad_str = (f"{labels[0]}: {rv[0]:+.6f}   {labels[1]}: {rv[1]:+.6f}   "
                   f"{labels[2]}: {rv[2]:+.6f}   [rad]")
        deg_str = (f"{labels[0]}: {rv_d[0]:+.8f}   {labels[1]}: {rv_d[1]:+.8f}   "
                   f"{labels[2]}: {rv_d[2]:+.8f}   [deg]")
        return f"{pos_str}\n{rad_str}\n{deg_str}"

    # Euler
    a_rad = rot.as_euler(seq, degrees=False)
    a_deg = np.degrees(a_rad)
    rad_str = (f"{labels[0]}: {a_rad[0]:+.6f}   {labels[1]}: {a_rad[1]:+.6f}   "
               f"{labels[2]}: {a_rad[2]:+.6f}   [rad]")
    deg_str = (f"{labels[0]}: {a_deg[0]:+.8f}   {labels[1]}: {a_deg[1]:+.8f}   "
               f"{labels[2]}: {a_deg[2]:+.8f}   [deg]")
    return f"{pos_str}\n{rad_str}\n{deg_str}"


# ─────────────────────────────────────────────────────────────────────────────
#  Matrix helpers
# ─────────────────────────────────────────────────────────────────────────────

def _parse_matrix_text(text: str) -> np.ndarray:
    try:
        nums = [float(x) for x in text.split()]
    except ValueError as exc:
        raise ValueError(f"숫자 파싱 오류: {exc}") from exc
    if len(nums) != 16:
        raise ValueError(f"16개의 숫자가 필요합니다. 입력: {len(nums)}개")
    m = np.array(nums, dtype=float).reshape(4, 4)
    try:
        zivid.calibration.Pose(m)
    except RuntimeError as exc:
        raise ValueError(f"유효하지 않은 affine 매트릭스: {exc}") from exc
    return m


def _parse_xyz_rot_text(text: str, fmt_idx: int, use_deg: bool) -> np.ndarray:
    """Parse 'x y z r1 r2 r3 [r4]' into 4×4 matrix. Translation in mm."""
    try:
        nums = [float(v) for v in text.split()]
    except ValueError as exc:
        raise ValueError(f"숫자 파싱 오류: {exc}") from exc

    _, seq, _ = ROT_FMT_DEFS[fmt_idx]

    if seq == "quat":
        if len(nums) != 7:
            raise ValueError(
                f"Quaternion 모드: 7개 값(x y z qx qy qz qw) 필요, 입력: {len(nums)}")
        tx, ty, tz = nums[:3]
        rot = Rotation.from_quat(nums[3:])
    else:
        if len(nums) != 6:
            raise ValueError(f"6개 값(x y z r1 r2 r3) 필요, 입력: {len(nums)}")
        tx, ty, tz = nums[:3]
        rv = np.array(nums[3:], dtype=float)
        if seq == "rotvec":
            if use_deg:
                rv = np.radians(rv)
            rot = Rotation.from_rotvec(rv)
        else:
            rot = Rotation.from_euler(seq, rv, degrees=use_deg)

    mat = np.eye(4)
    mat[:3, :3] = rot.as_matrix()
    mat[:3, 3]  = [tx, ty, tz]
    return mat


def _xyz_rot_placeholder(fmt_idx: int, use_deg: bool) -> str:
    _, seq, labels = ROT_FMT_DEFS[fmt_idx]
    unit = "deg" if use_deg else "rad"
    if seq == "quat":
        return "x  y  z  qx  qy  qz  qw   (위치: mm)"
    rot_part = "  ".join(labels[:3])
    return f"x  y  z  {rot_part}   [{unit}]   (위치: mm)"


def _fmt_mat(m: np.ndarray) -> str:
    return "\n".join(
        "  ".join(f"{v:+10.4f}" for v in row)
        for row in m
    )


# ─────────────────────────────────────────────────────────────────────────────
#  Open3D visualizer thread
# ─────────────────────────────────────────────────────────────────────────────

class _Open3DThread(QThread):
    done = pyqtSignal()

    def __init__(self, xyz: np.ndarray, rgba: np.ndarray, poses: list):
        super().__init__()
        self._xyz   = xyz
        self._rgba  = rgba
        self._poses = poses
        self._stop  = False

    def stop(self) -> None:
        self._stop = True

    def run(self) -> None:
        xyz_flat = self._xyz.reshape(-1, 3)
        rgb_flat = self._rgba[:, :, :3].reshape(-1, 3).astype(float) / 255.0
        valid    = ~np.isnan(xyz_flat).any(axis=1)

        pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(xyz_flat[valid]))
        pcd.colors = o3d.utility.Vector3dVector(rgb_flat[valid])

        cam_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=40)
        geometries = [pcd, cam_frame]

        for _label, T in self._poses:
            frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=60)
            frame.transform(T)
            geometries.append(frame)

        vis = o3d.visualization.Visualizer()
        vis.create_window("Board/Marker Pose — 3D View", 1600, 900)
        for g in geometries:
            vis.add_geometry(g)

        opt = vis.get_render_option()
        opt.point_size       = 1.5
        opt.background_color = [0.12, 0.12, 0.12]

        vc = vis.get_view_control()
        vc.set_front([0, 0, -1])
        vc.set_up([0, -1, 0])

        while not self._stop and vis.poll_events():
            vis.update_renderer()

        vis.destroy_window()
        self.done.emit()


# ─────────────────────────────────────────────────────────────────────────────
#  Main application window
# ─────────────────────────────────────────────────────────────────────────────

class CalBoardMarkerPoseApp(QMainWindow):

    _C_BLUE   = f"rgb{ZividColors.DARK_BLUE}"
    _C_GREEN  = "rgb(60, 150, 90)"
    _C_PURPLE = "rgb(130, 70, 200)"

    def __init__(self, zivid_app: zivid.Application):
        super().__init__()
        self._zivid_app    = zivid_app
        self._camera:      Optional[zivid.Camera]  = None
        self._frame:       Optional[zivid.Frame]   = None
        self._xyz:         Optional[np.ndarray]    = None
        self._rgba:        Optional[np.ndarray]    = None
        self._pixmap:      Optional[QPixmap]       = None
        self._cam_poses:   list                    = []
        self._robot_poses: list                    = []
        self._handeye:     Optional[np.ndarray]    = None
        self._robot_cap:   Optional[np.ndarray]    = None
        self._o3d_thread:  Optional[_Open3DThread] = None
        self._build_ui()

    # ─────────────────────────────────────────────────────────────────────────
    #  UI construction
    # ─────────────────────────────────────────────────────────────────────────

    def _build_ui(self) -> None:
        self.setWindowTitle("Calibration Board / Marker Pose Estimator")
        self.setMinimumSize(1480, 820)

        root     = QWidget()
        root_lay = QVBoxLayout(root)
        root_lay.setContentsMargins(6, 6, 6, 4)
        root_lay.setSpacing(6)
        self.setCentralWidget(root)

        root_lay.addWidget(self._make_toolbar())

        splitter = QSplitter(Qt.Horizontal)
        splitter.setHandleWidth(5)

        self._img_label = QLabel("ZDF 파일을 로드하거나 카메라를 연결하세요")
        self._img_label.setAlignment(Qt.AlignCenter)
        self._img_label.setMinimumSize(680, 450)
        self._img_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self._img_label.setStyleSheet("background:#181818; border:1px solid #444;")
        self._img_label.setFont(QFont("Helvetica", 13))
        self._img_label.setWordWrap(True)

        splitter.addWidget(self._img_label)
        splitter.addWidget(self._make_right_panel())
        splitter.setSizes([800, 640])
        root_lay.addWidget(splitter, stretch=1)

        self._sb = QStatusBar()
        self.setStatusBar(self._sb)
        self._status("Ready  —  ZDF 파일을 로드하거나 카메라를 연결하세요")

    # ── Toolbar ───────────────────────────────────────────────────────────────

    def _make_toolbar(self) -> QWidget:
        bar = QWidget()
        bar.setFixedHeight(52)
        bar.setStyleSheet("background-color:#2a2a2a; border-radius:4px;")
        lay = QHBoxLayout(bar)
        lay.setContentsMargins(8, 6, 8, 6)
        lay.setSpacing(8)

        self._btn_load    = self._mk_btn("📂  Load ZDF",       self._C_BLUE,  self._load_zdf)
        self._btn_connect = self._mk_btn("📷  Connect Camera", self._C_BLUE,  self._connect_camera)
        self._btn_capture = self._mk_btn("⚡  Capture",        self._C_GREEN, self._capture)
        self._btn_capture.setEnabled(False)

        lay.addWidget(self._btn_load)
        lay.addWidget(self._btn_connect)
        lay.addWidget(self._btn_capture)
        lay.addStretch()
        return bar

    # ── Right panel ───────────────────────────────────────────────────────────

    def _make_right_panel(self) -> QScrollArea:
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setMinimumWidth(580)

        inner = QWidget()
        lay   = QVBoxLayout(inner)
        lay.setSpacing(8)
        lay.setContentsMargins(8, 8, 8, 8)

        # Groups are created first so all self._xxx refs exist before signals fire
        grp_source    = self._make_group_detection_source()
        grp_marker    = self._make_group_marker_config()
        grp_cam_cfg   = self._make_group_camera_config()
        grp_handeye   = self._make_group_handeye()
        grp_robot_pos = self._make_group_robot_pose()
        grp_out_fmt   = self._make_group_output_format()

        # Layout order: marker config sits directly under detection source
        lay.addWidget(grp_source)
        lay.addWidget(grp_marker)     # shown/hidden by source toggle
        lay.addWidget(grp_cam_cfg)
        lay.addWidget(grp_handeye)
        lay.addWidget(grp_robot_pos)
        lay.addWidget(grp_out_fmt)
        lay.addStretch()

        self._btn_detect    = self._mk_btn("🎯  Detect & Estimate", self._C_PURPLE, self._detect_and_estimate)
        self._btn_visualize = self._mk_btn("🌐  Visualize in 3D",   self._C_BLUE,   self._visualize_3d)
        self._btn_detect.setFixedHeight(44)
        self._btn_visualize.setFixedHeight(44)
        self._btn_detect.setEnabled(False)
        self._btn_visualize.setEnabled(False)

        lay.addWidget(self._btn_detect)
        lay.addWidget(self._btn_visualize)
        lay.addWidget(self._make_group_result())

        scroll.setWidget(inner)
        return scroll

    # ── Detection source ──────────────────────────────────────────────────────

    def _make_group_detection_source(self) -> QGroupBox:
        grp = QGroupBox("Detection Source")
        lay = QHBoxLayout(grp)
        self._radio_board  = QRadioButton("Calibration Board  (체커보드)")
        self._radio_marker = QRadioButton("ArUco Marker")
        self._radio_board.setChecked(True)
        self._radio_board.toggled.connect(self._on_source_changed)
        lay.addWidget(self._radio_board)
        lay.addWidget(self._radio_marker)
        lay.addStretch()
        return grp

    # ── Marker configuration ──────────────────────────────────────────────────

    def _make_group_marker_config(self) -> QGroupBox:
        self._grp_marker = QGroupBox("Marker Configuration")
        self._grp_marker.setVisible(False)   # shown when ArUco Marker is selected
        lay = QFormLayout(self._grp_marker)
        lay.setSpacing(6)

        self._combo_dict = QComboBox()
        try:
            from zivid.calibration import MarkerDictionary
            self._combo_dict.addItems(MarkerDictionary.valid_values())
            self._combo_dict.setCurrentText(MarkerDictionary.aruco4x4_50)
        except Exception:
            self._combo_dict.addItem("aruco4x4_50")
        lay.addRow("Dictionary:", self._combo_dict)

        self._edit_ids = QLineEdit()
        self._edit_ids.setPlaceholderText("예) 0, 1, 2   (쉼표 또는 공백으로 구분)")
        self._edit_ids.setText("0, 1, 2, 3, 4")
        lay.addRow("Marker IDs:", self._edit_ids)

        id_note = QLabel("여러 마커 감지 시 ID가 가장 작은 마커 우선 사용 (0 → 1 → 2 …)")
        id_note.setFont(QFont("", 9))
        id_note.setStyleSheet("color:#aaaaaa;")
        lay.addRow("", id_note)
        return self._grp_marker

    # ── Camera configuration ──────────────────────────────────────────────────

    def _make_group_camera_config(self) -> QGroupBox:
        grp = QGroupBox("Camera Configuration")
        lay = QVBoxLayout(grp)

        self._radio_eye2hand    = QRadioButton(
            "Hand-to-Eye  (Eye-to-Hand)  —  카메라 고정")
        self._radio_eye_in_hand = QRadioButton(
            "Hand-in-Eye  (Eye-in-Hand)  —  카메라가 로봇에 부착, 캡처 포즈 필요")
        self._radio_eye2hand.setChecked(True)
        self._radio_eye2hand.toggled.connect(self._on_eye_type_changed)
        bg = QButtonGroup(self)
        bg.addButton(self._radio_eye2hand)
        bg.addButton(self._radio_eye_in_hand)
        lay.addWidget(self._radio_eye2hand)
        lay.addWidget(self._radio_eye_in_hand)

        note = QLabel(
            "  Hand-to-Eye:  T_robot = T_hand_eye × T_object_camera\n"
            "  Hand-in-Eye:  T_robot = T_capture × T_hand_eye × T_object_camera"
        )
        note.setFont(QFont("Courier New", 9))
        note.setStyleSheet("color:#999999;")
        lay.addWidget(note)
        return grp

    # ── Hand-Eye calibration ──────────────────────────────────────────────────

    def _make_group_handeye(self) -> QGroupBox:
        grp = QGroupBox("Hand-Eye Calibration Matrix  (4×4)")
        lay = QVBoxLayout(grp)
        lay.setSpacing(6)

        src_row = QHBoxLayout()
        self._radio_he_file   = QRadioButton("File (YAML)")
        self._radio_he_manual = QRadioButton("Manual")
        self._radio_he_file.setChecked(True)
        self._radio_he_file.toggled.connect(self._on_he_src_changed)
        src_row.addWidget(self._radio_he_file)
        src_row.addWidget(self._radio_he_manual)
        src_row.addStretch()
        lay.addLayout(src_row)

        self._he_file_widget = QWidget()
        fw = QHBoxLayout(self._he_file_widget)
        fw.setContentsMargins(0, 0, 0, 0)
        btn_he = QPushButton("📂 Load YAML")
        btn_he.setFixedHeight(28)
        btn_he.clicked.connect(self._load_handeye_yaml)
        self._lbl_he_file = QLabel("—  미로드")
        self._lbl_he_file.setFont(QFont("Courier New", 9))
        self._lbl_he_file.setStyleSheet("color:#888888;")
        fw.addWidget(btn_he)
        fw.addWidget(self._lbl_he_file, stretch=1)
        lay.addWidget(self._he_file_widget)

        self._he_manual_widget = QWidget()
        mw = QVBoxLayout(self._he_manual_widget)
        mw.setContentsMargins(0, 0, 0, 0)
        self._txt_he = QTextEdit()
        self._txt_he.setFont(QFont("Courier New", 9))
        self._txt_he.setFixedHeight(76)
        self._txt_he.setPlaceholderText(
            "r11 r12 r13 tx\nr21 r22 r23 ty\nr31 r32 r33 tz\n0   0   0   1")
        btn_he_apply = QPushButton("Apply")
        btn_he_apply.setFixedHeight(24)
        btn_he_apply.clicked.connect(self._apply_handeye_manual)
        mw.addWidget(self._txt_he)
        mw.addWidget(btn_he_apply)
        lay.addWidget(self._he_manual_widget)
        self._he_manual_widget.setVisible(False)
        return grp

    # ── Robot capture pose (Hand-in-Eye only) ─────────────────────────────────

    def _make_group_robot_pose(self) -> QGroupBox:
        self._grp_robot_pose = QGroupBox("Robot Capture Pose  (Hand-in-Eye 전용)")
        self._grp_robot_pose.setEnabled(False)
        lay = QVBoxLayout(self._grp_robot_pose)
        lay.setSpacing(6)

        fmt_row = QHBoxLayout()
        fmt_row.addWidget(QLabel("입력 형식:"))
        self._combo_rc_input_fmt = QComboBox()
        self._combo_rc_input_fmt.addItem("4×4 Matrix  (File YAML)")
        self._combo_rc_input_fmt.addItem("4×4 Matrix  (Manual Text)")
        for label, _, _ in ROT_FMT_DEFS:
            self._combo_rc_input_fmt.addItem(
                f"xyz + {label.split('  →')[0].strip()}")
        self._combo_rc_input_fmt.currentIndexChanged.connect(
            self._on_rc_input_fmt_changed)
        fmt_row.addWidget(self._combo_rc_input_fmt, stretch=1)
        lay.addLayout(fmt_row)

        # File input
        self._rc_file_widget = QWidget()
        rfw = QHBoxLayout(self._rc_file_widget)
        rfw.setContentsMargins(0, 0, 0, 0)
        btn_rc = QPushButton("📂 Load YAML")
        btn_rc.setFixedHeight(28)
        btn_rc.clicked.connect(self._load_robot_cap_yaml)
        self._lbl_rc_file = QLabel("—  미로드")
        self._lbl_rc_file.setFont(QFont("Courier New", 9))
        self._lbl_rc_file.setStyleSheet("color:#888888;")
        rfw.addWidget(btn_rc)
        rfw.addWidget(self._lbl_rc_file, stretch=1)
        lay.addWidget(self._rc_file_widget)

        # Text input
        self._rc_text_widget = QWidget()
        rtw = QVBoxLayout(self._rc_text_widget)
        rtw.setContentsMargins(0, 0, 0, 0)
        rtw.setSpacing(3)
        self._txt_rc = QTextEdit()
        self._txt_rc.setFont(QFont("Courier New", 9))
        self._txt_rc.setFixedHeight(76)
        self._txt_rc.setPlaceholderText(
            "r11 r12 r13 tx\nr21 r22 r23 ty\nr31 r32 r33 tz\n0   0   0   1")
        btn_rc_apply = QPushButton("Apply")
        btn_rc_apply.setFixedHeight(24)
        btn_rc_apply.clicked.connect(self._apply_rc_manual)
        rtw.addWidget(self._txt_rc)
        rtw.addWidget(btn_rc_apply)
        lay.addWidget(self._rc_text_widget)
        self._rc_text_widget.setVisible(False)

        self._lbl_rc_status = QLabel("—  미입력")
        self._lbl_rc_status.setFont(QFont("Courier New", 9))
        self._lbl_rc_status.setStyleSheet("color:#888888;")
        lay.addWidget(self._lbl_rc_status)
        return self._grp_robot_pose

    # ── Output format ─────────────────────────────────────────────────────────

    def _make_group_output_format(self) -> QGroupBox:
        grp = QGroupBox("출력 회전 표현 형식")
        lay = QFormLayout(grp)
        lay.setSpacing(6)

        self._combo_rot_fmt = QComboBox()
        for label, _, _ in ROT_FMT_DEFS:
            self._combo_rot_fmt.addItem(label)
        self._combo_rot_fmt.setCurrentIndex(0)
        self._combo_rot_fmt.currentIndexChanged.connect(self._refresh_results)
        lay.addRow("회전 형식:", self._combo_rot_fmt)

        unit_row = QHBoxLayout()
        self._radio_rad = QRadioButton("Radian  (기본)")
        self._radio_deg = QRadioButton("Degree")
        self._radio_rad.setChecked(True)
        unit_bg = QButtonGroup(self)
        unit_bg.addButton(self._radio_rad)
        unit_bg.addButton(self._radio_deg)
        self._radio_rad.toggled.connect(self._on_unit_changed)
        unit_row.addWidget(self._radio_rad)
        unit_row.addWidget(self._radio_deg)
        unit_row.addStretch()
        lay.addRow("각도 단위:", unit_row)
        return grp

    # ── Results ────────────────────────────────────────────────────────────────

    def _make_group_result(self) -> QGroupBox:
        grp = QGroupBox("결과")
        lay = QVBoxLayout(grp)
        lay.setSpacing(6)

        # Camera Frame 4×4
        lbl_cam = QLabel("Camera Frame Pose  (4×4):")
        lbl_cam.setFont(QFont("", 9))
        lay.addWidget(lbl_cam)
        self._txt_cam_pose = QTextEdit()
        self._txt_cam_pose.setReadOnly(True)
        self._txt_cam_pose.setFont(QFont("Courier New", 9))
        self._txt_cam_pose.setFixedHeight(108)
        self._txt_cam_pose.setPlaceholderText("—  감지 후 표시됩니다  —")
        lay.addWidget(self._txt_cam_pose)

        # Robot Base 4×4
        lbl_rob = QLabel("Robot Base Frame Pose  (4×4):")
        lbl_rob.setFont(QFont("", 9))
        lay.addWidget(lbl_rob)
        self._txt_robot_pose = QTextEdit()
        self._txt_robot_pose.setReadOnly(True)
        self._txt_robot_pose.setFont(QFont("Courier New", 9))
        self._txt_robot_pose.setFixedHeight(108)
        self._txt_robot_pose.setPlaceholderText("—  감지 후 표시됩니다  —")
        lay.addWidget(self._txt_robot_pose)

        # 6DoF — separate, larger, highlighted
        lbl_6dof = QLabel("Robot Base Frame  6DoF  (xyz [mm]  +  회전):")
        lbl_6dof.setFont(QFont("", 9, QFont.Bold))
        lbl_6dof.setStyleSheet("color:#80d0ff;")
        lay.addWidget(lbl_6dof)
        self._txt_6dof = QTextEdit()
        self._txt_6dof.setReadOnly(True)
        self._txt_6dof.setFont(QFont("Courier New", 11))
        self._txt_6dof.setFixedHeight(80)
        self._txt_6dof.setStyleSheet(
            "QTextEdit {"
            "  color: #80ffb0;"
            "  background-color: #0d1f0d;"
            "  border: 1px solid #336633;"
            "  padding: 4px;"
            "}"
        )
        self._txt_6dof.setPlaceholderText("—  감지 후 표시됩니다  —")
        lay.addWidget(self._txt_6dof)
        return grp

    # ── Button factory ────────────────────────────────────────────────────────

    @staticmethod
    def _mk_btn(label: str, bg: str, cb) -> QPushButton:
        btn = QPushButton(label)
        btn.setFixedHeight(36)
        btn.setStyleSheet(f"""
            QPushButton {{
                background-color: {bg};
                color: white;
                border: none;
                border-radius: 4px;
                padding: 0 14px;
                font-weight: bold;
                font-size: 13px;
            }}
            QPushButton:hover    {{ border: 1px solid #ffffff55; }}
            QPushButton:disabled {{ background-color: rgb(70,70,70); color: rgb(140,140,140); }}
        """)
        btn.clicked.connect(cb)
        return btn

    # ─────────────────────────────────────────────────────────────────────────
    #  Slots — UI state transitions
    # ─────────────────────────────────────────────────────────────────────────

    def _on_source_changed(self) -> None:
        self._grp_marker.setVisible(self._radio_marker.isChecked())

    def _on_eye_type_changed(self) -> None:
        self._grp_robot_pose.setEnabled(self._radio_eye_in_hand.isChecked())

    def _on_he_src_changed(self) -> None:
        is_file = self._radio_he_file.isChecked()
        self._he_file_widget.setVisible(is_file)
        self._he_manual_widget.setVisible(not is_file)

    def _on_rc_input_fmt_changed(self) -> None:
        idx = self._combo_rc_input_fmt.currentIndex()
        self._rc_file_widget.setVisible(idx == 0)
        self._rc_text_widget.setVisible(idx > 0)
        if idx <= 1:
            self._txt_rc.setPlaceholderText(
                "r11 r12 r13 tx\nr21 r22 r23 ty\nr31 r32 r33 tz\n0   0   0   1")
        else:
            fmt_idx = idx - 2
            use_deg = self._radio_deg.isChecked()
            self._txt_rc.setPlaceholderText(_xyz_rot_placeholder(fmt_idx, use_deg))

    def _on_unit_changed(self) -> None:
        self._on_rc_input_fmt_changed()
        self._refresh_results()

    def _refresh_results(self) -> None:
        if not self._cam_poses:
            return
        fmt_idx = self._combo_rot_fmt.currentIndex()
        use_deg = self._radio_deg.isChecked()

        cam_parts   = []
        robot_parts = []
        dof_parts   = []
        for label, T_cam in self._cam_poses:
            cam_parts.append(f"[{label}]\n{_fmt_mat(T_cam)}")
        for label, T_rob in self._robot_poses:
            robot_parts.append(f"[{label}]\n{_fmt_mat(T_rob)}")
            dof_parts.append(
                f"[{label}]\n{_pose_to_6dof_str(T_rob, fmt_idx, use_deg)}")

        self._txt_cam_pose.setText("\n\n".join(cam_parts))
        self._txt_robot_pose.setText("\n\n".join(robot_parts))
        self._txt_6dof.setText("\n\n".join(dof_parts))

    # ─────────────────────────────────────────────────────────────────────────
    #  Data loading
    # ─────────────────────────────────────────────────────────────────────────

    def _load_zdf(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Open ZDF File", "", "ZDF Files (*.zdf)")
        if not path:
            return
        self._status(f"로딩 중  {Path(path).name} …")
        try:
            self._ingest_frame(zivid.Frame(path))
            self._status(f"로드 완료:  {Path(path).name}")
        except Exception as exc:
            self._status(f"로드 오류: {exc}", error=True)
            QMessageBox.critical(self, "Load Error", str(exc))

    def _connect_camera(self) -> None:
        self._status("카메라 연결 중 …")
        try:
            self._camera = self._zivid_app.connect_camera()
            self._btn_capture.setEnabled(True)
            self._status(f"연결됨:  {self._camera.info.model}")
        except Exception as exc:
            self._status(f"연결 실패: {exc}", error=True)
            QMessageBox.critical(self, "Camera Error", str(exc))

    def _capture(self) -> None:
        if self._camera is None:
            return
        self._status("캡처 중 …")
        try:
            settings = zivid.capture_assistant.suggest_settings(
                self._camera,
                zivid.capture_assistant.SuggestSettingsParameters(
                    max_capture_time=datetime.timedelta(milliseconds=1200),
                    ambient_light_frequency=(
                        zivid.capture_assistant.SuggestSettingsParameters
                        .AmbientLightFrequency.none
                    ),
                ),
            )
            self._ingest_frame(self._camera.capture_2d_3d(settings))
            self._status("캡처 완료")
        except Exception as exc:
            self._status(f"캡처 오류: {exc}", error=True)
            QMessageBox.critical(self, "Capture Error", str(exc))

    def _ingest_frame(self, frame: zivid.Frame) -> None:
        self._frame = frame
        pc         = frame.point_cloud()
        self._xyz  = pc.copy_data("xyz")
        self._rgba = pc.copy_data("rgba")
        H, W       = self._rgba.shape[:2]

        qimg = QImage(self._rgba[:, :, :3].tobytes(), W, H, 3 * W, QImage.Format_RGB888)
        self._pixmap = QPixmap.fromImage(qimg)
        self._img_label.setPixmap(
            self._pixmap.scaled(self._img_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))

        self._cam_poses   = []
        self._robot_poses = []
        self._txt_cam_pose.clear()
        self._txt_robot_pose.clear()
        self._txt_6dof.clear()
        self._btn_detect.setEnabled(True)
        self._btn_visualize.setEnabled(False)

    def _load_handeye_yaml(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Hand-Eye YAML 로드", "", "YAML Files (*.yaml *.yml)")
        if not path:
            return
        try:
            self._handeye = load_and_assert_affine_matrix(Path(path))
            self._lbl_he_file.setText(Path(path).name)
            self._lbl_he_file.setStyleSheet("color:#80ffb0;")
            self._status(f"Hand-Eye 로드: {Path(path).name}")
        except Exception as exc:
            self._status(f"Hand-Eye 로드 오류: {exc}", error=True)
            QMessageBox.critical(self, "Load Error", str(exc))

    def _apply_handeye_manual(self) -> None:
        try:
            self._handeye = _parse_matrix_text(self._txt_he.toPlainText())
            self._status("Hand-Eye 매트릭스 적용 완료")
        except Exception as exc:
            QMessageBox.critical(self, "Input Error", str(exc))

    def _load_robot_cap_yaml(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Robot Capture Pose YAML 로드", "", "YAML Files (*.yaml *.yml)")
        if not path:
            return
        try:
            self._robot_cap = load_and_assert_affine_matrix(Path(path))
            self._lbl_rc_file.setText(Path(path).name)
            self._lbl_rc_file.setStyleSheet("color:#80ffb0;")
            self._lbl_rc_status.setText(f"✅  {Path(path).name}")
            self._lbl_rc_status.setStyleSheet("color:#80ffb0;")
            self._status(f"Robot Capture Pose 로드: {Path(path).name}")
        except Exception as exc:
            self._status(f"Robot Capture Pose 로드 오류: {exc}", error=True)
            QMessageBox.critical(self, "Load Error", str(exc))

    def _apply_rc_manual(self) -> None:
        idx     = self._combo_rc_input_fmt.currentIndex()
        use_deg = self._radio_deg.isChecked()
        text    = self._txt_rc.toPlainText().strip()
        if not text:
            QMessageBox.warning(self, "입력 오류", "값을 입력하세요.")
            return
        try:
            if idx == 1:
                self._robot_cap = _parse_matrix_text(text)
            else:
                fmt_idx = idx - 2
                self._robot_cap = _parse_xyz_rot_text(text, fmt_idx, use_deg)
            self._lbl_rc_status.setText("✅  Manual input 적용됨")
            self._lbl_rc_status.setStyleSheet("color:#80ffb0;")
            self._status("Robot Capture Pose 적용 완료")
        except Exception as exc:
            QMessageBox.critical(self, "Input Error", str(exc))

    # ─────────────────────────────────────────────────────────────────────────
    #  Detection & estimation
    # ─────────────────────────────────────────────────────────────────────────

    def _get_marker_ids(self) -> list:
        raw = self._edit_ids.text()
        ids = []
        for tok in raw.replace(",", " ").split():
            try:
                ids.append(int(tok))
            except ValueError:
                pass
        return sorted(set(ids)) if ids else [0]

    def _detect_and_estimate(self) -> None:
        if self._frame is None:
            return

        if self._handeye is None:
            QMessageBox.warning(
                self, "입력 누락",
                "Hand-Eye Calibration 매트릭스가 없습니다.\n"
                "YAML 파일을 로드하거나 직접 입력해 주세요.")
            return

        eye_in = self._radio_eye_in_hand.isChecked()
        if eye_in and self._robot_cap is None:
            QMessageBox.warning(
                self, "입력 누락",
                "Hand-in-Eye 모드에서는 Robot Capture Pose가 필요합니다.\n"
                "YAML 파일을 로드하거나 직접 입력해 주세요.")
            return

        self._status("감지 중 …")
        try:
            cam_poses = self._run_detection()
        except Exception as exc:
            self._status(f"감지 오류: {exc}", error=True)
            QMessageBox.critical(self, "Detection Error", str(exc))
            return

        if not cam_poses:
            self._status("감지 실패: 보드/마커를 찾을 수 없습니다", error=True)
            QMessageBox.warning(
                self, "감지 실패",
                "ZDF 파일에서 보드/마커를 찾을 수 없습니다.\n"
                "Detection Source와 Marker 설정을 확인해 주세요.")
            return

        self._cam_poses   = cam_poses
        self._robot_poses = []
        for label, T_cam in cam_poses:
            T_rob = (self._robot_cap @ self._handeye @ T_cam
                     if eye_in else self._handeye @ T_cam)
            self._robot_poses.append((label, T_rob))

        fmt_idx = self._combo_rot_fmt.currentIndex()
        use_deg = self._radio_deg.isChecked()
        self._refresh_results()
        self._btn_visualize.setEnabled(True)
        self._status(f"감지 완료  |  {len(cam_poses)}개 오브젝트 검출")
        self._print_results(fmt_idx, use_deg, eye_in)

    def _run_detection(self) -> list:
        if self._radio_board.isChecked():
            result = zivid.calibration.detect_calibration_board(self._frame)
            if not result.valid():
                raise RuntimeError(
                    f"Calibration Board 감지 실패: {result.status_description()}")
            return [("Calibration Board", np.asarray(result.pose().to_matrix()))]

        dictionary = self._combo_dict.currentText()
        id_list    = self._get_marker_ids()
        result     = zivid.calibration.detect_markers(
            self._frame, id_list, dictionary)
        if not result.valid():
            raise RuntimeError("ArUco Marker 감지 실패")

        markers = result.detected_markers()
        if not markers:
            return []

        markers_sorted = sorted(markers, key=lambda m: m.identifier)
        return [(f"Marker ID={m.identifier}",
                 np.asarray(m.pose.to_matrix())) for m in markers_sorted]

    # ─────────────────────────────────────────────────────────────────────────
    #  3D visualization
    # ─────────────────────────────────────────────────────────────────────────

    def _visualize_3d(self) -> None:
        if not self._cam_poses or self._xyz is None:
            return
        if self._o3d_thread and self._o3d_thread.isRunning():
            self._status("3D 뷰어가 이미 열려 있습니다")
            return
        if self._o3d_thread is not None:
            self._o3d_thread.stop()
        self._o3d_thread = _Open3DThread(self._xyz, self._rgba, self._cam_poses)
        self._o3d_thread.done.connect(lambda: self._status("3D 뷰어 닫힘"))
        self._o3d_thread.start()
        self._status("3D 뷰어 열림  (Camera Frame 기준)")

    # ─────────────────────────────────────────────────────────────────────────
    #  Terminal output
    # ─────────────────────────────────────────────────────────────────────────

    def _print_results(self, fmt_idx: int, use_deg: bool, eye_in: bool) -> None:
        W   = 76
        bar = "─" * W

        def row(text: str) -> str:
            return f"│  {text:<{W - 2}}│"

        eye_lbl = "Hand-in-Eye (Eye-in-Hand)" if eye_in else "Hand-to-Eye (Eye-to-Hand)"
        ts      = datetime.datetime.now().strftime("%Y-%m-%d  %H:%M:%S")
        fmt_lbl = ROT_FMT_DEFS[fmt_idx][0].split("   →")[0].strip()

        lines = [
            f"┌{bar}┐",
            row(f"BOARD/MARKER POSE ESTIMATION  ·  {ts}"),
            row(f"Config: {eye_lbl}"),
            row(f"Rotation format: {fmt_lbl}"),
            f"╞{'═' * W}╡",
        ]

        for (label, T_cam), (_, T_rob) in zip(self._cam_poses, self._robot_poses):
            mat_rows_cam = [
                "  ".join(f"{v:+10.4f}" for v in T_cam[i]) for i in range(4)]
            mat_rows_rob = [
                "  ".join(f"{v:+10.4f}" for v in T_rob[i]) for i in range(4)]
            dof_str = _pose_to_6dof_terminal_str(T_rob, fmt_idx)

            lines += [
                row(f"  Object: {label}"),
                f"├{bar}┤",
                row("  Camera Frame Pose  4×4  [R | t]  (mm)"),
                *[row(f"    {r}") for r in mat_rows_cam],
                f"├{bar}┤",
                row("  Robot Base Frame Pose  4×4  [R | t]  (mm)"),
                *[row(f"    {r}") for r in mat_rows_rob],
                f"├{bar}┤",
                row(f"  6DoF  [{fmt_lbl}]"),
                *[row(f"    {line}") for line in dof_str.split("\n")],
                f"╞{'═' * W}╡",
            ]

        lines[-1] = f"└{bar}┘"
        print("\n" + "\n".join(lines) + "\n")

    # ─────────────────────────────────────────────────────────────────────────
    #  Utilities
    # ─────────────────────────────────────────────────────────────────────────

    def _status(self, msg: str, error: bool = False) -> None:
        self._sb.setStyleSheet("color:#ff7070;" if error else "color:#cccccc;")
        self._sb.showMessage(msg)

    def resizeEvent(self, e) -> None:
        super().resizeEvent(e)
        if self._pixmap is not None:
            self._img_label.setPixmap(
                self._pixmap.scaled(self._img_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))

    def closeEvent(self, e) -> None:
        QTimer.singleShot(800, _force_kill)
        if self._o3d_thread and self._o3d_thread.isRunning():
            self._o3d_thread.stop()
        if self._camera:
            try:
                self._camera.disconnect()
            except Exception:
                pass
        super().closeEvent(e)


# ─────────────────────────────────────────────────────────────────────────────
#  Entry point
# ─────────────────────────────────────────────────────────────────────────────

def _force_kill() -> None:
    os.kill(os.getpid(), signal.SIGTERM)


def main() -> None:
    with ZividQtApplication() as qt_app:
        win = CalBoardMarkerPoseApp(qt_app.zivid_app)
        qt_app.run(win, "Calibration Board / Marker Pose Estimator")
        _force_kill()


if __name__ == "__main__":
    main()
