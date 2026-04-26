"""
2D Point-Based Touch Pose Estimator  —  GUI Application

Run:
    python touch_pose_estimation.py

No CLI arguments needed. Everything is driven from the GUI:
  1. Load a ZDF file  OR  connect a camera and capture.
  2. Click a touch point on the 2D image  OR  enter (u, v) manually.
  3. Choose a ROI method: sphere radius around the point, or drag a rectangle.
  4. Select X-axis mode: SVD principal axis or camera +X projection.
  5. Click  "Estimate Pose"  →  SVD plane fit  →  4×4 pose matrix.
  6. Click  "Visualize in 3D"  →  Open3D viewer with pose frame + ROI highlight.
  7. (Optional) Enable Advanced to compute robot-base-frame pose via hand-eye calibration.
"""

import os
import signal
import datetime
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import open3d as o3d
import zivid
from scipy.spatial.transform import Rotation
from PyQt5.QtCore import Qt, QRect, QPoint, QThread, QTimer, pyqtSignal
from PyQt5.QtGui import QFont, QImage, QPixmap, QPainter, QPen, QColor, QMouseEvent
from PyQt5.QtWidgets import (
    QMainWindow, QWidget, QLabel, QPushButton, QStatusBar,
    QHBoxLayout, QVBoxLayout, QFormLayout, QGroupBox,
    QSpinBox, QDoubleSpinBox, QRadioButton, QTextEdit, QComboBox,
    QFileDialog, QMessageBox, QSizePolicy, QScrollArea, QSplitter,
    QButtonGroup,
)
from zividsamples.gui.qt_application import ZividQtApplication, ZividColors
from zividsamples.save_load_matrix import load_and_assert_affine_matrix


_6DOF_LABELS = (
    "4×4 Matrix",
    "Rotation Vector  [rx ry rz  (rad)]",
    "Quaternion  [qx qy qz qw]",
    "Euler XYZ extrinsic  (RPY: Roll Pitch Yaw)  deg",
    "Euler ZYX extrinsic  (Yaw-Pitch-Roll)  deg",
    "Euler ZYX intrinsic  (KUKA A-B-C)  deg",
    "Euler ZYZ extrinsic  deg",
    "Euler ZYZ intrinsic  deg",
    "Euler XYZ intrinsic  deg",
)


# ─────────────────────────────────────────────────────────────────────────────
#  Open3D thread  (runs in background so GUI stays responsive)
# ─────────────────────────────────────────────────────────────────────────────

class _Open3DThread(QThread):
    done = pyqtSignal()

    def __init__(
        self,
        xyz:     np.ndarray,
        rgba:    np.ndarray,
        pose:    np.ndarray,
        roi_pts: Optional[np.ndarray],
    ):
        super().__init__()
        self._xyz     = xyz
        self._rgba    = rgba
        self._pose    = pose
        self._roi_pts = roi_pts
        self._stop    = False

    def stop(self) -> None:
        self._stop = True

    def run(self) -> None:
        xyz_flat = self._xyz.reshape(-1, 3)
        rgb_flat = self._rgba[:, :, :3].reshape(-1, 3).astype(float) / 255.0
        valid = ~np.isnan(xyz_flat).any(axis=1)
        pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(xyz_flat[valid]))
        pcd.colors = o3d.utility.Vector3dVector(rgb_flat[valid])

        # Touch pose coordinate frame
        frame_mesh = o3d.geometry.TriangleMesh.create_coordinate_frame(size=70)
        frame_mesh.transform(self._pose)

        # Camera origin coordinate frame at [0, 0, 0] (Zivid camera optical center)
        camera_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=40)

        geometries = [pcd, frame_mesh, camera_frame]

        if self._roi_pts is not None and len(self._roi_pts) > 0:
            roi_pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(self._roi_pts))
            roi_pcd.paint_uniform_color([1.0, 0.35, 0.1])
            geometries.append(roi_pcd)

        vis = o3d.visualization.Visualizer()
        vis.create_window("Touch Pose — 3D View", 1600, 900)
        for g in geometries:
            vis.add_geometry(g)

        opt = vis.get_render_option()
        opt.point_size            = 1.5
        opt.background_color      = [0.12, 0.12, 0.12]
        opt.show_coordinate_frame = False

        vc = vis.get_view_control()
        vc.set_front([0, 0, -1])
        vc.set_up([0, -1, 0])

        while not self._stop and vis.poll_events():
            vis.update_renderer()

        vis.destroy_window()
        self.done.emit()


# ─────────────────────────────────────────────────────────────────────────────
#  Interactive 2D image viewer
# ─────────────────────────────────────────────────────────────────────────────

class ImageViewer(QLabel):
    """QLabel that shows a Zivid 2D frame and handles:
      - left-click  → select touch point   (MODE_POINT)
      - click-drag  → select ROI rectangle (MODE_RECT)
    Emits pixel coordinates in **original image space**.
    """

    point_selected    = pyqtSignal(int, int)            # u, v
    rect_roi_selected = pyqtSignal(int, int, int, int)  # x, y, w, h

    MODE_POINT = "point"
    MODE_RECT  = "rect"

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAlignment(Qt.AlignCenter)
        self.setMinimumSize(640, 400)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setStyleSheet("background-color: #181818; border: 1px solid #444;")
        self.setFont(QFont("Helvetica", 13))
        self.setWordWrap(True)
        self.setText("Load a ZDF file or connect a camera to begin")

        self._pixmap_orig: Optional[QPixmap]       = None
        self.mode:         str                      = self.MODE_POINT
        self._sel_pt:      Optional[Tuple[int,int]] = None
        self._roi_r:       float                    = 10.0
        self._drag_start:  Optional[QPoint]         = None
        self._drag_live:   Optional[QRect]          = None
        self._sel_rect:    Optional[QRect]          = None

    # ── Public API ─────────────────────────────────────────────────────────

    def load_rgb(self, rgb: np.ndarray) -> None:
        h, w   = rgb.shape[:2]
        qimg   = QImage(rgb.tobytes(), w, h, 3 * w, QImage.Format_RGB888)
        self._pixmap_orig = QPixmap.fromImage(qimg)
        self._sel_pt = self._drag_live = self._sel_rect = None
        self.setText("")
        self._redraw()

    def set_mode(self, mode: str) -> None:
        self.mode = mode
        self._drag_start = self._drag_live = None
        self._redraw()

    def set_roi_radius(self, r: float) -> None:
        self._roi_r = r
        self._redraw()

    def set_selected_point(self, u: int, v: int) -> None:
        self._sel_pt = (u, v)
        self._redraw()

    # ── Coordinate mapping ──────────────────────────────────────────────────

    def _pm_rect(self) -> Optional[QRect]:
        if self._pixmap_orig is None:
            return None
        pm = self._pixmap_orig.scaled(self.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        x = (self.width()  - pm.width())  // 2
        y = (self.height() - pm.height()) // 2
        return QRect(x, y, pm.width(), pm.height())

    def _widget_to_image(self, pt: QPoint) -> Optional[Tuple[int, int]]:
        r = self._pm_rect()
        if r is None or r.width() == 0 or r.height() == 0:
            return None
        sx = self._pixmap_orig.width()  / r.width()
        sy = self._pixmap_orig.height() / r.height()
        u = int((pt.x() - r.x()) * sx)
        v = int((pt.y() - r.y()) * sy)
        u = max(0, min(u, self._pixmap_orig.width()  - 1))
        v = max(0, min(v, self._pixmap_orig.height() - 1))
        return (u, v)

    def _image_to_pm_local(self, u: int, v: int) -> Tuple[float, float]:
        r = self._pm_rect()
        sx = r.width()  / self._pixmap_orig.width()
        sy = r.height() / self._pixmap_orig.height()
        return (u * sx, v * sy)

    # ── Overlay drawing ─────────────────────────────────────────────────────

    def _redraw(self) -> None:
        if self._pixmap_orig is None:
            return
        r  = self._pm_rect()
        pm = self._pixmap_orig.scaled(self.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)

        painter = QPainter(pm)
        painter.setRenderHint(QPainter.Antialiasing)

        if self._sel_pt is not None and self.mode == self.MODE_POINT:
            px, py = self._image_to_pm_local(*self._sel_pt)

            r_vis = max(12.0, self._roi_r * pm.width() / 900.0)
            painter.setPen(QPen(QColor(255, 200, 40), 1.5, Qt.DashLine))
            painter.setBrush(Qt.NoBrush)
            painter.drawEllipse(QPoint(int(px), int(py)), int(r_vis), int(r_vis))

            arm = 16
            painter.setPen(QPen(QColor(50, 255, 110), 2))
            painter.drawLine(int(px - arm), int(py), int(px + arm), int(py))
            painter.drawLine(int(px), int(py - arm), int(px), int(py + arm))

            painter.setBrush(QColor(50, 255, 110))
            painter.setPen(Qt.NoPen)
            painter.drawEllipse(QPoint(int(px), int(py)), 5, 5)

        if self._drag_live is not None and self.mode == self.MODE_RECT:
            dr = self._drag_live
            rx = dr.x() - r.x()
            ry = dr.y() - r.y()
            painter.setPen(QPen(QColor(255, 200, 40), 2, Qt.SolidLine))
            painter.setBrush(QColor(255, 200, 40, 45))
            painter.drawRect(rx, ry, dr.width(), dr.height())

        if self._sel_rect is not None and self.mode == self.MODE_RECT:
            sr = self._sel_rect
            sx = pm.width()  / self._pixmap_orig.width()
            sy = pm.height() / self._pixmap_orig.height()
            painter.setPen(QPen(QColor(50, 255, 110), 2))
            painter.setBrush(QColor(50, 255, 110, 45))
            painter.drawRect(int(sr.x()*sx), int(sr.y()*sy),
                             int(sr.width()*sx), int(sr.height()*sy))

        painter.end()
        self.setPixmap(pm)

    # ── Mouse events ─────────────────────────────────────────────────────────

    def mousePressEvent(self, e: QMouseEvent) -> None:
        if self._pixmap_orig is None or e.button() != Qt.LeftButton:
            return
        if self.mode == self.MODE_POINT:
            pt = self._widget_to_image(e.pos())
            if pt:
                self._sel_pt = pt
                self.point_selected.emit(*pt)
                self._redraw()
        else:
            self._drag_start = e.pos()
            self._drag_live  = None

    def mouseMoveEvent(self, e: QMouseEvent) -> None:
        if self.mode == self.MODE_RECT and self._drag_start:
            self._drag_live = QRect(self._drag_start, e.pos()).normalized()
            self._redraw()

    def mouseReleaseEvent(self, e: QMouseEvent) -> None:
        if self.mode != self.MODE_RECT or not self._drag_start or e.button() != Qt.LeftButton:
            return
        end_rect          = QRect(self._drag_start, e.pos()).normalized()
        self._drag_start  = None
        self._drag_live   = None
        tl = self._widget_to_image(end_rect.topLeft())
        br = self._widget_to_image(end_rect.bottomRight())
        if tl and br:
            self._sel_rect = QRect(QPoint(*tl), QPoint(*br)).normalized()
            self.rect_roi_selected.emit(
                self._sel_rect.x(), self._sel_rect.y(),
                self._sel_rect.width(), self._sel_rect.height(),
            )
        self._redraw()

    def resizeEvent(self, e) -> None:
        super().resizeEvent(e)
        self._redraw()


# ─────────────────────────────────────────────────────────────────────────────
#  Main application window
# ─────────────────────────────────────────────────────────────────────────────

class TouchPoseEstimatorApp(QMainWindow):

    _C_BLUE   = f"rgb{ZividColors.DARK_BLUE}"
    _C_GREEN  = "rgb(60, 150, 90)"
    _C_PURPLE = "rgb(130, 70, 200)"

    def __init__(self, zivid_app: zivid.Application):
        super().__init__()
        self._zivid_app              = zivid_app
        self._camera:                Optional[zivid.Camera]  = None
        self._xyz:                   Optional[np.ndarray]    = None
        self._rgba:                  Optional[np.ndarray]    = None
        self._pose:                  Optional[np.ndarray]    = None  # camera-frame pose
        self._robot_pose:            Optional[np.ndarray]    = None  # robot-base-frame pose
        self._handeye_matrix:        Optional[np.ndarray]    = None
        self._robot_capture_matrix:  Optional[np.ndarray]    = None
        self._o3d_thread:            Optional[_Open3DThread] = None
        self._build_ui()

    # ─────────────────────────────────────────────────────────────────────────
    #  UI construction
    # ─────────────────────────────────────────────────────────────────────────

    def _build_ui(self) -> None:
        self.setWindowTitle("2D Point-Based Touch Pose Estimator")
        self.setMinimumSize(1420, 760)

        root = QWidget()
        self.setCentralWidget(root)
        root_lay = QVBoxLayout(root)
        root_lay.setContentsMargins(6, 6, 6, 4)
        root_lay.setSpacing(6)

        root_lay.addWidget(self._make_toolbar())

        splitter = QSplitter(Qt.Horizontal)
        splitter.setHandleWidth(5)
        self._viewer = ImageViewer()
        self._viewer.point_selected.connect(self._on_point_selected)
        self._viewer.rect_roi_selected.connect(self._on_rect_roi_selected)
        splitter.addWidget(self._viewer)
        splitter.addWidget(self._make_right_panel())
        splitter.setSizes([880, 500])
        root_lay.addWidget(splitter, stretch=1)

        self._sb = QStatusBar()
        self.setStatusBar(self._sb)
        self._status("Ready  —  load a ZDF file or connect a camera")

    # ── Toolbar ───────────────────────────────────────────────────────────────

    def _make_toolbar(self) -> QWidget:
        bar = QWidget()
        bar.setFixedHeight(52)
        bar.setStyleSheet("background-color: #2a2a2a; border-radius: 4px;")
        lay = QHBoxLayout(bar)
        lay.setContentsMargins(8, 6, 8, 6)
        lay.setSpacing(8)

        self._btn_load    = self._mk_btn("📂  Load ZDF",        self._C_BLUE,  self._load_zdf)
        self._btn_connect = self._mk_btn("📷  Connect Camera",  self._C_BLUE,  self._connect_camera)
        self._btn_capture = self._mk_btn("⚡  Capture",         self._C_GREEN, self._capture)
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
        scroll.setMinimumWidth(460)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)

        inner = QWidget()
        lay   = QVBoxLayout(inner)
        lay.setSpacing(10)
        lay.setContentsMargins(8, 8, 8, 8)

        lay.addWidget(self._make_group_touch_point())
        lay.addWidget(self._make_group_roi())
        lay.addWidget(self._make_group_xaxis())
        lay.addWidget(self._make_group_normal())
        lay.addWidget(self._make_group_pose())
        lay.addStretch()

        self._btn_estimate  = self._mk_btn("🎯  Estimate Pose",   self._C_PURPLE, self._estimate_pose)
        self._btn_visualize = self._mk_btn("🌐  Visualize in 3D", self._C_BLUE,   self._visualize_3d)
        self._btn_estimate.setFixedHeight(44)
        self._btn_visualize.setFixedHeight(44)
        self._btn_estimate.setEnabled(False)
        self._btn_visualize.setEnabled(False)
        lay.addWidget(self._btn_estimate)
        lay.addWidget(self._btn_visualize)
        lay.addWidget(self._make_group_advanced())

        scroll.setWidget(inner)
        return scroll

    def _make_group_touch_point(self) -> QGroupBox:
        grp = QGroupBox("Touch Point")
        f   = QFormLayout(grp)
        f.setSpacing(6)

        self._spin_u = QSpinBox(); self._spin_u.setRange(0, 9999)
        self._spin_v = QSpinBox(); self._spin_v.setRange(0, 9999)
        self._spin_u.editingFinished.connect(self._on_uv_edited)
        self._spin_v.editingFinished.connect(self._on_uv_edited)
        f.addRow("u  (col):", self._spin_u)
        f.addRow("v  (row):", self._spin_v)

        self._lbl_xyz = QLabel("X: —\nY: —\nZ: —")
        self._lbl_xyz.setFont(QFont("Courier New", 10))
        self._lbl_xyz.setStyleSheet("color: #80ffb0;")
        f.addRow("3D (mm):", self._lbl_xyz)
        return grp

    def _make_group_roi(self) -> QGroupBox:
        grp     = QGroupBox("Region of Interest")
        grp_lay = QVBoxLayout(grp)
        grp_lay.setSpacing(6)

        mode_row = QHBoxLayout()
        self._radio_radius = QRadioButton("Radius")
        self._radio_rect   = QRadioButton("Rectangle  (drag on image)")
        self._radio_radius.setChecked(True)
        self._radio_radius.toggled.connect(self._on_roi_mode_changed)
        mode_row.addWidget(self._radio_radius)
        mode_row.addWidget(self._radio_rect)
        grp_lay.addLayout(mode_row)

        r_row = QHBoxLayout()
        r_row.addWidget(QLabel("r ="))
        self._spin_r = QDoubleSpinBox()
        self._spin_r.setRange(1.0, 500.0)
        self._spin_r.setValue(10.0)
        self._spin_r.setSuffix("  mm")
        self._spin_r.setSingleStep(1.0)
        self._spin_r.valueChanged.connect(self._on_radius_changed)
        r_row.addWidget(self._spin_r)
        r_row.addStretch()
        grp_lay.addLayout(r_row)

        self._lbl_roi = QLabel("No ROI defined")
        self._lbl_roi.setFont(QFont("Courier New", 10))
        self._lbl_roi.setStyleSheet("color: #aaaaaa;")
        self._lbl_roi.setWordWrap(True)
        grp_lay.addWidget(self._lbl_roi)
        return grp

    def _make_group_xaxis(self) -> QGroupBox:
        grp = QGroupBox("X-axis Mode")
        lay = QVBoxLayout(grp)
        lay.setSpacing(4)
        self._radio_xaxis_svd  = QRadioButton(
            "SVD 주축  —  카메라 +X 방향으로 부호 정렬")
        self._radio_xaxis_proj = QRadioButton(
            "Camera +X 투영  —  이미지 좌→우 방향을 표면에 투영")
        self._radio_xaxis_svd.setChecked(True)
        lay.addWidget(self._radio_xaxis_svd)
        lay.addWidget(self._radio_xaxis_proj)
        return grp

    def _make_group_normal(self) -> QGroupBox:
        grp = QGroupBox("Surface Normal  (Z-axis)")
        lay = QVBoxLayout(grp)
        self._lbl_n = QLabel("nx: —\nny: —\nnz: —")
        self._lbl_n.setFont(QFont("Courier New", 10))
        self._lbl_n.setStyleSheet("color: #aad4ff;")
        lay.addWidget(self._lbl_n)
        return grp

    def _make_group_pose(self) -> QGroupBox:
        grp = QGroupBox("Camera Frame Pose  4 × 4")
        lay = QVBoxLayout(grp)
        self._txt_pose = QTextEdit()
        self._txt_pose.setReadOnly(True)
        self._txt_pose.setFont(QFont("Courier New", 10))
        self._txt_pose.setFixedHeight(116)
        self._txt_pose.setPlaceholderText("—  click  Estimate Pose  —")
        lay.addWidget(self._txt_pose)
        return grp

    def _make_group_advanced(self) -> QGroupBox:
        self._grp_advanced = QGroupBox("Advanced — Robot Base Pose")
        self._grp_advanced.setCheckable(True)
        self._grp_advanced.setChecked(False)

        lay = QVBoxLayout(self._grp_advanced)
        lay.setSpacing(8)

        # ── Hand-Eye calibration ───────────────────────────────────────────
        he_grp = QGroupBox("Hand-Eye Calibration")
        he_lay = QVBoxLayout(he_grp)
        he_lay.setSpacing(4)

        src_row = QHBoxLayout()
        lbl_src = QLabel("Input:")
        lbl_src.setFixedWidth(42)
        self._radio_he_file   = QRadioButton("File")
        self._radio_he_manual = QRadioButton("Manual")
        self._radio_he_file.setChecked(True)
        self._radio_he_file.toggled.connect(self._on_he_src_changed)
        src_row.addWidget(lbl_src)
        src_row.addWidget(self._radio_he_file)
        src_row.addWidget(self._radio_he_manual)
        src_row.addStretch()
        he_lay.addLayout(src_row)

        self._he_file_widget = QWidget()
        hfw = QHBoxLayout(self._he_file_widget)
        hfw.setContentsMargins(0, 0, 0, 0)
        btn_he = QPushButton("📂 Load YAML")
        btn_he.setFixedHeight(28)
        btn_he.clicked.connect(self._load_handeye_yaml)
        self._lbl_he_file = QLabel("—  not loaded")
        self._lbl_he_file.setFont(QFont("Courier New", 9))
        self._lbl_he_file.setStyleSheet("color: #888888;")
        hfw.addWidget(btn_he)
        hfw.addWidget(self._lbl_he_file, stretch=1)
        he_lay.addWidget(self._he_file_widget)

        self._he_manual_widget = QWidget()
        hmw = QVBoxLayout(self._he_manual_widget)
        hmw.setContentsMargins(0, 0, 0, 0)
        hmw.setSpacing(3)
        self._txt_he_manual = QTextEdit()
        self._txt_he_manual.setFont(QFont("Courier New", 9))
        self._txt_he_manual.setFixedHeight(76)
        self._txt_he_manual.setPlaceholderText(
            "r11 r12 r13 tx\nr21 r22 r23 ty\nr31 r32 r33 tz\n0   0   0   1")
        btn_he_apply = QPushButton("Apply")
        btn_he_apply.setFixedHeight(24)
        btn_he_apply.clicked.connect(self._apply_handeye_manual)
        hmw.addWidget(self._txt_he_manual)
        hmw.addWidget(btn_he_apply)
        he_lay.addWidget(self._he_manual_widget)
        self._he_manual_widget.setVisible(False)

        lay.addWidget(he_grp)

        # ── Eye configuration ──────────────────────────────────────────────
        eye_row = QHBoxLayout()
        lbl_cfg = QLabel("Config:")
        lbl_cfg.setFixedWidth(50)
        self._radio_eye2hand    = QRadioButton("Eye-to-Hand")
        self._radio_eye_in_hand = QRadioButton("Eye-in-Hand")
        self._radio_eye2hand.setChecked(True)
        self._radio_eye2hand.toggled.connect(self._on_eye_type_changed)
        self._eye_type_group = QButtonGroup(self)
        self._eye_type_group.addButton(self._radio_eye2hand)
        self._eye_type_group.addButton(self._radio_eye_in_hand)
        eye_row.addWidget(lbl_cfg)
        eye_row.addWidget(self._radio_eye2hand)
        eye_row.addWidget(self._radio_eye_in_hand)
        eye_row.addStretch()
        lay.addLayout(eye_row)

        # ── Robot capture pose (eye-in-hand only) ─────────────────────────
        self._rc_group = QGroupBox("Robot Capture Pose")
        self._rc_group.setEnabled(False)
        rc_lay = QVBoxLayout(self._rc_group)
        rc_lay.setSpacing(4)

        rc_src_row = QHBoxLayout()
        lbl_rc_src = QLabel("Input:")
        lbl_rc_src.setFixedWidth(42)
        self._radio_rc_file   = QRadioButton("File")
        self._radio_rc_manual = QRadioButton("Manual")
        self._radio_rc_file.setChecked(True)
        self._radio_rc_file.toggled.connect(self._on_rc_src_changed)
        rc_src_row.addWidget(lbl_rc_src)
        rc_src_row.addWidget(self._radio_rc_file)
        rc_src_row.addWidget(self._radio_rc_manual)
        rc_src_row.addStretch()
        rc_lay.addLayout(rc_src_row)

        self._rc_file_widget = QWidget()
        rfw = QHBoxLayout(self._rc_file_widget)
        rfw.setContentsMargins(0, 0, 0, 0)
        btn_rc = QPushButton("📂 Load YAML")
        btn_rc.setFixedHeight(28)
        btn_rc.clicked.connect(self._load_robot_capture_yaml)
        self._lbl_rc_file = QLabel("—  not loaded")
        self._lbl_rc_file.setFont(QFont("Courier New", 9))
        self._lbl_rc_file.setStyleSheet("color: #888888;")
        rfw.addWidget(btn_rc)
        rfw.addWidget(self._lbl_rc_file, stretch=1)
        rc_lay.addWidget(self._rc_file_widget)

        self._rc_manual_widget = QWidget()
        rmw = QVBoxLayout(self._rc_manual_widget)
        rmw.setContentsMargins(0, 0, 0, 0)
        rmw.setSpacing(3)
        self._txt_rc_manual = QTextEdit()
        self._txt_rc_manual.setFont(QFont("Courier New", 9))
        self._txt_rc_manual.setFixedHeight(76)
        self._txt_rc_manual.setPlaceholderText(
            "r11 r12 r13 tx\nr21 r22 r23 ty\nr31 r32 r33 tz\n0   0   0   1")
        btn_rc_apply = QPushButton("Apply")
        btn_rc_apply.setFixedHeight(24)
        btn_rc_apply.clicked.connect(self._apply_rc_manual)
        rmw.addWidget(self._txt_rc_manual)
        rmw.addWidget(btn_rc_apply)
        rc_lay.addWidget(self._rc_manual_widget)
        self._rc_manual_widget.setVisible(False)

        lay.addWidget(self._rc_group)

        # ── 6DoF format selector ───────────────────────────────────────────
        fmt_row = QHBoxLayout()
        lbl_fmt = QLabel("6DoF:")
        lbl_fmt.setFixedWidth(42)
        self._combo_6dof = QComboBox()
        for lbl in _6DOF_LABELS:
            self._combo_6dof.addItem(lbl)
        self._combo_6dof.setFont(QFont("", 9))
        self._combo_6dof.currentIndexChanged.connect(self._refresh_robot_pose_display)
        fmt_row.addWidget(lbl_fmt)
        fmt_row.addWidget(self._combo_6dof, stretch=1)
        lay.addLayout(fmt_row)

        # ── Compute button ────────────────────────────────────────────────
        self._btn_compute_robot = self._mk_btn(
            "🤖  Compute Robot Pose", self._C_PURPLE, self._compute_robot_pose_action)
        self._btn_compute_robot.setFixedHeight(38)
        self._btn_compute_robot.setEnabled(False)
        lay.addWidget(self._btn_compute_robot)

        # ── Result display ─────────────────────────────────────────────────
        lay.addWidget(QLabel("Robot Base Pose:"))
        self._txt_robot_pose = QTextEdit()
        self._txt_robot_pose.setReadOnly(True)
        self._txt_robot_pose.setFont(QFont("Courier New", 9))
        self._txt_robot_pose.setFixedHeight(142)
        self._txt_robot_pose.setPlaceholderText("—  estimate pose to compute  —")
        lay.addWidget(self._txt_robot_pose)

        return self._grp_advanced

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
            QPushButton:hover    {{ background-color: {bg}; border: 1px solid #ffffff55; }}
            QPushButton:disabled {{ background-color: rgb(70,70,70); color: rgb(140,140,140); }}
        """)
        btn.clicked.connect(cb)
        return btn

    # ─────────────────────────────────────────────────────────────────────────
    #  Slots — toolbar controls
    # ─────────────────────────────────────────────────────────────────────────

    def _on_roi_mode_changed(self) -> None:
        if self._radio_radius.isChecked():
            self._spin_r.setEnabled(True)
            self._viewer.set_mode(ImageViewer.MODE_POINT)
        else:
            self._spin_r.setEnabled(False)
            self._viewer.set_mode(ImageViewer.MODE_RECT)

    def _on_radius_changed(self, val: float) -> None:
        self._viewer.set_roi_radius(val)
        if self._radio_radius.isChecked():
            self._lbl_roi.setText(f"Radius:  {val:.1f} mm")

    # ─────────────────────────────────────────────────────────────────────────
    #  Slots — image viewer interactions
    # ─────────────────────────────────────────────────────────────────────────

    def _on_point_selected(self, u: int, v: int) -> None:
        self._spin_u.setValue(u)
        self._spin_v.setValue(v)
        self._refresh_xyz_label(u, v)
        self._lbl_roi.setText(f"Radius:  {self._spin_r.value():.1f} mm")
        self._btn_estimate.setEnabled(True)

    def _on_uv_edited(self) -> None:
        u, v = self._spin_u.value(), self._spin_v.value()
        self._viewer.set_selected_point(u, v)
        self._refresh_xyz_label(u, v)
        if self._xyz is not None:
            self._btn_estimate.setEnabled(True)

    def _on_rect_roi_selected(self, x: int, y: int, w: int, h: int) -> None:
        self._lbl_roi.setText(f"Rect  x={x}  y={y}\n      w={w}  h={h}")
        if self._xyz is not None:
            self._btn_estimate.setEnabled(True)

    def _refresh_xyz_label(self, u: int, v: int) -> None:
        if self._xyz is None:
            return
        H, W = self._xyz.shape[:2]
        if not (0 <= v < H and 0 <= u < W):
            return
        pt = self._xyz[v, u]
        if np.any(np.isnan(pt)):
            self._lbl_xyz.setText("X: NaN  (no depth)")
        else:
            self._lbl_xyz.setText(f"X: {pt[0]:9.2f}\nY: {pt[1]:9.2f}\nZ: {pt[2]:9.2f}")

    # ─────────────────────────────────────────────────────────────────────────
    #  Slots — Advanced panel
    # ─────────────────────────────────────────────────────────────────────────

    def _on_he_src_changed(self) -> None:
        is_file = self._radio_he_file.isChecked()
        self._he_file_widget.setVisible(is_file)
        self._he_manual_widget.setVisible(not is_file)

    def _on_rc_src_changed(self) -> None:
        is_file = self._radio_rc_file.isChecked()
        self._rc_file_widget.setVisible(is_file)
        self._rc_manual_widget.setVisible(not is_file)

    def _on_eye_type_changed(self) -> None:
        self._rc_group.setEnabled(self._radio_eye_in_hand.isChecked())

    # ─────────────────────────────────────────────────────────────────────────
    #  Data loading
    # ─────────────────────────────────────────────────────────────────────────

    def _load_zdf(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Open ZDF File", "", "ZDF Files (*.zdf)")
        if not path:
            return
        self._status(f"Loading  {Path(path).name} …")
        try:
            self._ingest_frame(zivid.Frame(path))
            self._status(f"Loaded  {Path(path).name}")
        except Exception as exc:
            self._status(f"Load error: {exc}", error=True)
            QMessageBox.critical(self, "Load Error", str(exc))

    def _connect_camera(self) -> None:
        self._status("Connecting to camera …")
        try:
            self._camera = self._zivid_app.connect_camera()
            self._btn_capture.setEnabled(True)
            self._status(f"Connected:  {self._camera.info.model}")
        except Exception as exc:
            self._status(f"Connection failed: {exc}", error=True)
            QMessageBox.critical(self, "Camera Error", str(exc))

    def _capture(self) -> None:
        if self._camera is None:
            return
        self._status("Capturing …")
        try:
            settings = zivid.capture_assistant.suggest_settings(
                self._camera,
                zivid.capture_assistant.SuggestSettingsParameters(
                    max_capture_time=datetime.timedelta(milliseconds=1200),
                    ambient_light_frequency=(
                        zivid.capture_assistant.SuggestSettingsParameters.AmbientLightFrequency.none
                    ),
                ),
            )
            self._ingest_frame(self._camera.capture_2d_3d(settings))
            self._status("Capture complete")
        except Exception as exc:
            self._status(f"Capture error: {exc}", error=True)
            QMessageBox.critical(self, "Capture Error", str(exc))

    def _ingest_frame(self, frame: zivid.Frame) -> None:
        pc         = frame.point_cloud()
        self._xyz  = pc.copy_data("xyz")
        self._rgba = pc.copy_data("rgba")
        H, W       = self._xyz.shape[:2]
        self._spin_u.setMaximum(W - 1)
        self._spin_v.setMaximum(H - 1)
        self._viewer.load_rgb(self._rgba[:, :, :3])
        self._viewer.set_roi_radius(self._spin_r.value())
        self._pose       = None
        self._robot_pose = None
        self._txt_pose.clear()
        self._txt_robot_pose.clear()
        self._lbl_n.setText("nx: —\nny: —\nnz: —")
        self._lbl_xyz.setText("X: —\nY: —\nZ: —")
        self._lbl_roi.setText("No ROI defined")
        self._btn_estimate.setEnabled(False)
        self._btn_visualize.setEnabled(False)
        self._btn_compute_robot.setEnabled(False)

    def _load_handeye_yaml(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Open Hand-Eye Calibration YAML", "", "YAML Files (*.yaml *.yml)")
        if not path:
            return
        try:
            self._handeye_matrix = load_and_assert_affine_matrix(Path(path))
            self._lbl_he_file.setText(Path(path).name)
            self._lbl_he_file.setStyleSheet("color: #80ffb0;")
            self._status(f"Hand-Eye loaded: {Path(path).name}")
        except Exception as exc:
            self._status(f"Hand-Eye load error: {exc}", error=True)
            QMessageBox.critical(self, "Load Error", str(exc))

    def _load_robot_capture_yaml(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Open Robot Capture Pose YAML", "", "YAML Files (*.yaml *.yml)")
        if not path:
            return
        try:
            self._robot_capture_matrix = load_and_assert_affine_matrix(Path(path))
            self._lbl_rc_file.setText(Path(path).name)
            self._lbl_rc_file.setStyleSheet("color: #80ffb0;")
            self._status(f"Robot capture pose loaded: {Path(path).name}")
        except Exception as exc:
            self._status(f"Robot capture pose load error: {exc}", error=True)
            QMessageBox.critical(self, "Load Error", str(exc))

    def _apply_handeye_manual(self) -> None:
        try:
            self._handeye_matrix = self._parse_matrix_text(self._txt_he_manual.toPlainText())
            self._status("Hand-Eye matrix applied from manual input")
        except Exception as exc:
            QMessageBox.critical(self, "Input Error", str(exc))

    def _apply_rc_manual(self) -> None:
        try:
            self._robot_capture_matrix = self._parse_matrix_text(self._txt_rc_manual.toPlainText())
            self._status("Robot capture pose applied from manual input")
        except Exception as exc:
            QMessageBox.critical(self, "Input Error", str(exc))

    # ─────────────────────────────────────────────────────────────────────────
    #  Pose estimation
    # ─────────────────────────────────────────────────────────────────────────

    def _nearest_valid_3d(self, u: int, v: int) -> Optional[np.ndarray]:
        """Return the closest non-NaN 3D point to pixel (u, v) in pixel distance."""
        valid_mask = ~np.isnan(self._xyz).any(axis=2)
        ys, xs = np.where(valid_mask)
        if len(xs) == 0:
            return None
        dists = (xs - u) ** 2 + (ys - v) ** 2
        idx = int(np.argmin(dists))
        return self._xyz[ys[idx], xs[idx]]

    def _get_roi_points(self) -> Optional[np.ndarray]:
        """Return (N, 3) array of valid 3D points inside the selected ROI."""
        if self._xyz is None:
            return None

        if self._radio_radius.isChecked():
            u, v   = self._spin_u.value(), self._spin_v.value()
            anchor = self._xyz[v, u]
            if np.any(np.isnan(anchor)):
                anchor = self._nearest_valid_3d(u, v)
                if anchor is None:
                    return None
            dist = np.linalg.norm(self._xyz - anchor, axis=2)
            pts  = self._xyz[dist <= self._spin_r.value()]
        else:
            roi = self._viewer._sel_rect
            if roi is None:
                return None
            H, W  = self._xyz.shape[:2]
            x, y  = max(0, roi.x()), max(0, roi.y())
            x2    = min(x + roi.width(),  W)
            y2    = min(y + roi.height(), H)
            pts   = self._xyz[y:y2, x:x2].reshape(-1, 3)

        return pts[~np.isnan(pts).any(axis=1)]

    def _estimate_pose(self) -> None:
        if self._xyz is None:
            return

        pts = self._get_roi_points()
        if pts is None or len(pts) < 3:
            QMessageBox.warning(
                self, "Insufficient Points",
                "Not enough valid 3D points in the ROI.\n"
                "Try a larger radius or select a different region.",
            )
            return

        self._status(f"SVD plane fitting on {len(pts):,} points …")

        centroid = pts.mean(axis=0)
        M        = (pts - centroid).T @ (pts - centroid)
        U        = np.linalg.svd(M)[0]   # columns: [dominant, mid, normal]

        touch_nan = False
        if self._radio_radius.isChecked():
            touch_3d = self._xyz[self._spin_v.value(), self._spin_u.value()]
            if np.any(np.isnan(touch_3d)):
                touch_3d  = centroid
                touch_nan = True
        else:
            touch_3d = centroid

        # Z-axis: surface normal pointing away from camera (+Z in Zivid frame)
        z_ax = U[:, 2]
        if z_ax[2] < 0:
            z_ax = -z_ax

        # X-axis: mode selection
        if self._radio_xaxis_svd.isChecked():
            # SVD principal axis, sign-corrected to align with camera +X
            x_ax = U[:, 0]
            if x_ax[0] < 0:
                x_ax = -x_ax
        else:
            # Camera +X [1,0,0] projected onto the surface plane
            cam_x = np.array([1.0, 0.0, 0.0])
            x_ax  = cam_x - np.dot(cam_x, z_ax) * z_ax
            norm  = np.linalg.norm(x_ax)
            if norm < 1e-6:
                # Degenerate: surface normal is parallel to camera X — fall back to camera Y
                cam_y = np.array([0.0, 1.0, 0.0])
                x_ax  = cam_y - np.dot(cam_y, z_ax) * z_ax
                norm  = np.linalg.norm(x_ax)
            x_ax /= norm

        y_ax = np.cross(z_ax, x_ax)
        y_ax /= np.linalg.norm(y_ax)
        x_ax  = np.cross(y_ax, z_ax)
        x_ax /= np.linalg.norm(x_ax)

        pose = np.eye(4)
        pose[:3, 0] = x_ax
        pose[:3, 1] = y_ax
        pose[:3, 2] = z_ax
        pose[:3, 3] = touch_3d
        self._pose  = pose

        self._txt_pose.setText(self._fmt_mat(pose))
        nx, ny, nz = z_ax
        self._lbl_n.setText(f"nx: {nx:+.5f}\nny: {ny:+.5f}\nnz: {nz:+.5f}")
        self._btn_visualize.setEnabled(True)
        self._btn_compute_robot.setEnabled(True)
        self._robot_pose = None
        self._txt_robot_pose.clear()
        status_suffix = "  ⚠ touch pixel NaN → centroid used" if touch_nan else ""
        self._status(
            f"Pose estimated  |  {len(pts):,} pts  "
            f"|  touch Z = {touch_3d[2]:.1f} mm{status_suffix}"
        )
        self._print_result(pose, touch_3d, z_ax, len(pts), touch_nan=touch_nan)

    def _visualize_3d(self) -> None:
        if self._pose is None or self._xyz is None:
            return
        if self._o3d_thread and self._o3d_thread.isRunning():
            self._status("3D viewer is already open")
            return

        roi_pts = self._get_roi_points()
        self._o3d_thread = _Open3DThread(self._xyz, self._rgba, self._pose, roi_pts)
        self._o3d_thread.done.connect(lambda: self._status("3D viewer closed"))
        self._o3d_thread.start()
        self._status("3D viewer opened")

    # ─────────────────────────────────────────────────────────────────────────
    #  Advanced helpers
    # ─────────────────────────────────────────────────────────────────────────

    def _parse_matrix_text(self, text: str) -> np.ndarray:
        try:
            nums = [float(x) for x in text.split()]
        except ValueError as exc:
            raise ValueError(f"Could not parse number: {exc}") from exc
        if len(nums) != 16:
            raise ValueError(f"Expected 16 numbers, got {len(nums)}")
        m = np.array(nums, dtype=float).reshape(4, 4)
        try:
            zivid.calibration.Pose(m)
        except RuntimeError as exc:
            raise ValueError(f"Not a valid affine matrix: {exc}") from exc
        return m

    def _compute_robot_base_pose(self, T_camera: np.ndarray) -> np.ndarray:
        if self._radio_eye2hand.isChecked():
            return self._handeye_matrix @ T_camera
        return self._robot_capture_matrix @ self._handeye_matrix @ T_camera

    def _pose_to_6dof_str(self, T: np.ndarray) -> Optional[str]:
        """Returns compact position + rotation string for the selected 6DoF format.
        Returns None when the format is '4×4 Matrix' (handled separately)."""
        tx, ty, tz = T[:3, 3]
        rot = Rotation.from_matrix(T[:3, :3])
        idx = self._combo_6dof.currentIndex()
        pos = f"t   {tx:+11.3f}  {ty:+11.3f}  {tz:+11.3f}   mm"

        if idx == 0:
            return None
        if idx == 1:
            rv = rot.as_rotvec()
            return (f"{pos}\n"
                    f"rv  {rv[0]:+11.6f}  {rv[1]:+11.6f}  {rv[2]:+11.6f}   rad")
        if idx == 2:
            q = rot.as_quat()  # [x, y, z, w]
            return (f"{pos}\n"
                    f"q   {q[0]:+11.6f}  {q[1]:+11.6f}  {q[2]:+11.6f}  {q[3]:+11.6f}   [x y z w]")
        if idx == 3:
            a = rot.as_euler('xyz', degrees=True)
            return (f"{pos}\n"
                    f"Roll {a[0]:+9.4f}   Pitch {a[1]:+9.4f}   Yaw {a[2]:+9.4f}   deg")
        if idx == 4:
            a = rot.as_euler('zyx', degrees=True)
            return (f"{pos}\n"
                    f"Yaw  {a[0]:+9.4f}   Pitch {a[1]:+9.4f}   Roll {a[2]:+9.4f}   deg")
        if idx == 5:
            a = rot.as_euler('ZYX', degrees=True)
            return (f"{pos}\n"
                    f"A(Z) {a[0]:+9.4f}   B(Y) {a[1]:+9.4f}   C(X) {a[2]:+9.4f}   deg")
        if idx == 6:
            a = rot.as_euler('zyz', degrees=True)
            return (f"{pos}\n"
                    f"α(Z) {a[0]:+9.4f}   β(Y) {a[1]:+9.4f}   γ(Z) {a[2]:+9.4f}   deg")
        if idx == 7:
            a = rot.as_euler('ZYZ', degrees=True)
            return (f"{pos}\n"
                    f"α(Z) {a[0]:+9.4f}   β(Y) {a[1]:+9.4f}   γ(Z) {a[2]:+9.4f}   deg")
        # idx == 8
        a = rot.as_euler('XYZ', degrees=True)
        return (f"{pos}\n"
                f"α(X) {a[0]:+9.4f}   β(Y) {a[1]:+9.4f}   γ(Z) {a[2]:+9.4f}   deg")

    def _refresh_robot_pose_display(self) -> None:
        if self._robot_pose is None:
            return
        mat_str = self._fmt_mat(self._robot_pose)
        dof_str = self._pose_to_6dof_str(self._robot_pose)
        self._txt_robot_pose.setText(
            mat_str + ("\n─────\n" + dof_str if dof_str else ""))

    def _compute_robot_pose_action(self) -> None:
        if self._pose is None:
            return
        if self._handeye_matrix is None:
            QMessageBox.warning(
                self, "Advanced — Missing Input",
                "Hand-Eye calibration matrix is not loaded.\n"
                "Please load a YAML file or enter the matrix manually.",
            )
            return
        if self._radio_eye_in_hand.isChecked() and self._robot_capture_matrix is None:
            QMessageBox.warning(
                self, "Advanced — Missing Input",
                "Robot Capture Pose is not loaded (required for Eye-in-Hand).\n"
                "Please load a YAML file or enter the matrix manually.",
            )
            return
        self._robot_pose = self._compute_robot_base_pose(self._pose)
        self._refresh_robot_pose_display()
        self._status("Robot base pose computed")
        self._print_robot_result()

    def _print_robot_result(self) -> None:
        if self._robot_pose is None:
            return
        W   = 67
        bar = "─" * W

        def row(text: str) -> str:
            return f"│  {text:<{W - 2}}│"

        eye_type  = "Eye-to-Hand" if self._radio_eye2hand.isChecked() else "Eye-in-Hand"
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d  %H:%M:%S")
        rb_rows   = ["  ".join(f"{v:+10.4f}" for v in self._robot_pose[i]) for i in range(4)]
        dof_str   = self._pose_to_6dof_str(self._robot_pose)
        dof_lbl   = _6DOF_LABELS[self._combo_6dof.currentIndex()]
        lines = [
            f"┌{bar}┐",
            row(f"ROBOT BASE POSE  [{eye_type}]  ·  {timestamp}"),
            f"├{bar}┤",
            row("4×4  [R|t]  (mm)"),
            *[row(f"  {r}") for r in rb_rows],
        ]
        if dof_str:
            lines += [
                f"├{bar}┤",
                row(f"6DoF  [{dof_lbl}]"),
                *[row(f"  {line}") for line in dof_str.split("\n")],
            ]
        lines.append(f"└{bar}┘")
        print("\n" + "\n".join(lines) + "\n")

    # ─────────────────────────────────────────────────────────────────────────
    #  Output helpers
    # ─────────────────────────────────────────────────────────────────────────

    def _print_result(
        self,
        pose:      np.ndarray,
        touch_3d:  np.ndarray,
        normal:    np.ndarray,
        n_pts:     int,
        touch_nan: bool = False,
    ) -> None:
        W = 67
        bar = "─" * W

        def row(text: str) -> str:
            return f"│  {text:<{W - 2}}│"

        timestamp = datetime.datetime.now().strftime("%Y-%m-%d  %H:%M:%S")

        if self._radio_radius.isChecked():
            u, v = self._spin_u.value(), self._spin_v.value()
            roi_line = f"Radius   r = {self._spin_r.value():.1f} mm     pixel  u = {u}  v = {v}"
        else:
            r = self._viewer._sel_rect
            roi_line = (
                f"Rectangle   x = {r.x()}  y = {r.y()}  "
                f"w = {r.width()}  h = {r.height()}  px"
            )

        xmode = ("SVD 주축 (Camera +X 부호 정렬)"
                 if self._radio_xaxis_svd.isChecked()
                 else "Camera +X 투영")
        nx, ny, nz = normal
        tx, ty, tz = touch_3d
        mat_rows = ["  ".join(f"{v:+10.4f}" for v in pose[i]) for i in range(4)]

        lines = [
            f"┌{bar}┐",
            row(f"TOUCH POSE ESTIMATION  ·  {timestamp}"),
        ]
        if touch_nan:
            lines += [
                f"├{bar}┤",
                row("⚠  WARNING: Selected touch pixel has no depth (NaN)."),
                row("   Touch position set to ROI centroid instead."),
            ]
        lines += [
            f"├{bar}┤",
            row(f"ROI      {roi_line}"),
            row(f"X-axis   {xmode}"),
            row(f"Touch    X = {tx:+9.3f}   Y = {ty:+9.3f}   Z = {tz:+9.3f}   mm"),
            row(f"Points   {n_pts:,}"),
            row(f"Normal   nx = {nx:+.5f}   ny = {ny:+.5f}   nz = {nz:+.5f}"),
            f"├{bar}┤",
            row("Camera Frame Pose  4×4  [R|t]  (mm)"),
            *[row(f"  {r}") for r in mat_rows],
        ]

        lines.append(f"└{bar}┘")
        print("\n" + "\n".join(lines) + "\n")

    @staticmethod
    def _fmt_mat(m: np.ndarray) -> str:
        return "\n".join(
            "  ".join(f"{v:9.4f}" for v in row)
            for row in m
        )

    def _status(self, msg: str, error: bool = False) -> None:
        self._sb.setStyleSheet("color: #ff7070;" if error else "color: #cccccc;")
        self._sb.showMessage(msg)

    def closeEvent(self, e) -> None:
        # Guard: if exec_() does not return on its own, fire forced kill from
        # inside the event loop after 800 ms.
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
    # os.kill(SIGTERM) calls TerminateProcess() via Python's properly-typed
    # Win32 binding, bypassing DLL_PROCESS_DETACH callbacks that Intel TBB
    # (used by Open3D) blocks in, causing os._exit() to hang.
    os.kill(os.getpid(), signal.SIGTERM)


def main() -> None:
    with ZividQtApplication() as qt_app:
        win = TouchPoseEstimatorApp(qt_app.zivid_app)
        qt_app.run(win, "2D Point-Based Touch Pose Estimator")
        _force_kill()


if __name__ == "__main__":
    main()
