"""
2D Point-Based Touch Pose Estimator  —  GUI Application

Run:
    python touch_pose_estimation.py

No CLI arguments needed. Everything is driven from the GUI:
  1. Load a ZDF file  OR  connect a camera and capture.
  2. Click a touch point on the 2D image  OR  enter (u, v) manually.
  3. Choose a ROI method: sphere radius around the point, or drag a rectangle.
  4. Click  "Estimate Pose"  →  SVD plane fit  →  4×4 pose matrix.
  5. Click  "Visualize in 3D"  →  Open3D viewer with pose frame + ROI highlight.
"""

import sys
import datetime
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import open3d as o3d
import zivid
from PyQt5.QtCore import Qt, QRect, QPoint, QThread, pyqtSignal, QSize
from PyQt5.QtGui import QFont, QImage, QPixmap, QPainter, QPen, QColor, QMouseEvent
from PyQt5.QtWidgets import (
    QMainWindow, QWidget, QLabel, QPushButton, QStatusBar,
    QHBoxLayout, QVBoxLayout, QFormLayout, QGroupBox,
    QSpinBox, QDoubleSpinBox, QRadioButton, QTextEdit,
    QFileDialog, QMessageBox, QSizePolicy, QScrollArea, QSplitter,
)
from zividsamples.gui.qt_application import ZividQtApplication, ZividColors


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

    def run(self) -> None:
        # Full scene point cloud
        pts = np.nan_to_num(self._xyz).reshape(-1, 3)
        rgb = self._rgba[:, :, :3].reshape(-1, 3).astype(float) / 255.0
        pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pts))
        pcd.colors = o3d.utility.Vector3dVector(rgb)
        pcd.remove_non_finite_points(remove_nan=True, remove_infinite=True)

        # Pose coordinate frame
        frame_mesh = o3d.geometry.TriangleMesh.create_coordinate_frame(size=20)
        frame_mesh.transform(self._pose)

        geometries = [pcd, frame_mesh]

        # ROI highlight
        if self._roi_pts is not None and len(self._roi_pts) > 0:
            roi_pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(self._roi_pts))
            roi_pcd.paint_uniform_color([1.0, 0.35, 0.1])
            geometries.append(roi_pcd)

        vis = o3d.visualization.Visualizer()
        vis.create_window("Touch Pose — 3D View", 1600, 900)
        for g in geometries:
            vis.add_geometry(g)

        opt = vis.get_render_option()
        opt.point_size        = 1.5
        opt.background_color  = [0.12, 0.12, 0.12]
        opt.show_coordinate_frame = True

        vc = vis.get_view_control()
        vc.set_front([0, 0, -1])
        vc.set_up([0, -1, 0])

        vis.run()
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
        self._sel_pt:      Optional[Tuple[int,int]] = None   # (u, v) image px
        self._roi_r:       float                    = 10.0
        self._drag_start:  Optional[QPoint]         = None
        self._drag_live:   Optional[QRect]          = None   # rect during drag (widget coords)
        self._sel_rect:    Optional[QRect]          = None   # confirmed (image coords)

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
        """Bounding rect of the scaled pixmap within the widget."""
        if self._pixmap_orig is None:
            return None
        pm = self._pixmap_orig.scaled(self.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        x = (self.width()  - pm.width())  // 2
        y = (self.height() - pm.height()) // 2
        return QRect(x, y, pm.width(), pm.height())

    def _widget_to_image(self, pt: QPoint) -> Optional[Tuple[int, int]]:
        """Widget coordinates → original image pixel coordinates."""
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
        """Image pixel coords → pixmap-local coords for painting."""
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

        # ── Selected point + ROI circle ──────────────────────────────────
        if self._sel_pt is not None and self.mode == self.MODE_POINT:
            px, py = self._image_to_pm_local(*self._sel_pt)

            # Dashed ROI circle (visual hint — radius is in mm, circle is approximate)
            r_vis = max(12.0, self._roi_r * pm.width() / 900.0)
            painter.setPen(QPen(QColor(255, 200, 40), 1.5, Qt.DashLine))
            painter.setBrush(Qt.NoBrush)
            painter.drawEllipse(QPoint(int(px), int(py)), int(r_vis), int(r_vis))

            # Crosshair
            arm = 16
            painter.setPen(QPen(QColor(50, 255, 110), 2))
            painter.drawLine(int(px - arm), int(py), int(px + arm), int(py))
            painter.drawLine(int(px), int(py - arm), int(px), int(py + arm))

            # Centre dot
            painter.setBrush(QColor(50, 255, 110))
            painter.setPen(Qt.NoPen)
            painter.drawEllipse(QPoint(int(px), int(py)), 5, 5)

        # ── Live drag rectangle ──────────────────────────────────────────
        if self._drag_live is not None and self.mode == self.MODE_RECT:
            dr = self._drag_live
            # Convert from widget coords to pm-local
            rx = dr.x() - r.x()
            ry = dr.y() - r.y()
            painter.setPen(QPen(QColor(255, 200, 40), 2, Qt.SolidLine))
            painter.setBrush(QColor(255, 200, 40, 45))
            painter.drawRect(rx, ry, dr.width(), dr.height())

        # ── Confirmed rectangle ───────────────────────────────────────────
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

    # ── Colours used for custom buttons ──────────────────────────────────────
    _C_BLUE   = "rgb(74, 143, 164)"
    _C_GREEN  = "rgb(60, 150, 90)"
    _C_PURPLE = "rgb(130, 70, 200)"
    _C_GRAY   = "rgb(82, 82, 82)"

    def __init__(self, zivid_app: zivid.Application):
        super().__init__()
        self._zivid_app   = zivid_app
        self._camera:     Optional[zivid.Camera]     = None
        self._xyz:        Optional[np.ndarray]       = None   # (H, W, 3) mm
        self._rgba:       Optional[np.ndarray]       = None   # (H, W, 4) uint8
        self._pose:       Optional[np.ndarray]       = None   # 4×4
        self._o3d_thread: Optional[_Open3DThread]    = None
        self._build_ui()

    # ─────────────────────────────────────────────────────────────────────────
    #  UI construction
    # ─────────────────────────────────────────────────────────────────────────

    def _build_ui(self) -> None:
        self.setWindowTitle("2D Point-Based Touch Pose Estimator")
        self.setMinimumSize(1350, 760)

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
        splitter.setSizes([960, 360])
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

        sep = QLabel("  Selection:")
        sep.setStyleSheet("color: #aaaaaa;")
        lay.addWidget(sep)
        self._radio_click = QRadioButton("Click Point")
        self._radio_drag  = QRadioButton("Drag Rectangle")
        self._radio_click.setChecked(True)
        self._radio_click.toggled.connect(self._on_sel_mode_changed)
        lay.addWidget(self._radio_click)
        lay.addWidget(self._radio_drag)

        return bar

    # ── Right panel ───────────────────────────────────────────────────────────

    def _make_right_panel(self) -> QScrollArea:
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFixedWidth(360)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)

        inner = QWidget()
        lay   = QVBoxLayout(inner)
        lay.setSpacing(10)
        lay.setContentsMargins(8, 8, 8, 8)

        lay.addWidget(self._make_group_touch_point())
        lay.addWidget(self._make_group_roi())
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

    def _make_group_normal(self) -> QGroupBox:
        grp = QGroupBox("Surface Normal  (Z-axis)")
        lay = QVBoxLayout(grp)
        self._lbl_n = QLabel("nx: —\nny: —\nnz: —")
        self._lbl_n.setFont(QFont("Courier New", 10))
        self._lbl_n.setStyleSheet("color: #aad4ff;")
        lay.addWidget(self._lbl_n)
        return grp

    def _make_group_pose(self) -> QGroupBox:
        grp = QGroupBox("Pose Matrix  4 × 4")
        lay = QVBoxLayout(grp)
        self._txt_pose = QTextEdit()
        self._txt_pose.setReadOnly(True)
        self._txt_pose.setFont(QFont("Courier New", 10))
        self._txt_pose.setFixedHeight(116)
        self._txt_pose.setPlaceholderText("—  click  Estimate Pose  —")
        lay.addWidget(self._txt_pose)
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
            QPushButton:hover    {{ background-color: {bg}; border: 1px solid #ffffff55; }}
            QPushButton:disabled {{ background-color: rgb(70,70,70); color: rgb(140,140,140); }}
        """)
        btn.clicked.connect(cb)
        return btn

    # ─────────────────────────────────────────────────────────────────────────
    #  Slots — toolbar controls
    # ─────────────────────────────────────────────────────────────────────────

    def _on_sel_mode_changed(self) -> None:
        mode = ImageViewer.MODE_POINT if self._radio_click.isChecked() else ImageViewer.MODE_RECT
        self._viewer.set_mode(mode)

    def _on_roi_mode_changed(self) -> None:
        if self._radio_radius.isChecked():
            self._spin_r.setEnabled(True)
            self._radio_click.setChecked(True)
            self._viewer.set_mode(ImageViewer.MODE_POINT)
        else:
            self._spin_r.setEnabled(False)
            self._radio_drag.setChecked(True)
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
            self._status(f"Connected:  {self._camera.info.model_name}")
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
        # reset results
        self._pose = None
        self._txt_pose.clear()
        self._lbl_n.setText("nx: —\nny: —\nnz: —")
        self._lbl_xyz.setText("X: —\nY: —\nZ: —")
        self._btn_estimate.setEnabled(False)
        self._btn_visualize.setEnabled(False)

    # ─────────────────────────────────────────────────────────────────────────
    #  Pose estimation
    # ─────────────────────────────────────────────────────────────────────────

    def _get_roi_points(self) -> Optional[np.ndarray]:
        """Return (N, 3) array of valid 3D points inside the selected ROI."""
        if self._xyz is None:
            return None

        if self._radio_radius.isChecked():
            u, v   = self._spin_u.value(), self._spin_v.value()
            anchor = self._xyz[v, u]
            if np.any(np.isnan(anchor)):
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

        # SVD plane fit
        centroid = pts.mean(axis=0)
        M        = (pts - centroid).T @ (pts - centroid)
        U        = np.linalg.svd(M)[0]   # columns: [dominant, mid, normal]

        # Touch origin: selected pixel in radius mode, centroid in rect mode
        if self._radio_radius.isChecked():
            touch_3d = self._xyz[self._spin_v.value(), self._spin_u.value()]
        else:
            touch_3d = centroid

        # Build 4×4 pose: Z aligned with surface normal
        z_ax = U[:, 2]
        x_ax = U[:, 0]
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

        # Update UI
        self._txt_pose.setText(self._fmt_mat(pose))
        nx, ny, nz = z_ax
        self._lbl_n.setText(f"nx: {nx:+.5f}\nny: {ny:+.5f}\nnz: {nz:+.5f}")
        self._btn_visualize.setEnabled(True)
        self._status(
            f"Pose estimated  |  {len(pts):,} pts  "
            f"|  touch Z = {touch_3d[2]:.1f} mm"
        )

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
    #  Helpers
    # ─────────────────────────────────────────────────────────────────────────

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
        if self._camera:
            try:
                self._camera.disconnect()
            except Exception:
                pass
        super().closeEvent(e)


# ─────────────────────────────────────────────────────────────────────────────
#  Entry point
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    with ZividQtApplication() as qt_app:
        win = TouchPoseEstimatorApp(qt_app.zivid_app)
        sys.exit(qt_app.run(win, "2D Point-Based Touch Pose Estimator"))


if __name__ == "__main__":
    main()
