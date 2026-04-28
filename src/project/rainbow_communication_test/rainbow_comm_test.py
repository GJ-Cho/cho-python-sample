"""
Rainbow Robotics RB Series — Communication Test GUI  (Phase 1)

Connects to a Rainbow RB robot via rbpodo (pip install rbpodo):
  - rb.CobotData(ip)  Port 5001 — data channel, polled in background thread
  - rb.Cobot(ip)      Port 5000 — command channel (reserved for Phase 2)

Displays in real time:
  - TCP Pose   [X  Y  Z  Rx  Ry  Rz]  —  mm + deg  OR  mm + rad (toggle)
  - Joint Angles [J1–J6]               —  always degrees
  - Robot State  (IDLE / MOVING / UNKNOWN)
  - Actual poll rate (Hz)

Run:
    python rainbow_comm_test.py
"""

import sys
import time
import datetime
from collections import deque
from typing import Optional

import numpy as np
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtGui import QFont, QPalette, QColor
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QLabel, QPushButton, QRadioButton,
    QLineEdit, QVBoxLayout, QHBoxLayout, QGridLayout, QFormLayout,
    QGroupBox, QButtonGroup, QTextEdit, QSizePolicy, QStatusBar, QSplitter,
)

try:
    import rbpodo as rb
    RB_AVAILABLE = True
except ImportError:
    RB_AVAILABLE = False

IP_DEFAULT = "192.168.0.1"
POLL_HZ    = 50.0          # target CobotData poll frequency


# ─────────────────────────────────────────────────────────────────────────────
#  CobotData polling thread  (Port 5001)
# ─────────────────────────────────────────────────────────────────────────────

class CobotDataThread(QThread):
    """Polls rb.CobotData at POLL_HZ and forwards SystemState to the GUI."""

    state_updated = pyqtSignal(object)        # SystemState object
    conn_changed  = pyqtSignal(bool, str)     # (connected, message)

    def __init__(self, ip: str, poll_hz: float = POLL_HZ):
        super().__init__()
        self._ip       = ip
        self._interval = 1.0 / poll_hz
        self._running  = False

    def stop(self) -> None:
        self._running = False

    def run(self) -> None:
        if not RB_AVAILABLE:
            self.conn_changed.emit(False, "rbpodo 패키지가 설치되지 않았습니다.  pip install rbpodo")
            return

        try:
            data_ch = rb.CobotData(self._ip)
        except Exception as exc:
            self.conn_changed.emit(False, f"CobotData 연결 실패: {exc}")
            return

        self._running = True
        self.conn_changed.emit(True, f"{self._ip}  (Port 5001)  연결 완료")

        while self._running:
            t0 = time.monotonic()

            try:
                state = data_ch.request_data(timeout=1.0)
            except Exception as exc:
                self.conn_changed.emit(False, f"데이터 수신 오류: {exc}")
                break

            if state is None:
                self.conn_changed.emit(False, "데이터 수신 없음 (timeout)")
                break

            self.state_updated.emit(state)

            elapsed = time.monotonic() - t0
            sleep_t = self._interval - elapsed
            if sleep_t > 0:
                time.sleep(sleep_t)

        self._running = False
        self.conn_changed.emit(False, "연결 종료")


# ─────────────────────────────────────────────────────────────────────────────
#  Main application window
# ─────────────────────────────────────────────────────────────────────────────

class RainbowCommApp(QMainWindow):

    _C_GREEN  = "rgb(45, 130, 75)"
    _C_RED    = "rgb(150, 55, 55)"
    _C_BLUE   = "rgb(45, 90, 160)"

    def __init__(self):
        super().__init__()
        self._data_thread:    Optional[CobotDataThread] = None
        self._last_tcp:       Optional[list]            = None
        self._last_state_t:   Optional[float]           = None
        self._hz_buf:         deque                     = deque(maxlen=30)
        self._build_ui()

    # ─────────────────────────────────────────────────────────────────────────
    #  UI construction
    # ─────────────────────────────────────────────────────────────────────────

    def _build_ui(self) -> None:
        self.setWindowTitle("Rainbow Robotics — RB Series Communication Test  (Phase 1)")
        self.setMinimumSize(960, 660)

        root = QWidget()
        lay  = QVBoxLayout(root)
        lay.setContentsMargins(8, 8, 8, 6)
        lay.setSpacing(6)
        self.setCentralWidget(root)

        lay.addWidget(self._make_connect_bar())

        splitter = QSplitter(Qt.Horizontal)
        splitter.setHandleWidth(5)
        splitter.setStyleSheet("QSplitter::handle { background:#2a2a3a; }")
        splitter.addWidget(self._make_left_panel())
        splitter.addWidget(self._make_right_panel())
        splitter.setSizes([520, 400])
        lay.addWidget(splitter, stretch=1)

        lay.addWidget(self._make_log_panel())

        self._sb = QStatusBar()
        self._sb.setStyleSheet("color:#aaaaaa; font-size:11px;")
        self.setStatusBar(self._sb)
        self._status("Ready  —  Robot IP를 입력하고 Connect를 눌러주세요.")

        if not RB_AVAILABLE:
            self._log("⚠️  rbpodo 미설치:  pip install rbpodo")

    # ── Connect bar ───────────────────────────────────────────────────────────

    def _make_connect_bar(self) -> QWidget:
        bar = QWidget()
        bar.setFixedHeight(54)
        bar.setStyleSheet("background:#252535; border-radius:5px;")
        lay = QHBoxLayout(bar)
        lay.setContentsMargins(14, 8, 14, 8)
        lay.setSpacing(10)

        lbl = QLabel("Robot IP:")
        lbl.setFont(QFont("", 11))
        lay.addWidget(lbl)

        self._edit_ip = QLineEdit(IP_DEFAULT)
        self._edit_ip.setFixedWidth(160)
        self._edit_ip.setFont(QFont("Courier New", 11))
        self._edit_ip.setStyleSheet(
            "background:#111122; color:#ffffff; border:1px solid #445566; "
            "border-radius:3px; padding:2px 8px;")
        lay.addWidget(self._edit_ip)

        self._btn_connect    = self._mk_btn("  Connect",    self._C_GREEN, self._on_connect)
        self._btn_disconnect = self._mk_btn("  Disconnect", self._C_RED,   self._on_disconnect)
        self._btn_disconnect.setEnabled(False)
        lay.addWidget(self._btn_connect)
        lay.addWidget(self._btn_disconnect)

        lay.addSpacing(16)

        self._led = QLabel("●")
        self._led.setFont(QFont("", 18))
        self._led.setStyleSheet("color:#444455;")
        self._lbl_conn = QLabel("Disconnected")
        self._lbl_conn.setFont(QFont("", 11))
        self._lbl_conn.setStyleSheet("color:#888888;")
        lay.addWidget(self._led)
        lay.addWidget(self._lbl_conn)
        lay.addStretch()
        return bar

    # ── Left panel: TCP Pose + Robot Status ───────────────────────────────────

    def _make_left_panel(self) -> QWidget:
        w   = QWidget()
        lay = QVBoxLayout(w)
        lay.setSpacing(8)
        lay.setContentsMargins(0, 0, 4, 0)
        lay.addWidget(self._make_tcp_pose_group())
        lay.addWidget(self._make_status_group())
        lay.addStretch()
        return w

    def _make_tcp_pose_group(self) -> QGroupBox:
        grp = QGroupBox("TCP Pose  (Current Position)")
        grp.setStyleSheet(self._grp_style())
        lay = QVBoxLayout(grp)
        lay.setSpacing(8)

        # Unit toggle
        unit_row = QHBoxLayout()
        self._radio_deg = QRadioButton("mm + deg  (native)")
        self._radio_rad = QRadioButton("mm + rad")
        self._radio_deg.setChecked(True)
        self._radio_deg.setStyleSheet("color:#cccccc;")
        self._radio_rad.setStyleSheet("color:#cccccc;")
        bg = QButtonGroup(self)
        bg.addButton(self._radio_deg)
        bg.addButton(self._radio_rad)
        self._radio_deg.toggled.connect(self._on_unit_toggled)
        unit_row.addWidget(self._radio_deg)
        unit_row.addWidget(self._radio_rad)
        unit_row.addStretch()
        lay.addLayout(unit_row)

        # Value grid: X Y Z Rx Ry Rz
        grid = QGridLayout()
        grid.setHorizontalSpacing(10)
        grid.setVerticalSpacing(6)

        self._tcp_val_lbls  = {}   # axis → value QLabel
        self._tcp_unit_lbls = {}   # axis → unit  QLabel

        for row, axis in enumerate(["X", "Y", "Z", "Rx", "Ry", "Rz"]):
            lbl_name = QLabel(f"{axis}:")
            lbl_name.setFont(QFont("", 11, QFont.Bold))
            lbl_name.setFixedWidth(28)
            lbl_name.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
            lbl_name.setStyleSheet("color:#aaaacc;")

            lbl_val = QLabel("—")
            lbl_val.setFont(QFont("Courier New", 14))
            lbl_val.setMinimumWidth(180)
            lbl_val.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
            lbl_val.setStyleSheet("color:#80ffb0;")

            lbl_unit = QLabel("—")
            lbl_unit.setFont(QFont("", 10))
            lbl_unit.setFixedWidth(36)
            lbl_unit.setStyleSheet("color:#777788;")

            grid.addWidget(lbl_name, row, 0)
            grid.addWidget(lbl_val,  row, 1)
            grid.addWidget(lbl_unit, row, 2)

            self._tcp_val_lbls[axis]  = lbl_val
            self._tcp_unit_lbls[axis] = lbl_unit

        lay.addLayout(grid)
        return grp

    def _make_status_group(self) -> QGroupBox:
        grp = QGroupBox("Robot Status")
        grp.setStyleSheet(self._grp_style())
        lay = QFormLayout(grp)
        lay.setSpacing(10)

        self._lbl_robot_state = QLabel("—")
        self._lbl_robot_state.setFont(QFont("", 12, QFont.Bold))
        self._lbl_robot_state.setStyleSheet("color:#666677;")

        self._lbl_hz = QLabel("—")
        self._lbl_hz.setFont(QFont("Courier New", 10))
        self._lbl_hz.setStyleSheet("color:#888899;")

        lay.addRow("State:", self._lbl_robot_state)
        lay.addRow("Poll rate:", self._lbl_hz)
        return grp

    # ── Right panel: Joint Angles ─────────────────────────────────────────────

    def _make_right_panel(self) -> QWidget:
        w   = QWidget()
        lay = QVBoxLayout(w)
        lay.setSpacing(8)
        lay.setContentsMargins(4, 0, 0, 0)
        lay.addWidget(self._make_joint_group())
        lay.addStretch()
        return w

    def _make_joint_group(self) -> QGroupBox:
        grp = QGroupBox("Joint Angles  [deg]")
        grp.setStyleSheet(self._grp_style())
        grid = QGridLayout(grp)
        grid.setHorizontalSpacing(10)
        grid.setVerticalSpacing(8)

        self._joint_lbls = []
        for i in range(6):
            lbl_name = QLabel(f"J{i + 1}:")
            lbl_name.setFont(QFont("", 11, QFont.Bold))
            lbl_name.setFixedWidth(28)
            lbl_name.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
            lbl_name.setStyleSheet("color:#aaaacc;")

            lbl_val = QLabel("—")
            lbl_val.setFont(QFont("Courier New", 14))
            lbl_val.setMinimumWidth(160)
            lbl_val.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
            lbl_val.setStyleSheet("color:#80d0ff;")

            lbl_unit = QLabel("°")
            lbl_unit.setFont(QFont("", 10))
            lbl_unit.setFixedWidth(18)
            lbl_unit.setStyleSheet("color:#777788;")

            grid.addWidget(lbl_name, i, 0)
            grid.addWidget(lbl_val,  i, 1)
            grid.addWidget(lbl_unit, i, 2)
            self._joint_lbls.append(lbl_val)

        return grp

    # ── Log panel ─────────────────────────────────────────────────────────────

    def _make_log_panel(self) -> QGroupBox:
        grp = QGroupBox("Log")
        grp.setStyleSheet(self._grp_style())
        grp.setFixedHeight(170)
        lay = QVBoxLayout(grp)
        lay.setSpacing(4)

        self._txt_log = QTextEdit()
        self._txt_log.setReadOnly(True)
        self._txt_log.setFont(QFont("Courier New", 9))
        self._txt_log.setStyleSheet(
            "QTextEdit { background:#0f0f18; color:#cccccc; "
            "border:1px solid #2a2a3a; border-radius:3px; }")
        lay.addWidget(self._txt_log)

        btn_clear = QPushButton("Clear")
        btn_clear.setFixedSize(72, 24)
        btn_clear.setStyleSheet(
            "QPushButton { background:#2a2a3a; color:#888899; border:none; "
            "border-radius:3px; } QPushButton:hover { background:#3a3a4a; color:#bbbbcc; }")
        btn_clear.clicked.connect(self._txt_log.clear)

        btn_row = QHBoxLayout()
        btn_row.addStretch()
        btn_row.addWidget(btn_clear)
        lay.addLayout(btn_row)
        return grp

    # ─────────────────────────────────────────────────────────────────────────
    #  Slots — connect / disconnect
    # ─────────────────────────────────────────────────────────────────────────

    def _on_connect(self) -> None:
        ip = self._edit_ip.text().strip()
        if not ip:
            self._log("⚠️  IP 주소를 입력해 주세요.")
            return

        if self._data_thread is not None:
            self._data_thread.stop()
            self._data_thread.wait(2000)

        self._log(f"→  {ip}  연결 시도 중 …")
        self._btn_connect.setEnabled(False)
        self._btn_disconnect.setEnabled(False)

        self._data_thread = CobotDataThread(ip)
        self._data_thread.state_updated.connect(self._on_state)
        self._data_thread.conn_changed.connect(self._on_conn_changed)
        self._data_thread.start()

    def _on_disconnect(self) -> None:
        self._btn_disconnect.setEnabled(False)
        if self._data_thread is not None:
            self._data_thread.stop()

    # ─────────────────────────────────────────────────────────────────────────
    #  Slots — data reception
    # ─────────────────────────────────────────────────────────────────────────

    def _on_conn_changed(self, connected: bool, msg: str) -> None:
        if connected:
            self._led.setStyleSheet("color:#44ff88;")
            self._lbl_conn.setText("Connected")
            self._lbl_conn.setStyleSheet("color:#44ff88;")
            self._btn_disconnect.setEnabled(True)
        else:
            self._led.setStyleSheet("color:#444455;")
            self._lbl_conn.setText("Disconnected")
            self._lbl_conn.setStyleSheet("color:#888888;")
            self._btn_connect.setEnabled(True)
            self._btn_disconnect.setEnabled(False)
            self._clear_displays()
        self._log(msg)
        self._status(msg)

    def _on_state(self, state) -> None:
        # ── Poll rate ────────────────────────────────────────────────────────
        now = time.monotonic()
        if self._last_state_t is not None:
            dt = now - self._last_state_t
            if dt > 0:
                self._hz_buf.append(1.0 / dt)
        self._last_state_t = now

        if self._hz_buf:
            self._lbl_hz.setText(f"{sum(self._hz_buf) / len(self._hz_buf):.1f}  Hz")

        # ── TCP pose ─────────────────────────────────────────────────────────
        try:
            tcp = list(state.sdata.tcp_pos)   # [x, y, z, rx, ry, rz] mm + deg
            self._last_tcp = tcp
            self._render_tcp(tcp)
        except AttributeError:
            pass

        # ── Joint angles ─────────────────────────────────────────────────────
        try:
            jnt = state.sdata.jnt_cur   # [j0..j5] deg
            for i, lbl in enumerate(self._joint_lbls):
                lbl.setText(f"{jnt[i]:+9.3f}")
        except AttributeError:
            pass

        # ── Robot state ───────────────────────────────────────────────────────
        try:
            self._render_robot_state(state.sdata.robot_state)
        except AttributeError:
            pass

    def _render_tcp(self, tcp: list) -> None:
        use_rad  = self._radio_rad.isChecked()
        rot_unit = "rad" if use_rad else "deg"

        axes     = ["X",   "Y",   "Z",   "Rx",      "Ry",      "Rz"]
        units    = ["mm",  "mm",  "mm",  rot_unit,  rot_unit,  rot_unit]

        for i, (axis, unit) in enumerate(zip(axes, units)):
            v = tcp[i]
            if use_rad and i >= 3:
                v = np.radians(v)

            fmt = f"{v:+10.3f}" if i < 3 else (f"{v:+10.6f}" if use_rad else f"{v:+10.4f}")
            self._tcp_val_lbls[axis].setText(fmt)
            self._tcp_unit_lbls[axis].setText(unit)

    def _render_robot_state(self, rs) -> None:
        if not RB_AVAILABLE:
            return
        try:
            label, color = {
                rb.RobotState.Idle:    ("● IDLE",    "#44ff88"),
                rb.RobotState.Moving:  ("● MOVING",  "#ffcc44"),
                rb.RobotState.Unknown: ("● UNKNOWN", "#888888"),
            }.get(rs, ("● UNKNOWN", "#888888"))
        except AttributeError:
            label, color = "● —", "#888888"

        self._lbl_robot_state.setText(label)
        self._lbl_robot_state.setStyleSheet(f"color:{color};")

    def _on_unit_toggled(self) -> None:
        if self._last_tcp is not None:
            self._render_tcp(self._last_tcp)

    def _clear_displays(self) -> None:
        for lbl in self._tcp_val_lbls.values():
            lbl.setText("—")
        for lbl in self._tcp_unit_lbls.values():
            lbl.setText("—")
        for lbl in self._joint_lbls:
            lbl.setText("—")
        self._lbl_robot_state.setText("—")
        self._lbl_robot_state.setStyleSheet("color:#666677;")
        self._lbl_hz.setText("—")
        self._last_tcp    = None
        self._last_state_t = None
        self._hz_buf.clear()

    # ─────────────────────────────────────────────────────────────────────────
    #  Utilities
    # ─────────────────────────────────────────────────────────────────────────

    def _log(self, msg: str) -> None:
        ts = datetime.datetime.now().strftime("%H:%M:%S.%f")[:-3]
        self._txt_log.append(f"[{ts}]  {msg}")

    def _status(self, msg: str) -> None:
        self._sb.showMessage(msg)

    def closeEvent(self, e) -> None:
        if self._data_thread is not None:
            self._data_thread.stop()
            self._data_thread.wait(2000)
        super().closeEvent(e)

    @staticmethod
    def _grp_style() -> str:
        return (
            "QGroupBox {"
            "  border: 1px solid #353545;"
            "  border-radius: 5px;"
            "  margin-top: 16px;"
            "  padding-top: 6px;"
            "  font-weight: bold;"
            "  color: #aaaacc;"
            "}"
            "QGroupBox::title {"
            "  subcontrol-origin: margin;"
            "  left: 10px; top: 2px;"
            "}"
        )

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
                padding: 0 18px;
                font-weight: bold;
                font-size: 12px;
            }}
            QPushButton:hover    {{ border: 1px solid #ffffff44; }}
            QPushButton:disabled {{ background-color: rgb(55,55,65); color: #666677; }}
        """)
        btn.clicked.connect(cb)
        return btn


# ─────────────────────────────────────────────────────────────────────────────
#  Entry point
# ─────────────────────────────────────────────────────────────────────────────

def _dark_palette() -> QPalette:
    p = QPalette()
    p.setColor(QPalette.Window,          QColor(22, 22, 32))
    p.setColor(QPalette.WindowText,      QColor(204, 204, 204))
    p.setColor(QPalette.Base,            QColor(15, 15, 24))
    p.setColor(QPalette.AlternateBase,   QColor(35, 35, 48))
    p.setColor(QPalette.Text,            QColor(204, 204, 204))
    p.setColor(QPalette.Button,          QColor(50, 50, 70))
    p.setColor(QPalette.ButtonText,      QColor(204, 204, 204))
    p.setColor(QPalette.Highlight,       QColor(65, 105, 180))
    p.setColor(QPalette.HighlightedText, QColor(255, 255, 255))
    p.setColor(QPalette.ToolTipBase,     QColor(30, 30, 45))
    p.setColor(QPalette.ToolTipText,     QColor(200, 200, 200))
    return p


def main() -> None:
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    app.setPalette(_dark_palette())
    win = RainbowCommApp()
    win.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
