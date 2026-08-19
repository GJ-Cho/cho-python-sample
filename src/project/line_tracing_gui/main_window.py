"""
Main window for the Line Tracing GUI.

Tab layout follows the same QTabWidget(objectName="main_tab_widget") convention
as zivid-python-samples' Hand-Eye GUI so the existing dark theme QSS applies
automatically (see zividsamples.gui.qt_application).

"""

import zivid
from PyQt5.QtWidgets import QMainWindow, QTabWidget, QVBoxLayout, QWidget

from line_tracing_gui.widgets.calibration_panel import CalibrationPanel
from line_tracing_gui.widgets.camera_panel import CameraPanel
from line_tracing_gui.widgets.robot_connection_widget import RobotConnectionWidget
from line_tracing_gui.widgets.trace_panel import TracePanel


class LineTracingMainWindow(QMainWindow):
    def __init__(self, zivid_app: zivid.Application, parent=None):
        super().__init__(parent)
        self.zivid_app = zivid_app

        self.main_tab_widget = QTabWidget()
        self.main_tab_widget.setObjectName("main_tab_widget")

        self.connect_tab = self._build_connect_tab()
        self.main_tab_widget.addTab(self.connect_tab, "Connect")

        self.calibration_panel = CalibrationPanel()
        self.main_tab_widget.addTab(self.calibration_panel, "Calibration")

        self.trace_panel = TracePanel(self.camera_panel, self.calibration_panel, self.robot_connection_widget)
        self.main_tab_widget.addTab(self.trace_panel, "Line Tracing")

        self.setCentralWidget(self.main_tab_widget)
        self.resize(1500, 840)  # ~1.5x wider, ~1.2x taller than the previous 1000x700

    def _build_connect_tab(self) -> QWidget:
        tab = QWidget()
        layout = QVBoxLayout()

        self.camera_panel = CameraPanel(self.zivid_app)
        layout.addWidget(self.camera_panel)

        self.robot_connection_widget = RobotConnectionWidget()
        layout.addWidget(self.robot_connection_widget)

        tab.setLayout(layout)
        return tab
