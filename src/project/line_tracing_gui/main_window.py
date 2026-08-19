"""
Main window for the Line Tracing GUI.

Tab layout follows the same QTabWidget(objectName="main_tab_widget") convention
as zivid-python-samples' Hand-Eye GUI so the existing dark theme QSS applies
automatically (see zividsamples.gui.qt_application).

"""

import zivid
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QMainWindow, QSplitter, QTabWidget, QVBoxLayout, QWidget

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
        """Camera on the left (connect + live 2D), robot on the right (connect, status,
        TCP offset), both visible at once - the same split the Line Tracing tab uses."""
        self.camera_panel = CameraPanel(self.zivid_app)
        self.robot_connection_widget = RobotConnectionWidget()

        splitter = QSplitter(Qt.Horizontal)
        splitter.addWidget(self.camera_panel)
        splitter.addWidget(self.robot_connection_widget)
        # The live 2D preview needs the width more than the robot's forms do.
        splitter.setStretchFactor(0, 3)
        splitter.setStretchFactor(1, 2)

        tab = QWidget()
        layout = QVBoxLayout()
        layout.addWidget(splitter)
        tab.setLayout(layout)
        return tab
