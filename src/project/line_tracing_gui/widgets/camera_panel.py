"""
Camera connection + live 2D preview panel for the Line Tracing GUI.

Reuses zividsamples.gui.buttons_widget.CameraButtonsWidget and
zividsamples.gui.live_2d_widget.Live2DWidget as-is; this module only wires
them together and handles Zivid camera connect/disconnect.

"""

from typing import Optional

import zivid
from PyQt5.QtCore import pyqtSignal
from PyQt5.QtWidgets import QGroupBox, QVBoxLayout, QWidget
from zividsamples.gui.widgets.camera_buttons_widget import CameraButtonsWidget
from zividsamples.gui.widgets.live_2d_widget import Live2DWidget
from zividsamples.gui.wizard.camera_selection import select_camera


class CameraPanel(QWidget):
    camera_connected = pyqtSignal(bool)

    def __init__(self, zivid_app: zivid.Application, parent=None):
        super().__init__(parent)
        self.zivid_app = zivid_app
        self.camera: Optional[zivid.Camera] = None

        self.buttons = CameraButtonsWidget(capture_button_text="Live 미리보기 시작")
        self.live_2d_widget = Live2DWidget()
        self.live_2d_widget.setMinimumHeight(300)
        self.live_2d_widget.camera_disconnected.connect(self.on_camera_disconnected)

        group_box = QGroupBox("카메라 (Zivid2+ MR130)")
        group_layout = QVBoxLayout()
        group_layout.addWidget(self.buttons)
        group_layout.addWidget(self.live_2d_widget)
        group_box.setLayout(group_layout)

        layout = QVBoxLayout()
        layout.addWidget(group_box)
        self.setLayout(layout)

        self.buttons.connect_button_clicked.connect(self.on_connect_button_clicked)
        self.buttons.capture_button_clicked.connect(self.on_toggle_live_clicked)

    def default_settings_2d(self) -> zivid.Settings2D:
        settings_2d = zivid.Settings2D()
        acquisition = zivid.Settings2D.Acquisition()
        # A freshly constructed Acquisition leaves brightness/exposure_time/etc as None
        # ("use camera default"). zividsamples.Live2DWidget.update_exposure_based_on_relative_brightness
        # only skips its projector-brightness-compensation branch when brightness == 0.0
        # exactly - None == 0.0 is False, so it falls through and crashes doing arithmetic
        # on a None exposure_time. We are not using the projector, so brightness must be 0.0.
        acquisition.brightness = 0.0
        settings_2d.acquisitions.append(acquisition)
        return settings_2d

    def start_live_preview(self) -> None:
        if self.camera is None:
            return
        settings_2d = self.default_settings_2d()
        self.live_2d_widget.camera = self.camera
        self.live_2d_widget.capture_function = self.camera.capture_2d
        if self.camera.info.model:
            self.live_2d_widget.update_settings_2d(settings_2d, self.camera.info.model)
        else:
            self.live_2d_widget.settings_2d = settings_2d
        self.live_2d_widget.start_live_2d()

    def stop_live_preview(self) -> None:
        if self.live_2d_widget.live_thread is not None and self.live_2d_widget.is_active():
            self.live_2d_widget.stop_live_2d()

    def update_capture_button_state(self) -> None:
        is_live = self.live_2d_widget.is_active()
        self.buttons.capture_button.setChecked(is_live)
        self.buttons.capture_button.setStyleSheet("background-color: green;" if is_live else "")
        self.buttons.capture_button.setText("Live 미리보기 중지" if is_live else "Live 미리보기 시작")

    def on_toggle_live_clicked(self) -> None:
        if self.live_2d_widget.is_active():
            self.stop_live_preview()
        else:
            self.start_live_preview()
        self.update_capture_button_state()

    def on_connect_button_clicked(self) -> None:
        if self.camera is not None and self.camera.state.connected:
            self.stop_live_preview()
            self.camera.disconnect()
            self.camera = None
        else:
            self.camera = select_camera(self.zivid_app, connect=True)
        self.buttons.set_connection_status(self.camera)
        connected = self.camera is not None and self.camera.state.connected
        self.camera_connected.emit(connected)
        if connected:
            self.start_live_preview()
        self.update_capture_button_state()

    def on_camera_disconnected(self, error_message: str) -> None:
        self.buttons.set_connection_status(self.camera)
        self.camera_connected.emit(False)
        self.update_capture_button_state()
