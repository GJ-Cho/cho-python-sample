"""
Standalone manual test for DrawableImageViewer (no camera/robot needed).

Run:
    python -m line_tracing_gui._dev_test_drawable_viewer

Loads sample/sample_MR130_2d.png, lets you draw with the mouse (draw mode is
on by default), and prints the collected line points on demand.

"""

import sys
from pathlib import Path

from PyQt5.QtGui import QPixmap
from PyQt5.QtWidgets import QApplication, QHBoxLayout, QMainWindow, QPushButton, QVBoxLayout, QWidget

from line_tracing_gui.widgets.drawable_image_viewer import DrawableImageViewer

SAMPLE_IMAGE_PATH = Path(__file__).resolve().parent.parent.parent.parent / "sample" / "sample_MR130_2d.png"


class DevTestWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("DrawableImageViewer manual test")

        self.viewer = DrawableImageViewer()
        self.viewer.set_draw_mode(True)
        self.viewer.setMinimumSize(800, 600)

        pixmap = QPixmap(str(SAMPLE_IMAGE_PATH))
        if pixmap.isNull():
            raise RuntimeError(f"Failed to load {SAMPLE_IMAGE_PATH}")
        self.viewer.set_pixmap(pixmap, reset_zoom=True)

        toggle_button = QPushButton("Draw mode: ON (click to toggle pan mode)")
        toggle_button.setCheckable(True)
        toggle_button.setChecked(True)
        toggle_button.clicked.connect(lambda checked: self._on_toggle(toggle_button, checked))

        clear_button = QPushButton("Clear line")
        clear_button.clicked.connect(self.viewer.clear_line)

        undo_button = QPushButton("Undo last stroke")
        undo_button.clicked.connect(self.viewer.undo_last_stroke)

        print_button = QPushButton("Print line points")
        print_button.clicked.connect(self._print_points)

        buttons_layout = QHBoxLayout()
        buttons_layout.addWidget(toggle_button)
        buttons_layout.addWidget(clear_button)
        buttons_layout.addWidget(undo_button)
        buttons_layout.addWidget(print_button)

        layout = QVBoxLayout()
        layout.addLayout(buttons_layout)
        layout.addWidget(self.viewer)

        central = QWidget()
        central.setLayout(layout)
        self.setCentralWidget(central)
        self.resize(900, 700)

    def _on_toggle(self, button: QPushButton, checked: bool) -> None:
        self.viewer.set_draw_mode(checked)
        button.setText(f"Draw mode: {'ON' if checked else 'OFF (pan/zoom)'} (click to toggle)")

    def _print_points(self) -> None:
        points = self.viewer.get_line_points()
        print(f"{len(points)} points:")
        print(points)


def _main() -> None:
    app = QApplication(sys.argv)
    window = DevTestWindow()
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    _main()
