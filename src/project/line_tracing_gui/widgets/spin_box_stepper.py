"""
A spin box paired with its own stepper buttons.

Qt draws QAbstractSpinBox's up/down buttons inside the frame using the platform
style, and they are effectively not stylable: giving ::up-button or ::down-button
any rule stops Qt drawing its built-in arrow and makes it expect an image file
instead, and the CSS border-triangle trick renders as a plain square. On Windows
that leaves a cramped raised sub-button that reads as a rendering glitch against
this app's dark palette (switching to +/- symbols only changed the glyph, not the
button).

So the spin box gets no buttons of its own, and two ordinary QPushButtons next to
it do the stepping. Those follow theme.py's stylesheet like every other button -
hover, pressed and focus states included.

The spin box is passed in rather than created here so callers keep the reference
they already use for value()/setValue(); this only owns the layout around it:

    self.speed_spinbox = QDoubleSpinBox()
    ...
    form.addRow("Speed", SpinBoxStepper(self.speed_spinbox))

Disable the SpinBoxStepper, not the spin box inside it - Qt propagates that to the
buttons, whereas disabling the spin box alone would leave them live.

"""

from typing import Optional

from PyQt5.QtWidgets import QAbstractSpinBox, QHBoxLayout, QPushButton, QWidget

from line_tracing_gui.theme import STEPPER_PROPERTY

BUTTON_SIZE_PX = 26
DEFAULT_FIELD_WIDTH_PX = 92


class SpinBoxStepper(QWidget):
    def __init__(
        self,
        spin_box: QAbstractSpinBox,
        field_width_px: int = DEFAULT_FIELD_WIDTH_PX,
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self.spin_box = spin_box
        self.spin_box.setButtonSymbols(QAbstractSpinBox.NoButtons)
        self.spin_box.setFixedWidth(field_width_px)

        self.step_down_button = QPushButton("−")  # minus sign, not a hyphen
        self.step_up_button = QPushButton("+")
        for button, steps in ((self.step_down_button, -1), (self.step_up_button, 1)):
            button.setProperty(STEPPER_PROPERTY, True)
            button.setFixedSize(BUTTON_SIZE_PX, BUTTON_SIZE_PX)
            button.setAutoRepeat(True)  # press and hold to keep stepping
            button.setFocusPolicy(self.spin_box.focusPolicy())
            button.clicked.connect(lambda _checked=False, s=steps: self.spin_box.stepBy(s))

        layout = QHBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)
        layout.addWidget(self.spin_box)
        layout.addWidget(self.step_down_button)
        layout.addWidget(self.step_up_button)
        self.setLayout(layout)
