"""
Local theme overlay for the Line Tracing GUI.

zividsamples.gui.qt_application.ZividQtApplication applies its own dark QSS in
its constructor. That sheet is shared by every zividsamples GUI and lives in an
*installed* package (site-packages), so it cannot be edited from this repo -
and should not be, since the other samples depend on its look.

Instead, apply_theme() appends the QSS below to the one already installed on the
QApplication. Later rules of equal specificity win in QSS, so each rule here
overrides the corresponding zividsamples rule without touching that package.

What it changes, and why (the base sheet is otherwise unmodified):

- Font family. The base sheet asks for "Helvetica", which Windows does not ship,
  so Qt silently substitutes and the result reads like an old Qt app. Segoe UI is
  the native Windows UI font, with fallbacks for Linux.
- QGroupBox. The base draws a 2px solid border around every group. This app has
  seven of them, which turns the window into a stack of heavy boxes. Zivid Studio
  instead separates sections with a header plus a thin rule, which is what the
  border-top + title-in-the-margin rule below produces.
- QPushButton. The base sheet defines no :hover state at all, so every button
  looks inert, and its :checked state paints black text on a dark gray fill,
  which is barely readable. Primary actions (capture/generate/execute) opt into
  the Zivid accent color via the "accent" property, destructive ones (stop) via
  "danger" - see ACCENT_PROPERTY / DANGER_PROPERTY.
- QDoubleSpinBox / QScrollBar / QRadioButton / QToolTip / QSplitter. The base
  sheet has no rules for these, so they fall back to the platform style and
  clash with the dark palette. This app leans on spin boxes especially (eleven of
  them), which the other zividsamples GUIs barely use. Their built-in up/down
  buttons cannot be styled at all - widgets.spin_box_stepper.SpinBoxStepper
  replaces them with real buttons, styled here via STEPPER_PROPERTY.

Colors follow Zivid's brand cyan (#03B9EB, as used in Zivid Studio) for accents;
the grays are a slightly darker take on the base sheet's palette.

"""

from PyQt5.QtWidgets import QApplication, QPushButton

# Set as a Qt dynamic property to opt a button into the accent/danger styling,
# e.g. button.setProperty(ACCENT_PROPERTY, True). Assign before the button is
# first shown so the initial polish picks it up.
ACCENT_PROPERTY = "accent"
DANGER_PROPERTY = "danger"
STEPPER_PROPERTY = "stepper"

SURFACE = "#2b2b2b"
SURFACE_RAISED = "#383838"
SURFACE_SUNKEN = "#2f2f2f"
SURFACE_PRESSED = "#1f1f1f"
SURFACE_HOVER = "#444444"
DIVIDER = "#4a4a4a"
BORDER_HOVER = "#5c5c5c"
BORDER_SUBTLE = "#3a3a3a"

TEXT = "#e8e8e8"
TEXT_MUTED = "#9a9a9a"
TEXT_ON_ACCENT = "#06232c"

ACCENT = "#03b9eb"
ACCENT_HOVER = "#35c9f0"
ACCENT_PRESSED = "#0294bc"

DANGER = "#a32d2d"
DANGER_HOVER = "#b83a3a"
DANGER_PRESSED = "#7d2020"

# Inline styles for transient button/label states. Widgets set these directly
# (setStyleSheet on the widget itself) because they change at runtime, which a
# static sheet cannot express; keeping the values here stops raw "green"/"yellow"
# from leaking into the widget code.
STATUS_OK_STYLE = "background-color: #3b6d11; color: white;"
STATUS_WARNING_STYLE = "background-color: #854f0b; color: white;"
STATUS_DANGER_STYLE = f"background-color: {DANGER}; color: white;"
BUSY_BUTTON_STYLE = "background-color: #8a6100; color: white;"

FONT_FAMILY = '"Segoe UI", "Malgun Gothic", "Ubuntu", "DejaVu Sans", sans-serif'

LINE_TRACING_STYLE = f"""
QWidget {{
    background-color: {SURFACE};
    color: {TEXT};
    font-family: {FONT_FAMILY};
    font-size: 10pt;
}}

/* Section = header text, a thin rule, then the content. No surrounding box.
   font-* belongs on ::title only: a font set on QGroupBox itself is inherited
   by everything inside it. */
QGroupBox {{
    border: none;
    border-top: 1px solid {DIVIDER};
    border-radius: 0px;
    margin-top: 24px;
    padding-top: 12px;
}}
QGroupBox::title {{
    subcontrol-origin: margin;
    subcontrol-position: top left;
    left: 0px;
    padding: 0px 0px 7px 0px;
    color: {TEXT};
    font-size: 11pt;
    font-weight: 600;
}}

QPushButton {{
    background-color: {SURFACE_RAISED};
    color: {TEXT};
    border: 1px solid {DIVIDER};
    border-radius: 3px;
    padding: 6px 14px;
}}
QPushButton:hover {{
    background-color: {SURFACE_HOVER};
    border-color: {BORDER_HOVER};
}}
QPushButton:pressed {{
    background-color: {SURFACE_PRESSED};
}}
QPushButton:checked {{
    background-color: {ACCENT_PRESSED};
    border-color: {ACCENT};
    color: white;
}}
QPushButton:disabled {{
    background-color: {SURFACE_SUNKEN};
    border-color: {BORDER_SUBTLE};
    color: {TEXT_MUTED};
}}

QPushButton[{ACCENT_PROPERTY}="true"] {{
    background-color: {ACCENT};
    border-color: {ACCENT};
    color: {TEXT_ON_ACCENT};
    font-weight: 600;
}}
QPushButton[{ACCENT_PROPERTY}="true"]:hover {{
    background-color: {ACCENT_HOVER};
    border-color: {ACCENT_HOVER};
}}
QPushButton[{ACCENT_PROPERTY}="true"]:pressed {{
    background-color: {ACCENT_PRESSED};
    border-color: {ACCENT_PRESSED};
}}
QPushButton[{ACCENT_PROPERTY}="true"]:disabled {{
    background-color: {SURFACE_SUNKEN};
    border-color: {BORDER_SUBTLE};
    color: {TEXT_MUTED};
    font-weight: normal;
}}

QPushButton[{DANGER_PROPERTY}="true"] {{
    background-color: {DANGER};
    border-color: {DANGER};
    color: white;
    font-weight: 600;
}}
QPushButton[{DANGER_PROPERTY}="true"]:hover {{
    background-color: {DANGER_HOVER};
    border-color: {DANGER_HOVER};
}}
QPushButton[{DANGER_PROPERTY}="true"]:pressed {{
    background-color: {DANGER_PRESSED};
    border-color: {DANGER_PRESSED};
}}
QPushButton[{DANGER_PROPERTY}="true"]:disabled {{
    background-color: {SURFACE_SUNKEN};
    border-color: {BORDER_SUBTLE};
    color: {TEXT_MUTED};
    font-weight: normal;
}}

QPushButton[{STEPPER_PROPERTY}="true"] {{
    padding: 0px;
    font-size: 12pt;
    color: {TEXT_MUTED};
}}
QPushButton[{STEPPER_PROPERTY}="true"]:hover {{
    color: {ACCENT};
    border-color: {ACCENT};
}}

QLineEdit {{
    background-color: {SURFACE_RAISED};
    color: {TEXT};
    border: 1px solid {DIVIDER};
    border-radius: 3px;
    padding: 4px 7px;
    selection-background-color: {ACCENT};
    selection-color: {TEXT_ON_ACCENT};
}}
QLineEdit:focus {{
    border-color: {ACCENT};
}}
QLineEdit:read-only {{
    background-color: {SURFACE_SUNKEN};
    border: 1px solid {BORDER_SUBTLE};
    color: {TEXT_MUTED};
}}

QDoubleSpinBox, QSpinBox {{
    background-color: {SURFACE_RAISED};
    color: {TEXT};
    border: 1px solid {DIVIDER};
    border-radius: 3px;
    padding: 4px 6px;
    selection-background-color: {ACCENT};
    selection-color: {TEXT_ON_ACCENT};
}}
QDoubleSpinBox:focus, QSpinBox:focus {{
    border-color: {ACCENT};
}}
QDoubleSpinBox:disabled, QSpinBox:disabled {{
    background-color: {SURFACE_SUNKEN};
    border-color: {BORDER_SUBTLE};
    color: {TEXT_MUTED};
}}

QRadioButton {{
    spacing: 7px;
}}
QRadioButton::indicator {{
    width: 14px;
    height: 14px;
    border-radius: 8px;
    border: 1px solid {BORDER_HOVER};
    background-color: {SURFACE_RAISED};
}}
QRadioButton::indicator:hover {{
    border-color: {ACCENT_HOVER};
}}
QRadioButton::indicator:checked {{
    background-color: {ACCENT};
    border-color: {ACCENT};
}}

QScrollBar:vertical {{
    background: transparent;
    width: 11px;
    margin: 0px;
}}
QScrollBar::handle:vertical {{
    background: {SURFACE_HOVER};
    border-radius: 5px;
    min-height: 28px;
}}
QScrollBar::handle:vertical:hover {{
    background: {BORDER_HOVER};
}}
QScrollBar:horizontal {{
    background: transparent;
    height: 11px;
    margin: 0px;
}}
QScrollBar::handle:horizontal {{
    background: {SURFACE_HOVER};
    border-radius: 5px;
    min-width: 28px;
}}
QScrollBar::handle:horizontal:hover {{
    background: {BORDER_HOVER};
}}
QScrollBar::add-line, QScrollBar::sub-line {{
    width: 0px;
    height: 0px;
}}
QScrollBar::add-page, QScrollBar::sub-page {{
    background: transparent;
}}

/* Flat tabs with an accent underline on the active one, replacing the base
   sheet's bordered-box tabs. */
QTabWidget::pane, QTabWidget#main_tab_widget::pane {{
    border: none;
    border-top: 1px solid {DIVIDER};
}}
QTabWidget::tab-bar {{
    left: 0px;
}}
QTabBar::tab {{
    background-color: transparent;
    border: none;
    border-bottom: 2px solid transparent;
    border-radius: 0px;
    padding: 8px 18px;
    margin-right: 4px;
    margin-top: 0px;
    color: {TEXT_MUTED};
}}
QTabBar::tab:selected {{
    background-color: transparent;
    border: none;
    border-bottom: 2px solid {ACCENT};
    color: {TEXT};
}}
QTabBar::tab:!selected {{
    border: none;
    border-bottom: 2px solid transparent;
    margin-top: 0px;
}}
QTabBar::tab:hover:!selected {{
    background-color: {SURFACE_RAISED};
    color: {TEXT};
}}

QSplitter::handle {{
    background-color: transparent;
}}
QSplitter::handle:horizontal {{
    width: 7px;
}}
QSplitter::handle:horizontal:hover {{
    background-color: {DIVIDER};
}}

QComboBox {{
    background-color: {SURFACE_RAISED};
    color: {TEXT};
    border: 1px solid {DIVIDER};
    border-radius: 3px;
    padding: 4px 7px;
}}
QComboBox:hover {{
    border-color: {BORDER_HOVER};
}}
QComboBox:focus, QComboBox:on {{
    border-color: {ACCENT};
}}
/* ::drop-down is deliberately left unstyled: overriding it drops Qt's built-in
   arrow, and QSS can only supply a replacement as an image file. */
QComboBox QAbstractItemView {{
    background-color: {SURFACE_RAISED};
    color: {TEXT};
    border: 1px solid {DIVIDER};
    selection-background-color: {ACCENT};
    selection-color: {TEXT_ON_ACCENT};
    outline: none;
}}

QFrame[isHorizontalLine="true"] {{
    border: none;
    border-top: 1px solid {DIVIDER};
    border-radius: 0px;
}}
QFrame[isVerticalLine="true"] {{
    border: none;
    border-left: 1px solid {DIVIDER};
    border-radius: 0px;
}}

QToolTip {{
    background-color: {SURFACE_PRESSED};
    color: {TEXT};
    border: 1px solid {DIVIDER};
    padding: 6px 8px;
}}
"""


def apply_theme(app: QApplication) -> None:
    """Append this app's QSS to whatever ZividQtApplication already installed."""
    app.setStyleSheet(app.styleSheet() + LINE_TRACING_STYLE)


def mark_as_accent(button: QPushButton) -> None:
    """Style `button` as the primary action of its section."""
    button.setProperty(ACCENT_PROPERTY, True)


def mark_as_danger(button: QPushButton) -> None:
    """Style `button` as a destructive/abort action."""
    button.setProperty(DANGER_PROPERTY, True)
