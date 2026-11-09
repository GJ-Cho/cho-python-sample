"""
Line Tracing GUI entry point.

Mirrors the bootstrap pattern of zivid-python-samples' hand_eye_gui.py:
    with ZividQtApplication() as qt_app:
        sys.exit(qt_app.run(MainWindow(qt_app.zivid_app), "<title>"))

Runnable directly (no `python -m` needed):
    python main.py                      # from inside line_tracing_gui/
    python line_tracing_gui/main.py     # from its parent folder

"""

import sys
from pathlib import Path

# main_window.py etc. are imported below via the absolute `line_tracing_gui.*` path
# (so it also still works when run with `python -m line_tracing_gui.main`). That only
# resolves if line_tracing_gui's *parent* folder is on sys.path - guarantee that here
# instead of relying on the current working directory when running this file directly.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from zividsamples.gui.qt_application import ZividQtApplication

from line_tracing_gui.main_window import LineTracingMainWindow
from line_tracing_gui.theme import apply_theme


def _main() -> None:
    with ZividQtApplication() as qt_app:
        # Must run before any widget is built, so the first polish already sees the
        # accent/danger properties the widgets set on their buttons (see theme.py).
        apply_theme(qt_app)
        sys.exit(qt_app.run(LineTracingMainWindow(qt_app.zivid_app), "Line Tracing GUI"))


if __name__ == "__main__":
    _main()
