"""
ImageViewer subclass that lets the user draw a freehand line on top of the
displayed 2D image.

zividsamples.gui.widgets.image_viewer.ImageViewer has no click/drag overlay
support (drag is consumed by ScrollHandDrag for panning). This widget adds a
"draw mode" that switches panning off, tracks mouse strokes in scene
coordinates (== original image pixel coordinates, since set_pixmap loads the
pixmap at native resolution), and renders them as an overlay path.

"""

from typing import List, Optional, Tuple

from PyQt5.QtCore import QPointF, Qt, pyqtSlot
from PyQt5.QtGui import QColor, QMouseEvent, QPainterPath, QPen, QPixmap
from PyQt5.QtWidgets import QGraphicsPathItem, QGraphicsView
from zividsamples.gui.widgets.image_viewer import ImageViewer

Stroke = List[Tuple[float, float]]

LINE_COLOR = QColor(237, 52, 114)  # ZividColors.PINK, kept in sync visually with the app's accent color
LINE_WIDTH = 3.0
MIN_POINT_DISTANCE = 3.0  # pixels (scene units), avoids one point per mouse-move event


class DrawableImageViewer(ImageViewer):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._draw_mode = False
        self._strokes: List[Stroke] = []
        self._active_stroke: Optional[Stroke] = None
        self._path_item: Optional[QGraphicsPathItem] = None
        self._pen = QPen(LINE_COLOR, LINE_WIDTH)
        self._pen.setCosmetic(True)

    def set_draw_mode(self, enabled: bool) -> None:
        self._draw_mode = enabled
        self.setDragMode(QGraphicsView.NoDrag if enabled else QGraphicsView.ScrollHandDrag)

    def is_draw_mode(self) -> bool:
        return self._draw_mode

    def get_line_points(self) -> Stroke:
        points: Stroke = []
        for stroke in self._strokes:
            points.extend(stroke)
        return points

    def has_line(self) -> bool:
        return len(self._strokes) > 0

    def clear_line(self) -> None:
        self._strokes = []
        self._active_stroke = None
        self._redraw_overlay()

    def undo_last_stroke(self) -> None:
        if self._strokes:
            self._strokes.pop()
            self._redraw_overlay()

    @pyqtSlot(QPixmap, bool)
    def set_pixmap(self, image: QPixmap, reset_zoom: bool = False) -> None:
        # The base class clears the whole QGraphicsScene on every call, which would
        # otherwise leave self._path_item pointing at a deleted item.
        super().set_pixmap(image, reset_zoom)
        self._strokes = []
        self._active_stroke = None
        self._path_item = None

    def mousePressEvent(self, event: QMouseEvent) -> None:
        if self._draw_mode and event.button() == Qt.LeftButton and self.scene() is not None:
            scene_pos = self.mapToScene(event.pos())
            self._active_stroke = [(scene_pos.x(), scene_pos.y())]
            event.accept()
            return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event: QMouseEvent) -> None:
        if self._draw_mode and self._active_stroke is not None:
            scene_pos = self.mapToScene(event.pos())
            last_x, last_y = self._active_stroke[-1]
            if (scene_pos.x() - last_x) ** 2 + (scene_pos.y() - last_y) ** 2 >= MIN_POINT_DISTANCE**2:
                self._active_stroke.append((scene_pos.x(), scene_pos.y()))
                self._redraw_overlay()
            event.accept()
            return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:
        if self._draw_mode and self._active_stroke is not None:
            if len(self._active_stroke) > 1:
                self._strokes.append(self._active_stroke)
            self._active_stroke = None
            self._redraw_overlay()
            event.accept()
            return
        super().mouseReleaseEvent(event)

    def _redraw_overlay(self) -> None:
        if self.scene() is None:
            return
        if self._path_item is not None:
            self.scene().removeItem(self._path_item)
            self._path_item = None
        strokes = list(self._strokes)
        if self._active_stroke:
            strokes.append(self._active_stroke)
        if not strokes:
            return
        path = QPainterPath()
        for stroke in strokes:
            if not stroke:
                continue
            path.moveTo(QPointF(*stroke[0]))
            for x, y in stroke[1:]:
                path.lineTo(QPointF(x, y))
        self._path_item = self.scene().addPath(path, self._pen)
        self._path_item.setZValue(10)
