"""
Embeddable 3D viewer: captured point cloud with the computed waypoint path
overlaid, so the user can sanity-check the path before running it on the
robot. Meant to sit side-by-side with the 2D DrawableImageViewer in the Trace
tab (not a popup) - see TracePanel.

Everything is drawn in camera frame (waypoints are transformed back from
base frame for display by the caller) since the point cloud itself is
captured in camera frame.

"""

from typing import List, Optional, Sequence, Tuple

import numpy as np
import pyqtgraph as pg
import pyqtgraph.opengl as gl
from PyQt5.QtWidgets import QSizePolicy, QVBoxLayout, QWidget
from zividsamples.transformation_matrix import TransformationMatrix

MAX_DISPLAY_POINTS = 80_000
PATH_COLOR = (0.93, 0.20, 0.45, 1.0)  # matches DrawableImageViewer's drawn-line color
POINT_SIZE = 2.0
WAYPOINT_MARKER_SIZE = 8.0
FALLBACK_POINT_COLOR = (0.6, 0.6, 0.6, 1.0)
DEFAULT_CAMERA_DISTANCE_MM = 500.0

# Waypoint orientation axes, drawn as short line segments from each waypoint's origin.
# Colors follow the common X=red/Y=green/Z=blue convention.
AXIS_LENGTH_MM = 15.0
AXIS_COLOR_X = (1.0, 0.25, 0.25, 1.0)
AXIS_COLOR_Y = (0.25, 1.0, 0.25, 1.0)
AXIS_COLOR_Z = (0.35, 0.55, 1.0, 1.0)

CURRENT_POSITION_COLOR = (1.0, 0.85, 0.1, 1.0)  # bright yellow, distinct from the path/axes
CURRENT_POSITION_SIZE = 14.0

# Reachability verdicts (see robot.reachability). Drawn as oversized markers on top of the
# normal path markers so a handful of bad points among hundreds is still findable by eye -
# the whole purpose of highlighting them is to answer "where do I redraw the line?".
UNREACHABLE_COLOR = (1.0, 0.15, 0.15, 1.0)  # red: the robot cannot go here
UNREACHABLE_MARKER_SIZE = 15.0
WARNING_COLOR = (1.0, 0.65, 0.0, 1.0)  # amber: reachable, but a suspected configuration flip
WARNING_MARKER_SIZE = 13.0


class PointCloudViewerWidget(QWidget):
    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)

        # rotationMethod='quaternion' avoids GLViewWidget's default euler-angle camera,
        # which clamps elevation to +-90 deg (rotating feels "stuck" once near-vertical -
        # that clamp exists to dodge gimbal-lock flips, but it's disorienting here).
        self.view = gl.GLViewWidget(rotationMethod="quaternion")
        self.view.setBackgroundColor((30, 30, 30))
        # GLViewWidget defaults to a Preferred size policy (unlike QGraphicsView, which
        # DrawableImageViewer uses and which defaults to Expanding) - without this it
        # doesn't claim leftover space in a layout and ends up squeezed into a corner.
        self.view.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self._scatter_item: Optional[gl.GLScatterPlotItem] = None
        self._path_items: List[gl.GLLinePlotItem] = []  # one polyline per drawn path
        self._marker_item: Optional[gl.GLScatterPlotItem] = None
        self._axis_items: List[gl.GLLinePlotItem] = []
        self._current_position_item: Optional[gl.GLScatterPlotItem] = None
        self._unreachable_item: Optional[gl.GLScatterPlotItem] = None
        self._warning_item: Optional[gl.GLScatterPlotItem] = None
        # Waypoint positions of the currently shown paths (camera frame), one Nx3 array per
        # path, kept so reachability verdicts can be drawn later by (path, waypoint) index
        # without re-passing the poses.
        self._path_positions: Optional[List[np.ndarray]] = None

        layout = QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.view)
        self.setLayout(layout)

    def clear(self) -> None:
        self._clear_waypoint_items()
        self.clear_reachability()
        for item in (self._scatter_item, self._current_position_item):
            if item is not None:
                self.view.removeItem(item)
        self._scatter_item = None
        self._current_position_item = None

    def show_current_position(self, position_camera_frame: np.ndarray) -> None:
        """Show/update a marker for the robot's current TCP position (camera frame),
        updated live during execution - see TracePanel."""
        position = np.asarray(position_camera_frame, dtype=np.float32).reshape(1, 3)
        if self._current_position_item is None:
            self._current_position_item = gl.GLScatterPlotItem(
                pos=position, color=CURRENT_POSITION_COLOR, size=CURRENT_POSITION_SIZE, pxMode=True
            )
            self.view.addItem(self._current_position_item)
        else:
            self._current_position_item.setData(pos=position)

    def clear_current_position(self) -> None:
        if self._current_position_item is not None:
            self.view.removeItem(self._current_position_item)
            self._current_position_item = None

    def show_point_cloud(self, point_cloud_xyz: np.ndarray, point_cloud_rgb: Optional[np.ndarray]) -> None:
        self.clear()
        positions, colors = self._prepare_point_cloud(point_cloud_xyz, point_cloud_rgb)
        if positions.shape[0] > 0:
            self._scatter_item = gl.GLScatterPlotItem(pos=positions, color=colors, size=POINT_SIZE, pxMode=True)
            self.view.addItem(self._scatter_item)
            centroid = positions.mean(axis=0)
            radius = float(np.linalg.norm(positions - centroid, axis=1).max()) if positions.shape[0] > 1 else 0.0
        else:
            centroid = np.zeros(3)
            radius = 0.0
        self.view.opts["center"] = pg.Vector(*centroid)
        self.view.setCameraPosition(distance=max(radius * 2.0, DEFAULT_CAMERA_DISTANCE_MM))

    def show_waypoints(self, paths_camera_frame: Sequence[Sequence[TransformationMatrix]]) -> None:
        """Draw one polyline per drawn path, with orientation axes on every waypoint.

        Paths are kept apart on purpose: drawing a single line through all of them would put
        a segment between the end of one and the start of the next, which is not a move the
        robot makes along the surface - it retreats, travels, and approaches again.
        """
        self._clear_waypoint_items()
        # Verdicts belong to the paths that were just replaced, so they are dropped here
        # rather than left pointing at indices of paths that no longer exist.
        self.clear_reachability()

        positions_per_path = [
            np.array([waypoint.translation for waypoint in path], dtype=np.float32)
            for path in paths_camera_frame
            if len(path) > 0
        ]
        if not positions_per_path:
            return
        self._path_positions = positions_per_path

        for positions in positions_per_path:
            if positions.shape[0] >= 2:
                path_item = gl.GLLinePlotItem(pos=positions, color=PATH_COLOR, width=4, antialias=True)
                self.view.addItem(path_item)
                self._path_items.append(path_item)

        all_positions = np.vstack(positions_per_path)
        self._marker_item = gl.GLScatterPlotItem(
            pos=all_positions, color=PATH_COLOR, size=WAYPOINT_MARKER_SIZE, pxMode=True
        )
        self.view.addItem(self._marker_item)

        all_waypoints = [waypoint for path in paths_camera_frame for waypoint in path]
        for axis_index, color in ((0, AXIS_COLOR_X), (1, AXIS_COLOR_Y), (2, AXIS_COLOR_Z)):
            axis_item = self._build_axis_item(all_waypoints, axis_index, color)
            self.view.addItem(axis_item)
            self._axis_items.append(axis_item)

    def show_reachability(
        self,
        unreachable_indices: Sequence[Tuple[int, int]],
        warned_indices: Sequence[Tuple[int, int]],
    ) -> None:
        """Mark waypoints the robot cannot reach (red) and ones suspected of an arm
        configuration flip (amber), addressed as (path index, waypoint index within it).

        Pairs that fall outside the paths currently shown are ignored rather than raising:
        the paths can be regenerated between running a check and drawing its result.
        """
        self.clear_reachability()
        if self._path_positions is None:
            return

        def positions_for(keys: Sequence[Tuple[int, int]]) -> List[np.ndarray]:
            found = []
            for path_index, waypoint_index in keys:
                if not 0 <= path_index < len(self._path_positions):
                    continue
                positions = self._path_positions[path_index]
                if 0 <= waypoint_index < positions.shape[0]:
                    found.append(positions[waypoint_index])
            return found

        # Red wins over amber - a waypoint the robot cannot reach at all should not be drawn
        # as a mere warning if it somehow ends up in both lists.
        unreachable_keys = set(map(tuple, unreachable_indices))
        warned_keys = set(map(tuple, warned_indices)) - unreachable_keys

        warned_positions = positions_for(sorted(warned_keys))
        if warned_positions:
            self._warning_item = gl.GLScatterPlotItem(
                pos=np.array(warned_positions, dtype=np.float32),
                color=WARNING_COLOR,
                size=WARNING_MARKER_SIZE,
                pxMode=True,
            )
            self.view.addItem(self._warning_item)

        unreachable_positions = positions_for(sorted(unreachable_keys))
        if unreachable_positions:
            self._unreachable_item = gl.GLScatterPlotItem(
                pos=np.array(unreachable_positions, dtype=np.float32),
                color=UNREACHABLE_COLOR,
                size=UNREACHABLE_MARKER_SIZE,
                pxMode=True,
            )
            self.view.addItem(self._unreachable_item)

    def clear_reachability(self) -> None:
        for item in (self._unreachable_item, self._warning_item):
            if item is not None:
                self.view.removeItem(item)
        self._unreachable_item = None
        self._warning_item = None

    def _clear_waypoint_items(self) -> None:
        for item in (self._marker_item, *self._path_items, *self._axis_items):
            if item is not None:
                self.view.removeItem(item)
        self._marker_item = None
        self._path_items = []
        self._axis_items = []
        self._path_positions = None

    def _build_axis_item(
        self, waypoints_camera_frame: Sequence[TransformationMatrix], axis_index: int, color
    ) -> gl.GLLinePlotItem:
        # One disjoint 2-point segment (origin -> tip) per waypoint, drawn in a single
        # GLLinePlotItem(mode="lines") rather than one item per waypoint.
        segment_points = []
        for waypoint in waypoints_camera_frame:
            origin = waypoint.translation
            axis_vector = waypoint.rotation.as_matrix()[:, axis_index]
            segment_points.append(origin)
            segment_points.append(origin + AXIS_LENGTH_MM * axis_vector)
        positions = np.array(segment_points, dtype=np.float32)
        return gl.GLLinePlotItem(pos=positions, color=color, width=2, antialias=True, mode="lines")

    def _prepare_point_cloud(self, xyz: np.ndarray, rgb: Optional[np.ndarray]):
        flat_xyz = xyz.reshape(-1, 3)
        valid = ~np.isnan(flat_xyz).any(axis=1)
        flat_xyz = flat_xyz[valid]

        if rgb is not None:
            flat_rgb = rgb.reshape(-1, rgb.shape[-1])[valid][:, :3].astype(np.float32) / 255.0
            alpha = np.ones((flat_rgb.shape[0], 1), dtype=np.float32)
            colors = np.hstack([flat_rgb, alpha])
        else:
            colors = np.tile(FALLBACK_POINT_COLOR, (flat_xyz.shape[0], 1)).astype(np.float32)

        if flat_xyz.shape[0] > MAX_DISPLAY_POINTS:
            stride = flat_xyz.shape[0] // MAX_DISPLAY_POINTS
            flat_xyz = flat_xyz[::stride]
            colors = colors[::stride]

        return flat_xyz, colors