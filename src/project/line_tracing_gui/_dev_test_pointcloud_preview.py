"""
Standalone manual test for PointCloudViewerWidget, using the local sample
point cloud and a straight-line waypoint path - no camera/robot needed.

Run:
    python -m line_tracing_gui._dev_test_pointcloud_preview

"""

import sys
from pathlib import Path

import zivid
from PyQt5.QtWidgets import QApplication, QMainWindow
from zividsamples.transformation_matrix import TransformationMatrix

from line_tracing_gui.geometry.waypoint_builder import build_waypoints
from line_tracing_gui.widgets.pointcloud_viewer_widget import PointCloudViewerWidget

SAMPLE_ZDF_PATH = Path(__file__).resolve().parent.parent.parent.parent / "sample" / "sample_MR130.zdf"


def _main() -> None:
    app_qt = QApplication(sys.argv)
    zivid_app = zivid.Application()  # noqa: F841
    frame = zivid.Frame(str(SAMPLE_ZDF_PATH))
    point_cloud = frame.point_cloud()
    xyz = point_cloud.copy_data("xyz")
    rgb = point_cloud.copy_data("rgba_srgb")

    line_points_px = [(400.0, 500.0), (600.0, 500.0), (800.0, 500.0)]
    result = build_waypoints(line_points_px, xyz, TransformationMatrix(), sample_spacing_mm=15.0)
    print(f"{len(result.waypoints)} waypoints, {result.skipped_pixel_count} skipped")

    window = QMainWindow()
    window.setWindowTitle("PointCloudViewerWidget manual test")
    viewer = PointCloudViewerWidget()
    viewer.show_point_cloud(xyz, rgb)
    viewer.show_waypoints(result.waypoints)
    window.setCentralWidget(viewer)
    window.resize(900, 700)
    window.show()
    sys.exit(app_qt.exec_())


if __name__ == "__main__":
    _main()
