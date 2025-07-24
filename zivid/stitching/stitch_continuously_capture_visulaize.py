"""
Stitch point clouds from a continuously rotating object without pre-alignment using Local Point Cloud Registration and apply Voxel Downsample.

It is assumed that the object is rotating around its own axis and the camera is stationary.
The camera settings should have defined a region of interest box that removes unnecessary points, keeping only the object to be stitched.

Note: This example uses experimental SDK features, which may be modified, moved, or deleted in the future without notice.

"""

import argparse
import time
from pathlib import Path

import numpy as np
import zivid
from zivid.experimental.point_cloud_export import export_unorganized_point_cloud
from zivid.experimental.point_cloud_export.file_format import PLY
from zivid.experimental.toolbox.point_cloud_registration import (
    LocalPointCloudRegistrationParameters,
    local_point_cloud_registration,
)
from zividsamples.display import display_pointcloud
import open3d as o3d


def _options() -> argparse.Namespace:
    """Function to read user arguments.

    Returns:
        Arguments from user

    """
    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument(
        "--settings-path",
        required=True,
        type=Path,
        help="Path to the camera settings YML file",
    )

    return parser.parse_args()


def show_pointcloud_open3d(xyz, rgb, vis=None, pcd=None):  # type: ignore
    # xyz: (N, 3), rgb: (N, 3)
    if pcd is None:
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(xyz)
        pcd.colors = o3d.utility.Vector3dVector(rgb / 255.0)
        vis = o3d.visualization.Visualizer()
        vis.create_window(window_name="Stitching", width=1920, height=1080)
        vis.add_geometry(pcd)
        # 배경색을 어두운 회색으로 설정
        render_option = vis.get_render_option()
        render_option.background_color = [0.1, 0.1, 0.1]
        vis.poll_events()
        vis.update_renderer()
        return vis, pcd
    else:
        assert vis is not None
        pcd.points = o3d.utility.Vector3dVector(xyz)
        pcd.colors = o3d.utility.Vector3dVector(rgb / 255.0)
        vis.update_geometry(pcd)
        vis.poll_events()
        vis.update_renderer()
        return vis, pcd


def _main() -> None:
    # user_options = _options()

    app = zivid.Application()

    # DOCTAG-START-STITCH-ROTATING-OBJECT-CONNECT-AND-LOAD-ROI
    print("Connecting to camera")
    camera = app.connect_camera()
    
    settings_file = Path(
        "C:/Users/GyeongjeCho/OneDrive - Zivid AS/Task/03_Customer/00_Image_Study/20250714_stitching_moving_robot/20250716/settings_03.yml"
    )

    # settings_file = Path(user_options.settings_path)
    print(f"Loading settings from file: {settings_file}")
    settings = zivid.Settings.load(settings_file)
    # DOCTAG-END-STITCH-ROTATING-OBJECT-CONNECT-AND-LOAD-ROI

    # DOCTAG-START-STITCH-ROTATING-OBJECT-CAPTURE-AND-STITCH
    previous_to_current_point_cloud_transform = np.eye(4)
    unorganized_stitched_point_cloud = zivid.UnorganizedPointCloud()
    registration_params = LocalPointCloudRegistrationParameters()

    vis = None
    pcd = None

    for number_of_captures in range(20):
        time.sleep(0.1)
        frame = camera.capture_2d_3d(settings)
        unorganized_point_cloud = (
            frame.point_cloud().to_unorganized_point_cloud().voxel_downsampled(voxel_size=1.0, min_points_per_voxel=2)
        )

        if number_of_captures != 0:
            local_point_cloud_registration_result = local_point_cloud_registration(
                target=unorganized_stitched_point_cloud,
                source=unorganized_point_cloud,
                parameters=registration_params,
                initial_transform=previous_to_current_point_cloud_transform,
            )
            if not local_point_cloud_registration_result.converged():
                print("Registration did not converge...")
                continue
            previous_to_current_point_cloud_transform = local_point_cloud_registration_result.transform().to_matrix()

            unorganized_stitched_point_cloud.transform(np.linalg.inv(previous_to_current_point_cloud_transform))
        unorganized_stitched_point_cloud.extend(unorganized_point_cloud)

        print(f"Captures done: {number_of_captures}")

        # open3d로 실시간 포인트 클라우드 업데이트
        xyz = unorganized_stitched_point_cloud.copy_data("xyz")
        rgb = unorganized_stitched_point_cloud.copy_data("rgba")[:, 0:3]
        vis, pcd = show_pointcloud_open3d(xyz, rgb, vis, pcd)

    print("Voxel-downsampling the stitched point cloud")
    unorganized_stitched_point_cloud = unorganized_stitched_point_cloud.voxel_downsampled(
        voxel_size=0.75, min_points_per_voxel=2
    )

    # 마지막 결과도 open3d로 갱신
    xyz = unorganized_stitched_point_cloud.copy_data("xyz")
    rgb = unorganized_stitched_point_cloud.copy_data("rgba")[:, 0:3] 
    vis, pcd = show_pointcloud_open3d(xyz, rgb, vis, pcd)

    # 창을 닫으려면 아래 코드 사용
    if vis is not None:
        vis.run()
        vis.destroy_window()

    file_name = Path(__file__).parent / "StitchedPointCloudOfRotatingObject.ply"
    export_unorganized_point_cloud(unorganized_stitched_point_cloud, PLY(str(file_name), layout=PLY.Layout.unordered))


if __name__ == "__main__":
    _main()
