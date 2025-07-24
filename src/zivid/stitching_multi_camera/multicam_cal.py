import os
import cv2
import numpy as np
import open3d as o3d
import zivid
from pathlib import Path

location_dir = Path.cwd()

def main() -> None:
    with zivid.Application():
        # imgXX.zdf : point cloud for Multi camera calibration
        # img_test_XX.zdf : point cloud for stitching 
        calibration_inputs = []
        idata = 1
        while True:
            frame_file_path = location_dir / "zivid" / "stitching_multi_camera" / f"img{idata:02d}.zdf"
            print("Loading ZDF for Multi camera calibration : ", frame_file_path)
            if frame_file_path.is_file():

                print(f"Detect feature points from img{idata:02d}.zdf")
                frame = zivid.Frame(frame_file_path)
                point_cloud = frame.point_cloud()
                detection_result = zivid.calibration.detect_feature_points(point_cloud)

                if not detection_result.valid():
                    raise RuntimeError(f"Failed to detect feature points from frame {frame_file_path}")

                calibration_inputs.append(detection_result)

            else:
                break

            idata += 1
            
        results = zivid.calibration.calibrate_multi_camera(calibration_inputs)

        if results:
            transforms = results.transforms()
            residuals = results.residuals()
            for i in range(len(transforms)):
                print(transforms[i])

        print("End : Multi ZDF = Multi camera Calibration")

        idata = 1 # reset count
        stitched_point_cloud = o3d.geometry.PointCloud()
        
        # Loading zdf for stitching and Stitching!
        for i in range(len(transforms)):
            frame_file_path = location_dir / "zivid" / "stitching_multi_camera" / f"img_test_{idata:02d}.zdf"
            print("Loading ZDF for Stitching : ", frame_file_path)
            if frame_file_path.is_file():
                print(f"Stitching point cloud from img_test_{idata:02d}.zdf")
                frame = zivid.Frame(frame_file_path)
                point_cloud = frame.point_cloud()
                point_cloud.transform(transforms[i])
                xyz = point_cloud.copy_data("xyz")
                rgba = point_cloud.copy_data("rgba")

                xyz = np.nan_to_num(xyz).reshape(-1, 3)
                rgb = rgba[:, :, 0:3].reshape(-1, 3)
                # Stitching
                if i==0:
                    stitched_xyz = xyz
                    stitched_rgb = rgb
                else:
                    stitched_xyz = np.concatenate((stitched_xyz,xyz), axis=0)
                    stitched_rgb = np.concatenate((stitched_rgb,rgb), axis=0)
            
            idata += 1

        stitched_point_cloud.points = o3d.utility.Vector3dVector(stitched_xyz)
        stitched_point_cloud.colors = o3d.utility.Vector3dVector(stitched_rgb / 255)
        stitched_point_cloud = o3d.geometry.PointCloud.remove_non_finite_points(
            stitched_point_cloud, remove_nan=True, remove_infinite=True
        )

        visualizer = o3d.visualization.Visualizer()
        visualizer.create_window()
        visualizer.add_geometry(stitched_point_cloud)
        visualizer.run()
        visualizer.destroy_window()


if __name__ == "__main__":
    main()