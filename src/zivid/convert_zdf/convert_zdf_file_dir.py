"""
Convert point cloud data from a ZDF file to your preferred format (PLY, CSV, TXT, PNG, JPG, BMP, TIFF).

Example: $ python convert_zdf.py --ply Zivid3D.zdf

Available formats:
    PLY - Polygon File Format
    CSV, TXT - [X, Y, Z, r, g, b, SNR]
    PNG, JPG, BMP, TIFF - 2D RGB image

"""

import argparse
from pathlib import Path
import typing
from typing import Optional
from zivid.experimental.point_cloud_export import export_frame
from zivid.experimental.point_cloud_export.file_format import PCD, PLY, XYZ, ColorSpace

import cv2
import numpy as np
import zivid


def _create_depth_map(frame: zivid.Frame, minz: Optional[int] = 0, maxz: Optional[int] = 3000) -> typing.Any:
    """Create Depth map from Point cloud.

    Args:
        frame: A frame captured by a Zivid camera
        [Optional]
        minz =0, maxz=3000

    Returns:
        depth_map_color: Any depth map image \n
        cv2.imshow("Depth map", depth_map_color)

    """
    point_cloud = frame.point_cloud()
    depth_map = point_cloud.copy_data("z")
    np.clip(depth_map, minz, maxz, out=depth_map)
    depth_map_uint8 = ((depth_map - np.nanmin(depth_map)) / (np.nanmax(depth_map) - np.nanmin(depth_map)) * 255).astype(np.uint8)
    depth_map_color = cv2.applyColorMap(depth_map_uint8, cv2.COLORMAP_VIRIDIS)
    
    depth_map_color[np.isnan(depth_map)[:, :]] = 0
    depth_map_gray = cv2.cvtColor(depth_map_color, cv2.COLOR_BGR2GRAY)

    return depth_map_color


def _create_depth_map_2(frame: zivid.Frame, minz: Optional[int] = 0, maxz: Optional[int] = 3000) -> typing.Any:
    """Create Depth map from Point cloud.

    Args:
        frame: A frame captured by a Zivid camera
        [Optional]
        minz =0, maxz=3000

    Returns:
        depth_map_color: Any depth map image \n
        cv2.imshow("Depth map", depth_map_color)

    """
    point_cloud = frame.point_cloud()
    depth_map = point_cloud.copy_data("z")
    np.clip(depth_map, minz, maxz, out=depth_map)
    depth_map_uint8 = ((depth_map - np.nanmin(depth_map)) / (np.nanmax(depth_map) - np.nanmin(depth_map)) * 255).astype(np.uint8)
    depth_map_color = cv2.applyColorMap(depth_map_uint8, cv2.COLORMAP_VIRIDIS)
    
    depth_map_color[np.isnan(depth_map)[:, :]] = 0
    depth_map_gray = cv2.cvtColor(depth_map_color, cv2.COLOR_BGR2GRAY)\
    
    depth_map_uint8_double = depth_map_uint8
    depth_map_uint8_double[:][depth_map_uint8<=127] = (depth_map_uint8[:][depth_map_uint8 <= 127] / 127) * 255
    depth_map_uint8_double[:][depth_map_uint8>127] = ((depth_map_uint8[:][depth_map_uint8 > 127]-128) / 127) * 255
    depth_map_color_double = cv2.applyColorMap(depth_map_uint8_double, cv2.COLORMAP_HSV)
    depth_map_color_double[np.isnan(depth_map)[:, :]] = 0

    # return depth_map_color
    return depth_map_color_double


def _create_normal_map(frame: zivid.Frame,) -> typing.Any:
    """Create Normal map from Point cloud.

    Args:
        frame: A frame captured by a Zivid camera

    Returns:
        normals_map_bgr: Any Normal map image \n 
        cv2.imshow("Depth map before transform", normals_map_bgr)

    """
    point_cloud = frame.point_cloud()
    rgba = point_cloud.copy_data("rgba")
    normals = point_cloud.copy_data("normals")
    normals_colormap= rgba.copy() # size copy
    normals_colormap[:,:,:3] = 0.5 * (1 - normals) * 255
    normals_colormap[np.isnan(normals_colormap)[:, :]] = 0
    normals_map_bgr = cv2.cvtColor(normals_colormap, cv2.COLOR_RGBA2BGR) # (RGB > BGR)

    return normals_map_bgr


def _convert_2_ply(frame: zivid.Frame, file_path: Path) -> None:
    """Convert from frame to PLY.

    Args:
        frame: A frame captured by a Zivid camera
        file_name: File name without extension

    """
    # Order vs Unorder , SRGB vs Linear RGB 
    _3d_object = PLY(file_path, layout=PLY.Layout.ordered, color_space=ColorSpace.srgb)
    export_frame(frame, _3d_object)


def _convert_2_2d(frame: zivid.Frame, file_name: str) -> None:
    """Convert from point cloud to 2D image.

    Args:
        frame: A frame captured by a Zivid camera
        file_name: File name without extension

    """
    image_2d = frame.frame_2d().image_rgba_srgb()
    # if you want 2d from point cloud resolution, 
    # image_2d_in_point_cloud_resolution = frame.point_cloud().copy_image("rgba_srgb")

    image_2d.save(file_name)
    # image_2d_in_point_cloud_resolution.save(file_name)


def _main() -> None:
    dir = "C:/Zivid/cho-python-sample/sample"
    path = Path(dir)

    with zivid.Application():
        for file in path.glob("*.zdf"):
            print(f"Reading point cloud from file: {file.stem}")
            frame = zivid.Frame(file)

            ply_image_file = dir + "/" + file.stem + ".ply"
            _convert_2_ply(frame, ply_image_file)

            image_file = dir + "/" + file.stem + "_2d.png"
            _convert_2_2d(frame, image_file)
            
            depthmap = _create_depth_map(frame)
            depth_image_file = dir + "/" + file.stem + "_depth.png"
            cv2.imwrite(depth_image_file , depthmap)

            depthmap = _create_depth_map_2(frame)
            depth_image_file = dir + "/" + file.stem + "_depth2.png"
            cv2.imwrite(depth_image_file , depthmap)

            normalmap = _create_normal_map(frame)
            normal_image_file = dir + "/" + file.stem + "_normal.png"
            cv2.imwrite(normal_image_file , normalmap)

if __name__ == "__main__":
    _main()
