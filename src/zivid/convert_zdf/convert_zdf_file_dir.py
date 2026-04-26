"""
Convert point cloud data from a ZDF file to your preferred format (PLY, CSV, TXT, PNG, JPG, BMP, TIFF).

Example: $ python convert_zdf.py --ply Zivid3D.zdf

Available formats:
    PLY - Polygon File Format
    CSV, TXT - [X, Y, Z, r, g, b, SNR]
    PNG, JPG, BMP, TIFF - 2D RGB image

"""

from pathlib import Path
import typing
from typing import Optional
from zivid.experimental.point_cloud_export import export_frame
from zivid.experimental.point_cloud_export.file_format import PLY, ColorSpace
from zividsamples.paths import get_sample_data_path

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

    return depth_map_color, depth_map_gray


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
    
    depth_map_uint8_double = depth_map_uint8.copy()
    depth_map_uint8_double[:][depth_map_uint8<=127] = (depth_map_uint8[:][depth_map_uint8 <= 127] / 127) * 255
    depth_map_uint8_double[:][depth_map_uint8>127] = ((depth_map_uint8[:][depth_map_uint8 > 127]-128) / 127) * 255
    depth_map_color_double = cv2.applyColorMap(depth_map_uint8_double, cv2.COLORMAP_HSV)
    depth_map_color_double[np.isnan(depth_map)[:, :]] = 0
    depth_map_gray_double = cv2.cvtColor(depth_map_color_double, cv2.COLOR_BGR2GRAY)

    # return depth_map_color, depth_map_gray
    return depth_map_color_double, depth_map_gray_double


def _create_normal_map(frame: zivid.Frame,) -> typing.Any:
    """Create Normal map from Point cloud.

    Args:
        frame: A frame captured by a Zivid camera

    Returns:
        normals_map_bgr: Any Normal map image \n 
        cv2.imshow("Normal map", normals_map_bgr)

    """
    point_cloud = frame.point_cloud()
    rgba = point_cloud.copy_data("rgba")
    normals = point_cloud.copy_data("normals")
    normals_colormap= rgba.copy() # size copy
    normals_colormap[:,:,:3] = 0.5 * (1 - normals) * 255
    normals_colormap[np.isnan(normals).any(axis=2)] = 0
    normals_map_bgr = cv2.cvtColor(normals_colormap, cv2.COLOR_RGBA2BGR) # (RGB > BGR)

    return normals_map_bgr


def _create_custom_colormap():
    """Create custom colormap for red-green colorblind friendly visualization.
    Please refer to 1.Red-green Variation #1 in the following website.
    https://visualisingdata.com/2019/08/five-ways-to-design-for-red-green-colour-blindness/

    Returns:
        Custom colormap array for OpenCV
    """
    # Red-green Variation #1 colors (BGR format for OpenCV)
    colors = np.array([
        [37, 67, 219],    # DB4325
        [71, 162, 237],   # EDA247  
        [188, 225, 230],  # E6E1BC
        [173, 196, 87],   # 57C4AD
        [100, 97, 0]      # 006164
    ], dtype=np.uint8)
    
    # Create 256-entry colormap by interpolating between colors
    colormap = np.zeros((256, 1, 3), dtype=np.uint8)
    for i in range(256):
        # Map to color index (0-4)
        pos = i / 255.0 * 4
        idx = int(pos)
        frac = pos - idx
        if idx >= 4:
            idx = 3
            frac = 1.0
        if frac == 0.0:
            colormap[i, 0] = colors[idx]
        else:
            # Linear interpolation between adjacent colors
            colormap[i, 0] = colors[idx] * (1 - frac) + colors[idx + 1] * frac
    
    return colormap


def _create_snr_map(frame: zivid.Frame,) -> typing.Any:
    """Create SNR map from Point cloud.

    Args:
        frame: A frame captured by a Zivid camera

    Returns:
        snr_map_bgr: Any SNR map image \n 
        cv2.imshow("SNR map", snr_map_bgr)

    """
    point_cloud = frame.point_cloud()
    snr = point_cloud.copy_data("snr")
    mask_zero = (snr == 0)
    snr = np.minimum(snr, 64)
    log_snr = np.log2(snr, where=(snr > 0), out=np.zeros_like(snr, dtype=float))
    log_snr[~mask_zero] += 1

    snr_uint8 = (log_snr / 7.0 * 255).astype(np.uint8)
    custom_colormap = _create_custom_colormap()
    snr_map_bgr = cv2.applyColorMap(snr_uint8, custom_colormap)
    
    snr_map_bgr[np.isnan(snr)[:, :]] = 0

    return snr_map_bgr


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


def _repo_root() -> Path:
    """Walk up from this file until the .git directory is found."""
    for parent in Path(__file__).resolve().parents:
        if (parent / ".git").is_dir():
            return parent
    return Path(__file__).resolve().parents[3]  # fallback: src/zivid/convert_zdf/ → repo root


def _main() -> None:
    sample_dir = _repo_root() / "sample"
    zivid_data_dir = get_sample_data_path()  # C:/ProgramData/Zivid on Windows

    print(f"Repository sample directory : {sample_dir}")
    print(f"Zivid ProgramData directory : {zivid_data_dir}")
    print()

    with zivid.Application():
        for file in sample_dir.glob("*.zdf"):
            print(f"Reading point cloud from file: {file.stem}")
            frame = zivid.Frame(file)

            _convert_2_ply(frame, sample_dir / (file.stem + ".ply"))
            _convert_2_2d(frame, str(sample_dir / (file.stem + "_2d.png")))

            depthmap, depthmap_gray = _create_depth_map(frame)
            cv2.imwrite(str(sample_dir / (file.stem + "_depth.png")), depthmap)
            cv2.imwrite(str(sample_dir / (file.stem + "_depth_gray.png")), depthmap_gray)

            depthmap2, depthmap2_gray = _create_depth_map_2(frame)
            cv2.imwrite(str(sample_dir / (file.stem + "_depth2.png")), depthmap2)
            cv2.imwrite(str(sample_dir / (file.stem + "_depth2_gray.png")), depthmap2_gray)

            normalmap = _create_normal_map(frame)
            cv2.imwrite(str(sample_dir / (file.stem + "_normal.png")), normalmap)

            snr_map = _create_snr_map(frame)
            cv2.imwrite(str(sample_dir / (file.stem + "_snr.png")), snr_map)

if __name__ == "__main__":
    _main()
