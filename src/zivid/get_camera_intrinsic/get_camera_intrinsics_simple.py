"""
Read intrinsic parameters from the Zivid camera (OpenCV model) or estimate them from the point cloud.

Note: This example uses experimental SDK features, which may be modified, moved, or deleted in the future without notice.

"""

import zivid
from zivid.experimental import calibration

def _main():
    with zivid.Application():
        dir = "C:/Zivid/cho-python-sample/sample/"
        zdf_path = "sample_MR130.zdf"
        print(f"Reading point cloud from file: {zdf_path}")
        frame = zivid.Frame(dir + zdf_path)
        estimated_intrinsics = calibration.estimate_intrinsics(frame)
        estimated_intrinsics_path = "intrinsic_sample_MR130.yml"
        print(
            f"Saving estimated camera intrinsics for loaded zdf: {estimated_intrinsics_path}"
        )
        estimated_intrinsics.save(dir + estimated_intrinsics_path)


if __name__ == "__main__":
    _main()
