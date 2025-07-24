from scipy.spatial.transform import Rotation as R
import numpy as np
from pathlib import Path
import zivid
import yaml
from typing import Union

# def assert_affine_matrix_and_save(matrix: np.ndarray, yaml_path: Path):
#     """Save transformation to directory. (Old version, just for reference)

#     Args:
#         matrix: 4x4 transformation matrix
#         yaml_path: Path to the YAML file

#     """
#     # Checks if matrix is affine
#     zivid.calibration.Pose(matrix)

#     dict_for_yaml = {}
#     dict_for_yaml["__version__"] = {"serializer": 1, "data": 1}
#     dict_for_yaml["FloatMatrix"] = {"Data": matrix.tolist()}

#     with open(yaml_path, "w", encoding="utf-8") as outfile:
#         yaml.safe_dump(dict_for_yaml, outfile, default_flow_style=None, sort_keys=False)

def assert_affine(matrix: Union[np.ndarray, zivid.Matrix4x4]) -> None:
    """Ensures that the matrix is affine.

    Args:
        matrix: 4x4 transformation matrix, np.ndarray or zivid.Matrix4x4

    Raises:
        RuntimeError: If matrix is not affine

    """
    try:
        zivid.calibration.Pose(matrix)
    except RuntimeError as ex:
        raise RuntimeError("matrix is not affine") from ex
    
def assert_affine_matrix_and_save(matrix: Union[np.ndarray, zivid.Matrix4x4], yaml_path: Path) -> None:
    """Save transformation matrix to YAML.

    Args:
        matrix: 4x4 transformation matrix, np.ndarray or zivid.Matrix4x4
        yaml_path: Path to the YAML file

    """
    assert_affine(matrix)

    zivid.Matrix4x4(matrix).save(yaml_path)

def _main():
    transform = np.eye(4)
    dir = "C:/Users/Public/Documents/cho-python-sample/zivid/4x4_matrix/dataset/" # Save yaml file here! 

    # x, y, z, Rx, Ry, Rz > translation + rotation
    value = [
        [   577.760562,  1472.276732,   835.002321,   133.771967,   135.398459,  -109.118401 ],
        [     0.000001,  1331.583861,   952.002070,    74.858307,   144.822208,  -108.317867 ],
        [     0.000001,  1331.583861,   952.002070,    0,   0,  0 ]
    ]
    for i in range(len(value)) :
        translation = value[i][:3]
        rotation = value[i][3:]
        RotationConvention = "ZYZ" # Kawasaki is zyz intrinsic.
    
        # r = R.from_rotvec(rotation, degrees=True)
        
        # rotation = R.from_matrix(rotation_matrix)
        # roll_pitch_yaw = rotation.as_euler(RotationConvention)

        r = R.from_euler(RotationConvention, rotation, degrees=True)
        # print("rotation 3x3", "\n" , r.as_matrix())

        transform[:3, :3] = r.as_matrix()
        transform[:3, 3] = translation
        
        # print(transform)

        # _save_pose(save_dir, transform)
        save_dir = dir + f"pos{i+1:02d}.yaml"
        assert_affine_matrix_and_save(transform, save_dir)
    
if __name__ == "__main__":
    _main()