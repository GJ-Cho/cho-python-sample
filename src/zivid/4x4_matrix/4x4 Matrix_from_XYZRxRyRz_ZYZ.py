from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation as R
from zividsamples.save_load_matrix import assert_affine_matrix_and_save


def _main():
    transform = np.eye(4)
    dataset_dir = Path(__file__).resolve().parent / "dataset"

    # x, y, z, Rx, Ry, Rz > translation + rotation
    value = [
        [   577.760562,  1472.276732,   835.002321,   133.771967,   135.398459,  -109.118401 ],
        [     0.000001,  1331.583861,   952.002070,    74.858307,   144.822208,  -108.317867 ],
        [     0.000001,  1331.583861,   952.002070,    0,   0,  0 ]
    ]
    rotation_convention = "ZYZ"  # Kawasaki is ZYZ intrinsic

    for i, vals in enumerate(value):
        translation = vals[:3]
        rotation = vals[3:]

        r = R.from_euler(rotation_convention, rotation, degrees=True)
        transform[:3, :3] = r.as_matrix()
        transform[:3, 3] = translation

        assert_affine_matrix_and_save(transform, dataset_dir / f"pos{i+1:02d}.yaml")


if __name__ == "__main__":
    _main()
