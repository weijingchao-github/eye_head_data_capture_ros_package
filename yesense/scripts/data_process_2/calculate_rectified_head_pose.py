import json
import os

import numpy as np
import transforms3d as t3d
from spatialmath import SO3, UnitQuaternion


def main():
    quaternion_current = np.array(
        [
            0.9868287677205183,
            0.04509014325937586,
            -0.001837119286493963,
            -0.1553463458623336,
        ]
    )  # w, x, y, z
    quaternion_current = UnitQuaternion(quaternion_current)
    with open(
        os.path.join(os.path.dirname(__file__), "human_parameters.json"),
        "r",
        encoding="utf-8",
    ) as f:
        human_parameters = json.load(f)
    quaternion_initial = [
        human_parameters["initial_head_camera_pose_quaternion"]["w"],
        human_parameters["initial_head_camera_pose_quaternion"]["x"],
        human_parameters["initial_head_camera_pose_quaternion"]["y"],
        human_parameters["initial_head_camera_pose_quaternion"]["z"],
    ]
    quaternion_initial = UnitQuaternion(quaternion_initial)
    quaternion_initial_so3_inv = quaternion_initial.SO3().inv()
    rectified_head_pose_so3 = quaternion_current.SO3() * quaternion_initial_so3_inv
    rectified_head_pose_quaternion = t3d.quaternions.mat2quat(
        rectified_head_pose_so3.A
    )  # w, x, y,z
    print(rectified_head_pose_quaternion)


if __name__ == "__main__":
    main()
