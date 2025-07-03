import json
import os

import numpy as np
from spatialmath import SO3, UnitQuaternion


def main():
    quaternion_current = (-1, -1, -1, -1)  # w, x, y, z
    quaternion_current = UnitQuaternion(quaternion_current)
    with open(
        os.path.join(os.path.dirname(__file__), "human_parameters.json"),
        "r",
        encoding="utf-8",
    ) as f:
        human_parameters = json.load(f)
    quaternion_initial = [
        human_parameters["initial_head_pose_quaternion"]["w"],
        human_parameters["initial_head_pose_quaternion"]["x"],
        human_parameters["initial_head_pose_quaternion"]["y"],
        human_parameters["initial_head_pose_quaternion"]["z"],
    ]
    quaternion_initial = UnitQuaternion(quaternion_initial)
    quaternion_initial_rotation_matrix_inv = quaternion_initial.SO3().inv()
    tf_ros_to_camera = SO3.Rz(90, "deg") * SO3.Ry(90, "deg")
    head_pose_rotation_matrix = (
        tf_ros_to_camera
        * quaternion_initial_rotation_matrix_inv
        * quaternion_current.SO3()
    )
    head_pose_quaternion = head_pose_rotation_matrix.q
    print(head_pose_quaternion)


if __name__ == "__main__":
    main()
