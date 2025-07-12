# 来源https://www.doubao.com/thread/we0c6d0f18bc930bf
import json
import os

import numpy as np


def quaternion_conjugate(q):
    """计算四元数的共轭"""
    return np.array([q[0], -q[1], -q[2], -q[3]])


def quaternion_multiply(q1, q2):
    """四元数乘法"""
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
    return np.array([w, x, y, z])


def main():
    quaternion_current = np.array(
        [
            0.9868287677205183,
            0.04509014325937586,
            -0.001837119286493963,
            -0.1553463458623336,
        ]
    )  # w, x, y, z
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
    delta_q = quaternion_multiply(
        quaternion_conjugate(quaternion_initial), quaternion_current
    )
    print(delta_q)


if __name__ == "__main__":
    main()
