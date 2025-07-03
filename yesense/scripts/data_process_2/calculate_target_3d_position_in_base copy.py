# 可参考https://www.doubao.com/thread/w2076d619dac21fe2
import json
import os

import numpy as np
from spatialmath import SE3, UnitQuaternion


def main():
    target_position_in_head_pose_coordinate = [-1, -1, -1]
    quaternion_current = (
        0,
        0,
        0.7071,
        0.7071,
    )  # x, y, z, w格式的头部姿态IMU四元数，在RealSense IMU坐标系下
    # 通过运行rosrun tf tf_echo camera_imu_optical_frame camera_depth_optical_frame
    # 发现IMU坐标系和彩色摄像头坐标系只有三维位置偏移，没有旋转偏移
    # IMU坐标系目前调研的和彩色摄像头坐标系基坐标形式是一样的，后面通过实验来确定这句话对不对，然后改这行注释
    quaternion_current = UnitQuaternion(quaternion_current)
    with open(
        os.path.join(os.path.dirname(__file__), "human_parameters.json"),
        "r",
        encoding="utf-8",
    ) as f:
        human_parameters = json.load(f)
    # base坐标系原点在头顶相机坐标系描述下的固定位置偏差(x, y, z)
    base_point_to_head_color_camera_position = [
        human_parameters["head_color_camera_to_base"]["y"],
        human_parameters["head_color_camera_to_base"]["z"],
        -human_parameters["head_color_camera_to_base"]["x"],
    ]
    quaternion_initial = [
        human_parameters["initial_head_pose_quaternion"]["w"],
        human_parameters["initial_head_pose_quaternion"]["x"],
        human_parameters["initial_head_pose_quaternion"]["y"],
        human_parameters["initial_head_pose_quaternion"]["z"],
    ]
    quaternion_initial = UnitQuaternion(quaternion_initial)
    base_point_in_initial_head_pose_coordinate = (
        quaternion_initial.SE3().inv()
        @ np.array(base_point_to_head_color_camera_position.append(1))
    )[:3]
    tf_ros_to_camera = SE3.Rz(90, "deg") * SE3.Ry(90, "deg")
    tf_base_to_head_pose_coordinate = (
        tf_ros_to_camera
        * quaternion_current.SE3()
        * SE3(base_point_in_initial_head_pose_coordinate)
    )
    target_position_in_base = (
        tf_base_to_head_pose_coordinate
        * np.array(target_position_in_head_pose_coordinate.append(1))
    )[:3]
    print(target_position_in_base)


if __name__ == "__main__":
    main()
