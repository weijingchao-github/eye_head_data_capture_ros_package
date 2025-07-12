import json
import math
import os

import numpy as np
from spatialmath import SE3, UnitQuaternion


def camera_to_ros_quaternion(q_camera):
    """
    将相机坐标系(A)中的四元数转换为ROS坐标系(B)中的四元数

    参数:
    q_camera (UnitQuaternion): 相机坐标系下的四元数

    返回:
    UnitQuaternion: ROS坐标系下的四元数
    """
    # 定义从相机坐标系到ROS坐标系的旋转矩阵
    # 相机坐标系(X右, Y下, Z前) -> ROS坐标系(X前, Y左, Z上)
    R_transform = np.array(
        [
            [0, 0, 1],  # 相机Z轴(前)对齐ROS X轴(前)
            [-1, 0, 0],  # 相机-X轴(左)对齐ROS Y轴(左)
            [0, -1, 0],  # 相机-Y轴(上)对齐ROS Z轴(上)
        ]
    )

    # 将旋转矩阵转换为四元数
    q_transform = UnitQuaternion(R_transform)

    # 应用坐标变换: q_ros = q_transform * q_camera * q_transform.inv()
    q_ros = q_transform * q_camera * q_transform.inv()

    return q_ros


def calculate_ray_intersection(
    original_position: np.ndarray,
    rotation_quaternion: UnitQuaternion,
    yaw: float,
    pitch: float,
    x_plane_distance: float,
) -> np.ndarray:
    """
    计算旋转后的坐标系A射出的射线与base坐标系YOZ平面平行且距离为x的平面的交点

    参数:
    - original_position: 坐标系A原始位置 [x, y, z]
    - rotation_quaternion: 坐标系A的旋转四元数
    - yaw: 射线在坐标系A中的偏航角（弧度）
    - pitch: 射线在坐标系A中的俯仰角（弧度）
    - x_plane_distance: 目标平面在base坐标系中的x坐标

    返回:
    - 交点在base坐标系中的坐标 [x, y, z]
    """
    # 构建坐标系A的原始位姿（只有平移）
    original_pose = SE3(
        original_position[0], original_position[1], original_position[2]
    )

    # 构建旋转矩阵（从四元数）
    rotation_matrix = rotation_quaternion.SE3()

    # 计算旋转后的坐标系A的位姿
    rotated_pose = rotation_matrix * original_pose

    # 在坐标系A中构建射线方向向量
    # 注意：在坐标系A中，初始射线方向为x轴正方向
    # 应用yaw和pitch旋转
    rotation_matrix_local = SE3.Rz(yaw) * SE3.Ry(pitch)
    ray_direction_local = np.array([1, 0, 0])  # 初始方向
    ray_direction_local = np.matmul(ray_direction_local, rotation_matrix_local.R)

    # 将射线方向转换到base坐标系
    ray_direction_base = np.matmul(rotated_pose.R, ray_direction_local)

    # 获取旋转后坐标系A的原点在base坐标系中的位置
    origin_base = rotated_pose.t

    # 计算射线参数t，使得射线与平面x=x_plane_distance相交
    if abs(ray_direction_base[0]) < 1e-10:  # 射线与目标平面平行
        if abs(origin_base[0] - x_plane_distance) < 1e-10:
            return origin_base  # 射线在平面上
        else:
            raise ValueError("射线与目标平面平行且不相交")

    t = (x_plane_distance - origin_base[0]) / ray_direction_base[0]

    # 计算交点坐标
    intersection_point_3d_position_in_base = origin_base + t * ray_direction_base

    return intersection_point_3d_position_in_base


def main():
    # 在ROS坐标系中，绕Z轴旋转的起点（即初始方向）是X轴正方向，绕Y轴旋转的起点（即初始方向）是X轴正方向，豆包说的
    eye_gaze_yaw_angle, eye_gaze_pitch_angle = (0.7465641289231583, -8.47386995253444)
    depth_to_base_orign = 2.07231207
    target_3d_position_in_base_frame = (2.07231207, -0.70498009, 0.04163128)
    current_head_camera_pose_quaternion_raw = (
        0.04509014325937586,
        -0.001837119286493963,
        -0.1553463458623336,
        0.9868287677205183,
    )  # x, y, z, w格式的头部姿态IMU四元数，在NWU坐标系下
    current_head_camera_pose_quaternion_wxyz = (
        current_head_camera_pose_quaternion_raw[-1],
        current_head_camera_pose_quaternion_raw[0],
        current_head_camera_pose_quaternion_raw[1],
        current_head_camera_pose_quaternion_raw[2],
    )
    current_head_camera_pose_quaternion = UnitQuaternion(
        current_head_camera_pose_quaternion_wxyz
    )
    head_pose_quaternion_in_ros_coordinate = camera_to_ros_quaternion(
        head_pose_quaternion_in_camera_coordinate
    )
    with open(
        os.path.join(os.path.dirname(__file__), "human_parameters.json"),
        "r",
        encoding="utf-8",
    ) as f:
        human_parameters = json.load(f)
    eye_origin_position_in_base = [
        human_parameters["eye_to_base"]["x"],
        human_parameters["eye_to_base"]["y"],
        human_parameters["eye_to_base"]["z"],
    ]
    recalculated_target_3d_position_in_base = calculate_ray_intersection(
        eye_origin_position_in_base,
        head_pose_quaternion_in_ros_coordinate,
        math.radians(eye_gaze_yaw_angle),
        math.radians(eye_gaze_pitch_angle),
        depth_to_base,
    )
    print(f"target_3d_position_in_base: {target_3d_position_in_base}")
    print(
        f"recalculated_target_3d_position_in_base: {tuple(recalculated_target_3d_position_in_base)}"
    )


if __name__ == "__main__":
    main()
