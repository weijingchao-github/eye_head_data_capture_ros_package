import os
import sys

path = os.path.dirname(__file__)
sys.path.insert(0, path)
import math

import numpy as np
from calculate_rectified_head_pose import calculate_rectified_head_pose
from spatialmath import SE3, UnitQuaternion

# from tf.transformations import euler_from_quaternion


def calculate_gaze_ray_intersection(
    human_parameters,
    current_head_camera_pose_quaternion_xyzw,
    target_position_in_base_frame,
    eye_gaze_direction,
):
    head_pose_quaternion_wxyz = (
        current_head_camera_pose_quaternion_xyzw[-1],
        current_head_camera_pose_quaternion_xyzw[0],
        current_head_camera_pose_quaternion_xyzw[1],
        current_head_camera_pose_quaternion_xyzw[2],
    )
    head_pose_quaternion = UnitQuaternion(head_pose_quaternion_wxyz)
    eye_gaze_yaw_angle, eye_gaze_pitch_angle = eye_gaze_direction
    x_plain_to_base_origin = target_position_in_base_frame[0]
    eye_origin_position_in_base_frame = [
        human_parameters["eye_to_base"]["x"],
        human_parameters["eye_to_base"]["y"],
        human_parameters["eye_to_base"]["z"],
    ]
    # 计算头部旋转后的眼球坐标系的位姿(先旋转然后相对物体坐标系平移)
    eye_frame_position_and_pose = head_pose_quaternion.SE3() * SE3(
        eye_origin_position_in_base_frame
    )
    # 在眼球坐标系中构建视线方向向量
    tf_gaze_ray_pose = SE3.Rz(math.radians(eye_gaze_yaw_angle)) * SE3.Ry(
        math.radians(eye_gaze_pitch_angle)
    )
    gaze_ray_in_eye_frame = np.array([1, 0, 0])  # 初始方向
    gaze_ray_in_eye_frame = (tf_gaze_ray_pose * gaze_ray_in_eye_frame).reshape(-1)
    # 将视线方向转换到base坐标系
    ray_direction_base = eye_frame_position_and_pose * gaze_ray_in_eye_frame
    ray_direction_base = ray_direction_base.reshape(-1)
    # 获取头部旋转后眼球坐标系的原点在base坐标系中的位置
    eye_frame_orign_to_base_frame_orign = eye_frame_position_and_pose.t
    unit_gaze_ray_in_base_frame = np.array(
        [
            ray_direction_base[0] - eye_frame_orign_to_base_frame_orign[0],
            ray_direction_base[1] - eye_frame_orign_to_base_frame_orign[1],
            ray_direction_base[2] - eye_frame_orign_to_base_frame_orign[2],
        ]
    )

    t = (
        x_plain_to_base_origin - eye_frame_orign_to_base_frame_orign[0]
    ) / unit_gaze_ray_in_base_frame[0]

    # 计算交点坐标
    recalculated_target_position_in_base_frame = (
        eye_frame_orign_to_base_frame_orign + t * unit_gaze_ray_in_base_frame
    )

    return recalculated_target_position_in_base_frame


# def main():
#     # 在ROS坐标系中，绕Z轴旋转的起点（即初始方向）是X轴正方向，绕Y轴旋转的起点（即初始方向）是X轴正方向，豆包说的
#     eye_gaze_yaw_angle, eye_gaze_pitch_angle = (0.7465641289231583, -8.47386995253444)
#     depth_to_base_orign = 2.07231207
#     target_3d_position_in_base_frame = (2.07231207, -0.70498009, 0.04163128)
#     rectified_current_head_camera_pose_quaternion_raw = (
#         -0.01662833,
#         -0.03782546,
#         -0.0215886,
#         0.99891274,
#     )  # x, y, z, w格式的头部姿态IMU四元数，在NWU坐标系下
#     current_head_camera_pose_quaternion_wxyz = (
#         rectified_current_head_camera_pose_quaternion_raw[-1],
#         rectified_current_head_camera_pose_quaternion_raw[0],
#         rectified_current_head_camera_pose_quaternion_raw[1],
#         rectified_current_head_camera_pose_quaternion_raw[2],
#     )
#     current_head_camera_pose_quaternion = UnitQuaternion(
#         current_head_camera_pose_quaternion_wxyz
#     )
#     with open(
#         os.path.join(os.path.dirname(__file__), "human_parameters.json"),
#         "r",
#         encoding="utf-8",
#     ) as f:
#         human_parameters = json.load(f)
#     eye_origin_position_in_base_frame = [
#         human_parameters["eye_to_base"]["x"],
#         human_parameters["eye_to_base"]["y"],
#         human_parameters["eye_to_base"]["z"],
#     ]
#     recalculated_target_3d_position_in_base_frame = calculate_gaze_ray_intersection(
#         eye_origin_position_in_base_frame,
#         current_head_camera_pose_quaternion,
#         math.radians(eye_gaze_yaw_angle),
#         math.radians(eye_gaze_pitch_angle),
#         depth_to_base_orign,
#     )
#     print(f"target_3d_position_in_base_frame: {target_3d_position_in_base_frame}")
#     print(
#         f"recalculated_target_3d_position_in_base_frame: {tuple(recalculated_target_3d_position_in_base_frame)}"
#     )


# if __name__ == "__main__":
#     main()
