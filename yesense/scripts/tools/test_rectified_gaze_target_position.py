import json
import math

import numpy as np
from spatialmath import SE3


def calculate_gaze_ray_intersection(
    target_head_pose,
    target_position_in_base_frame,
    eye_gaze_direction,
):
    eye_gaze_yaw_radians, eye_gaze_pitch_radians = eye_gaze_direction
    x_plain_to_base_origin = target_position_in_base_frame[0]
    eye_origin_position_in_base_frame = [0.09, 0, 0.13]
    # 计算头部旋转后的眼球坐标系的位姿(先旋转然后相对物体坐标系平移)
    target_head_pose_SE3 = (
        SE3.Rz(target_head_pose[0])
        * SE3.Ry(target_head_pose[1])
        * SE3.Rx(target_head_pose[2])
    )
    eye_frame_position_and_pose = target_head_pose_SE3 * SE3(
        eye_origin_position_in_base_frame
    )
    # 在眼球坐标系中构建视线方向向量
    tf_gaze_ray_pose = SE3.Rz(eye_gaze_yaw_radians) * SE3.Ry(eye_gaze_pitch_radians)
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


def main():
    file_path = "/home/wjc/Storage/humanoid_head/eye_head_data_capture/src/yesense/scripts/captured_data_3/gaze_shift_euler_angle_relative_movement/train_val_dataset/val_start_and_end_eye_head_pose_dataset.json"
    with open(file_path, "r", encoding="utf-8") as f:
        json_data = json.load(f)  # 加载后的json_data是一个字典
    for key in json_data:
        data_one_piece = json_data[key]
        current_eye_pose = np.array(data_one_piece["current_eye_pose"])
        current_head_pose = np.array(data_one_piece["current_head_pose"])
        target_delta_eye_movement = np.array(
            data_one_piece["target_delta_eye_movement"]
        )
        target_delta_head_movement = np.array(
            data_one_piece["target_delta_head_movement"]
        )
        target_eye_pose = current_eye_pose + target_delta_eye_movement
        target_head_pose = current_head_pose + target_delta_head_movement
        target_position = np.array(data_one_piece["target_position"])
        recalculated_target_position_in_base_frame = calculate_gaze_ray_intersection(
            target_head_pose, target_position, target_eye_pose
        )
        dist = math.dist(target_position, recalculated_target_position_in_base_frame)
        print(dist)


if __name__ == "__main__":
    main()
