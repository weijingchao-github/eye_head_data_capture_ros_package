import os
import sys

path = os.path.dirname(__file__)
sys.path.insert(0, path)

import copy
import json
import pickle

import cv2
import pandas as pd
from calculate_target_3d_position_in_base import calculate_target_position_in_base_frame
from get_eye_gaze_direction import GetEyeGazeDirection
from get_target_position_in_head_camera_frame import GetTargetPosition
from recalculate_target_3d_position_in_base import calculate_gaze_ray_intersection


def get_images_and_parameters(folder_name, timestamp, human_parameters_path):
    head_camera_color_image = cv2.imread(
        os.path.join(folder_name, f"head_color_image/{timestamp}.jpg")
    )
    head_camera_depth_image = cv2.imread(
        os.path.join(folder_name, f"head_depth_image/{timestamp}.png"), -1
    )
    face_camera_color_image = cv2.imread(
        os.path.join(folder_name, f"face_image/{timestamp}.jpg")
    )
    head_camera_info = {}
    with open(os.path.join(folder_name, "color_camera_info.pkl"), "rb") as f:
        camera_info = pickle.load(f)
        head_camera_info["intrinsics"] = camera_info["intrinsics"]
        head_camera_info["depth_scale"] = camera_info["depth_scale"]
    with open(human_parameters_path, "r", encoding="utf-8") as f:
        human_parameters = json.load(f)
    df = pd.read_csv(os.path.join(folder_name, "imu_data.csv")).set_index("timestamp")
    if float(timestamp) in df.index:
        row = df.loc[float(timestamp)]
        current_head_camera_pose_quaternion_xyzw = (
            row["x"],
            row["y"],
            row["z"],
            row["w"],
        )
    else:
        raise Exception(f"未找到timestamp为 {timestamp} 的记录")
    row = df.iloc[0]
    initial_head_camera_pose_quaternion_xyzw = (row["x"], row["y"], row["z"], row["w"])
    return (
        head_camera_color_image,
        head_camera_depth_image,
        face_camera_color_image,
        head_camera_info,
        human_parameters,
        initial_head_camera_pose_quaternion_xyzw,
        current_head_camera_pose_quaternion_xyzw,
    )


def pipeline(folder_name, timestamp, human_parameters_path, monitor):
    (
        head_camera_color_image,
        head_camera_depth_image,
        face_camera_color_image,
        head_camera_info,
        human_parameters,
        initial_head_camera_pose_quaternion_xyzw,
        current_head_camera_pose_quaternion_xyzw,
    ) = get_images_and_parameters(folder_name, timestamp, human_parameters_path)
    head_camera_color_image_copy = copy.deepcopy(head_camera_color_image)
    get_target_position_in_head_camera_frame = GetTargetPosition(
        head_camera_color_image_copy,
        head_camera_depth_image,
        head_camera_info,
        monitor,
    )
    target_position_in_head_camera_frame = (
        get_target_position_in_head_camera_frame.select_point_and_get_3d()
    )
    target_position_in_base_frame = calculate_target_position_in_base_frame(
        target_position_in_head_camera_frame,
        initial_head_camera_pose_quaternion_xyzw,
        current_head_camera_pose_quaternion_xyzw,
        human_parameters,
    )
    get_eye_gaze_direction = GetEyeGazeDirection(human_parameters, monitor)
    face_camera_color_image_copy = copy.deepcopy(face_camera_color_image)
    eye_gaze_direction = get_eye_gaze_direction.pipeline(face_camera_color_image_copy)
    recalculated_target_position_in_base_frame = calculate_gaze_ray_intersection(
        human_parameters,
        initial_head_camera_pose_quaternion_xyzw,
        current_head_camera_pose_quaternion_xyzw,
        target_position_in_base_frame,
        eye_gaze_direction,
    )
    print(f"original: {target_position_in_base_frame}")
    print(f"recalculate: {recalculated_target_position_in_base_frame}")
    return recalculated_target_position_in_base_frame


def main():
    # read_file_path = ""
    # write_file_path = ""
    # with open(read_file_path, "r", encoding="utf-8") as read_file:
    #     with open(write_file_path, "a", encoding="utf-8") as write_file:
    #         pass
    folder_name = "/home/wjc/Storage/humanoid_head/eye_head_data_capture/src/yesense/scripts/captured_data_3/20250707_105422"
    timestamp = "1751856870.0271964"
    monitor = 0
    human_parameters_path = os.path.join(
        os.path.dirname(__file__), "human_parameters.json"
    )
    pipeline(folder_name, timestamp, human_parameters_path, monitor)


if __name__ == "__main__":
    main()
