import os
import sys

path = os.path.dirname(__file__)
sys.path.insert(0, path)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["GRPC_VERBOSITY"] = "ERROR"
os.environ["GLOG_minloglevel"] = "2"

import copy
import csv
import json
import math
import pickle

import cv2
import pandas as pd
from calculate_target_3d_position_in_base import calculate_target_position_in_base_frame
from get_eye_gaze_direction import GetEyeGazeDirection
from get_rectified_head_pose import (
    get_current_rectified_head_pose,
    get_initial_rectified_head_pose,
)
from get_target_position_in_head_camera_frame import GetTargetPosition
from recalculate_target_3d_position_in_base import calculate_gaze_ray_intersection


def get_images_and_parameters(folder_name, timestamp, human_parameters_path):
    head_camera_color_image = cv2.imread(
        os.path.join(folder_name, f"head_color_image/{timestamp}.jpg")
    )
    head_camera_depth_image = cv2.imread(
        os.path.join(folder_name, f"head_depth_image/{timestamp}.png"), -1
    )
    head_camera_info = {}
    with open(os.path.join(folder_name, "color_camera_info.pkl"), "rb") as f:
        camera_info = pickle.load(f)
        head_camera_info["intrinsics"] = camera_info["intrinsics"]
        head_camera_info["depth_scale"] = camera_info["depth_scale"]
    with open(human_parameters_path, "r", encoding="utf-8") as f:
        human_parameters = json.load(f)
    df = pd.read_csv(
        os.path.join(folder_name, "eye_head_pose_sequence_30hz.csv")
    ).set_index("timestamp")
    if float(timestamp) in df.index:
        row = df.loc[float(timestamp)]
        rectified_current_head_camera_pose_quaternion_xyzw = (
            row["x"],
            row["y"],
            row["z"],
            row["w"],
        )
        eye_gaze_direction = [math.radians(row["yaw"]), math.radians(row["pitch"])]
    else:
        raise Exception(f"未找到timestamp为 {timestamp} 的记录")
    return (
        head_camera_color_image,
        head_camera_depth_image,
        head_camera_info,
        human_parameters,
        rectified_current_head_camera_pose_quaternion_xyzw,
        eye_gaze_direction,
    )


def pipeline(folder_name, timestamp, human_parameters_path, monitor):
    (
        head_camera_color_image,
        head_camera_depth_image,
        head_camera_info,
        human_parameters,
        rectified_current_head_camera_pose_quaternion_xyzw,
        eye_gaze_direction,
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
        rectified_current_head_camera_pose_quaternion_xyzw,
        human_parameters,
    )
    recalculated_target_position_in_base_frame = calculate_gaze_ray_intersection(
        human_parameters,
        rectified_current_head_camera_pose_quaternion_xyzw,
        target_position_in_base_frame,
        eye_gaze_direction,
    )
    print(f"original: {target_position_in_base_frame}")
    print(f"recalculate: {recalculated_target_position_in_base_frame}")
    return recalculated_target_position_in_base_frame


def get_rectified_target_position(folder_name, human_parameters_path, monitor):
    input_file = os.path.join(folder_name, "gaze_shift_start_and_end.csv")
    output_file = os.path.join(
        folder_name, "gaze_shift_start_and_end_with_target_position.csv"
    )
    with open(input_file, "r", newline="") as infile, open(
        output_file, "w", newline=""
    ) as outfile:
        reader = csv.reader(infile)
        writer = csv.writer(outfile)
        next(reader)  # 跳过infile的表头
        header = [
            "start_timestamp",
            "end_timestamp",
            "target_position_x",
            "target_position_y",
            "target_position_z",
        ]
        # outfile写入表头
        writer.writerow(header)

        for index, row in enumerate(reader):
            # 确保每行至少有start_timestamp和end_timestamp两列数据
            if len(row) == 2:
                end_timestamp = row[1]
                recalculated_target_position_in_base_frame = pipeline(
                    folder_name, end_timestamp, human_parameters_path, monitor
                )
                filled_row = row + list(recalculated_target_position_in_base_frame)
                writer.writerow(filled_row)
            else:
                print(f"警告：跳过不完整的行,行号(表头为1): {index+2}")
    print("Stage4: Get Rectified End Timestamp's Target Position Finished.")


def main():
    folder_name = "/home/wjc/Storage/humanoid_head/eye_head_data_capture/src/yesense/scripts/captured_data_3/20250707_105422"
    input_file = os.path.join(folder_name, "gaze_shift_start_and_end.csv")
    output_file = os.path.join(
        folder_name, "gaze_shift_start_and_end_with_target_position.csv"
    )
    human_parameters_path = os.path.join(
        os.path.dirname(__file__), "human_parameters.json"
    )
    monitor = 0
    with open(input_file, "r", newline="") as infile, open(
        output_file, "w", newline=""
    ) as outfile:
        reader = csv.reader(infile)
        writer = csv.writer(outfile)
        next(reader)  # 跳过infile的表头
        header = [
            "start_timestamp",
            "end_timestamp",
            "target_position_x",
            "target_position_y",
            "target_position_z",
        ]
        # outfile写入表头
        writer.writerow(header)

        for index, row in enumerate(reader):
            # 确保每行至少有start_timestamp和end_timestamp两列数据
            if len(row) == 2:
                end_timestamp = row[1]
                recalculated_target_position_in_base_frame = pipeline(
                    folder_name, end_timestamp, human_parameters_path, monitor
                )
                filled_row = row + list(recalculated_target_position_in_base_frame)
                writer.writerow(filled_row)
            else:
                print(f"警告：跳过不完整的行,行号(表头为1): {index+2}")
    print("Finished!")


if __name__ == "__main__":
    main()
