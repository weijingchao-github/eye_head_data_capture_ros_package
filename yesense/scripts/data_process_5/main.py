import os
import sys

path = os.path.dirname(__file__)
sys.path.insert(0, path)

import csv

from downsample_sequence import downsample_eye_and_head_pose_sequence
from face_and_head_image_viewer import view_face_and_head_image
from generate_json_file import generate_dataset_json_file_head6d_eye3d
from saccade_end_state_data_process_pipeline4 import get_rectified_target_position
from smooth_raw_data import smooth_eye_and_head_pose_raw_data


def main():
    data_folder_path = "/home/wjc/Storage/humanoid_head/eye_head_data_capture/src/yesense/scripts/captured_data_3/20250919_201243"
    cnt_abort = 9  # 跳过几条数据（表头已默认跳过，这几条数据里不含表头），可以为0
    initial_head_pose_quaternion_xyzw = (
        -0.021861306928578256,
        0.07042404991573166,
        -0.09718568553875656,
        0.9925308453540389,
    )
    show_image_monitor_index = 0
    human_parameters_path = os.path.join(
        os.path.dirname(__file__), "human_parameters.json"
    )

    # Stage1: Smooth eye and head pose raw data
    smooth_eye_and_head_pose_raw_data(
        data_folder_path,
        cnt_abort,
        human_parameters_path,
        initial_head_pose_quaternion_xyzw,
    )
    # Stage2: Downsample eye and head pose sequence from 60Hz to 30Hz
    downsample_eye_and_head_pose_sequence(data_folder_path)

    # Stage3: Find gaze shift start and end timestamp
    with open(
        os.path.join(data_folder_path, "gaze_shift_start_and_end.csv"),
        "w",
        newline="",
        encoding="utf-8",
    ) as f:
        writer = csv.writer(f)
        header = ["start_timestamp", "end_timestamp"]
        writer.writerow(header)
    print("注意: 选择end_timestamp必须选择有图的,没有图的计算不了target_position.")
    view_face_and_head_image(data_folder_path)

    # Stage4: Get rectified end timestamp's target position
    get_rectified_target_position(
        data_folder_path, human_parameters_path, show_image_monitor_index
    )

    # Stage5: Save two dataset(json file format)
    generate_dataset_json_file_head6d_eye3d(data_folder_path)


if __name__ == "__main__":
    main()
