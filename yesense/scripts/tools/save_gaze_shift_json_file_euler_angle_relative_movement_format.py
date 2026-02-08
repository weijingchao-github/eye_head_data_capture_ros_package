import csv
import json
import math
import os

import numpy as np
from scipy.spatial.transform import Rotation as R


def read_gaze_shift_record_file(file_path):
    """读取gaze_shift_start_and_end_with_target_position.csv文件"""
    gaze_shift_records = []
    with open(file_path, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        # 检查表头是否完整
        required_headers = {
            "start_timestamp",
            "end_timestamp",
            "target_position_x",
            "target_position_y",
            "target_position_z",
        }
        if not required_headers.issubset(reader.fieldnames):
            missing = required_headers - set(reader.fieldnames)
            raise ValueError(f"gaze_shift_record文件缺少必要表头: {missing}")

        for row_num, row in enumerate(reader, 2):  # 行号从2开始（表头是1）
            try:
                gaze_shift_records.append(
                    {
                        "start_ts": row["start_timestamp"],
                        "end_ts": row["end_timestamp"],
                        "target_x": float(row["target_position_x"]),
                        "target_y": float(row["target_position_y"]),
                        "target_z": float(row["target_position_z"]),
                    }
                )
            except (ValueError, KeyError) as e:
                print(
                    f"警告: gaze_shift_record文件第{row_num}行数据格式错误，已跳过 - {e}"
                )
    return gaze_shift_records


def read_eye_head_pose_sequence_file(file_path: str):
    """
    读取eye_head_pose_sequence_30hz.csv文件
    返回：(时间戳到数据的映射, 时间戳列表)
    """
    timestamp_map = {}
    timestamp_list = []
    with open(file_path, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        # 检查表头是否完整
        required_headers = {"timestamp", "x", "y", "z", "w", "yaw", "pitch"}
        if not required_headers.issubset(reader.fieldnames):
            missing = required_headers - set(reader.fieldnames)
            raise ValueError(f"eye_head_pose_sequence文件缺少必要表头: {missing}")

        for row_num, row in enumerate(reader, 2):  # 行号从2开始（表头是1）
            try:
                ts = row["timestamp"]
                # 存储x,y,z,w,yaw,pitch的列表
                data = [
                    float(row["x"]),
                    float(row["y"]),
                    float(row["z"]),
                    float(row["w"]),
                    float(row["yaw"]),
                    float(row["pitch"]),
                ]
                timestamp_map[ts] = data
                timestamp_list.append(ts)
            except (ValueError, KeyError) as e:
                print(
                    f"警告: eye_head_pose_sequence文件第{row_num}行数据格式错误，已跳过 - {e}"
                )

    return timestamp_map, timestamp_list


def rotation_matrix_to_zyx_euler(rotation_matrix):
    """
    将旋转矩阵前两列转换为ZYX欧拉角(Tait-Bryan角,先绕Z轴、再绕Y轴、最后绕X轴)
    旋转矩阵定义:R = R_X(roll) * R_Y(pitch) * R_Z(yaw)（右乘顺序）
    """

    r21 = rotation_matrix[1][0]
    r11 = rotation_matrix[0][0]
    r31 = rotation_matrix[2][0]
    r32 = rotation_matrix[2][1]
    r33 = rotation_matrix[2][2]

    yaw = math.atan2(r21, r11)
    pitch = math.asin(-r31)
    roll = math.atan2(r32, r33)

    return [yaw, pitch, roll]


def tf_head_pose_quaternion_to_zyx_euler_angle(quaternion_xyzw):
    rotation_matrix = R.from_quat(quaternion_xyzw).as_matrix()
    euler_angle = rotation_matrix_to_zyx_euler(rotation_matrix)
    return euler_angle


def save_json_file(input_path, output_path):
    gaze_shift_record_file = os.path.join(
        input_path, "gaze_shift_start_and_end_with_target_position.csv"
    )
    eye_head_pose_sequence_file = os.path.join(
        input_path, "eye_head_pose_sequence_30hz.csv"
    )
    # 1. 读取输入文件
    # print("开始读取文件...")
    gaze_shift_records = read_gaze_shift_record_file(gaze_shift_record_file)
    eye_head_pose_sequence_timestamp_map, eye_head_pose_sequence_timestamp_list = (
        read_eye_head_pose_sequence_file(eye_head_pose_sequence_file)
    )
    # 2. 初始化结果字典
    start_and_end_eye_head_pose_and_target_position_result = {}
    # 3. 处理每条gaze shift记录
    # print("开始处理数据...")
    for idx, record in enumerate(gaze_shift_records, 2):  # 行号从2开始（表头是1）
        start_ts = record["start_ts"]
        end_ts = record["end_ts"]
        start_ts_ = start_ts.replace(".", "_")
        end_ts_ = end_ts.replace(".", "_")
        key = f"{start_ts_}_{end_ts_}"  # 生成key

        # 3.1 提取目标位置
        target_position = [
            record["target_x"],
            record["target_y"],
            record["target_z"],
        ]

        # 3.2 查找start_timestamp和end_timestamp对应的眼颈姿态数据
        start_eye_head_pose_data = eye_head_pose_sequence_timestamp_map.get(start_ts)
        end_eye_head_pose_data = eye_head_pose_sequence_timestamp_map.get(end_ts)
        # 处理缺失的时间戳
        if not start_eye_head_pose_data:
            print(
                f"警告: gaze_shift记录第 {idx} 行的start_timestamp {start_ts} 在眼颈姿态序列数据中未找到"
            )
        if not end_eye_head_pose_data:
            print(
                f"警告: gaze_shift记录第 {idx} 行的end_timestamp {end_ts} 在眼颈姿态序列数据中未找到"
            )

        # 3.3 构建start_and_end_eye_head_pose_and_target_position_result
        current_eye_pose = [
            math.radians(start_eye_head_pose_data[4]),
            math.radians(start_eye_head_pose_data[5]),
        ]
        current_head_pose = tf_head_pose_quaternion_to_zyx_euler_angle(
            quaternion_xyzw=[
                start_eye_head_pose_data[0],
                start_eye_head_pose_data[1],
                start_eye_head_pose_data[2],
                start_eye_head_pose_data[3],
            ]
        )
        target_eye_pose = [
            math.radians(end_eye_head_pose_data[4]),
            math.radians(end_eye_head_pose_data[5]),
        ]
        target_head_pose = tf_head_pose_quaternion_to_zyx_euler_angle(
            quaternion_xyzw=[
                end_eye_head_pose_data[0],
                end_eye_head_pose_data[1],
                end_eye_head_pose_data[2],
                end_eye_head_pose_data[3],
            ]
        )
        target_delta_eye_movement = np.array(target_eye_pose) - np.array(
            current_eye_pose
        )
        target_delta_head_movement = np.array(target_head_pose) - np.array(
            current_head_pose
        )
        target_delta_eye_movement = target_delta_eye_movement.tolist()
        target_delta_head_movement = target_delta_head_movement.tolist()
        start_and_end_eye_head_pose_and_target_position_result[key] = {
            "current_eye_pose": current_eye_pose,
            "current_head_pose": current_head_pose,
            "target_delta_eye_movement": target_delta_eye_movement,
            "target_delta_head_movement": target_delta_head_movement,
            "target_position": target_position,
        }

    # 4. 写入JSON文件
    # print("开始写入JSON文件...")
    with open(
        output_path,
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            start_and_end_eye_head_pose_and_target_position_result,
            f,
            ensure_ascii=False,
            indent=2,
        )

    print("处理完成！")


def main():
    # 创建输出文件夹
    output_dir = os.path.join(
        os.path.dirname(__file__), "gaze_shift_euler_angle_relative_movement"
    )
    os.makedirs(output_dir, exist_ok=True)

    current_dir = os.getcwd()
    for folder_name in os.listdir(current_dir):
        if folder_name.startswith("20"):
            input_path = os.path.join(current_dir, folder_name)
            output_path = os.path.join(output_dir, folder_name + ".json")
            print(f"处理文件夹: {folder_name}")
            save_json_file(input_path, output_path)

    print("所有文件夹处理完成！")


if __name__ == "__main__":
    main()
