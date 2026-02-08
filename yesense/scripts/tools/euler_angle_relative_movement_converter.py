import json
import math
import os


def rotation_matrix_to_zy_euler(rot_col):
    """
    将旋转矩阵的第一列转换为ZY欧拉角(Yaw-Pitch)
    假设旋转顺序为: 先绕Z轴旋转(Yaw),再绕Y轴旋转(Pitch)
    """
    # 从旋转矩阵第一列构造简化的旋转矩阵（假设无Roll旋转）
    r11, r21, r31 = rot_col

    # 计算Yaw角 (绕Z轴)
    yaw = math.atan2(r21, r11)

    # 计算Pitch角 (绕Y轴)
    pitch = math.atan2(-r31, math.sqrt(r11**2 + r21**2))

    return [yaw, pitch]


def rotation_matrix_to_zyx_euler(rot_cols):
    """
    将旋转矩阵前两列转换为ZYX欧拉角(Tait-Bryan角,先绕Z轴、再绕Y轴、最后绕X轴)
    旋转矩阵定义:R = R_Z(yaw) * R_Y(pitch) * R_X(roll)（右乘顺序）
    """
    # 旋转矩阵前两列元素
    r11, r21, r31, r12, r22, r32 = rot_cols

    # 计算第三列（前两列的叉乘，满足旋转矩阵列正交性）
    r13 = r21 * r32 - r31 * r22  # 第三列第1元素
    r23 = r31 * r12 - r11 * r32  # 第三列第2元素
    r33 = r11 * r22 - r21 * r12  # 第三列第3元素

    yaw = math.atan2(r21, r11)
    pitch = math.asin(-r31)
    roll = math.atan2(r32, r33)

    return [yaw, pitch, roll]


def process_json_file(input_path, output_path):
    """处理单个JSON文件，转换旋转矩阵为欧拉角的相对量"""
    with open(input_path, "r") as f:
        data = json.load(f)

    converted_data = {}
    for timestamp, entry in data.items():
        converted_entry = {}

        # 转换当前眼球姿态 (ZY欧拉角)
        converted_entry["current_eye_pose"] = rotation_matrix_to_zy_euler(
            entry["current_eye_pose"]
        )

        # 转换当前头部姿态 (ZYX欧拉角)
        converted_entry["current_head_pose"] = rotation_matrix_to_zyx_euler(
            entry["current_head_pose"]
        )

        # 转换目标眼球姿态 (ZY欧拉角),求眼球相对运动量
        target_eye_pose = rotation_matrix_to_zy_euler(entry["target_eye_pose"])
        converted_entry["target_delta_eye_movement"] = [
            target_eye_pose[0] - converted_entry["current_eye_pose"][0],
            target_eye_pose[1] - converted_entry["current_eye_pose"][1],
        ]

        # 转换目标头部姿态 (ZYX欧拉角),求头部相对运动量
        target_head_pose = rotation_matrix_to_zyx_euler(
            entry["target_head_pose"]
        )
        converted_entry["target_delta_head_movement"] = [
            target_head_pose[0] - converted_entry["current_head_pose"][0],
            target_head_pose[1] - converted_entry["current_head_pose"][1],
            target_head_pose[2] - converted_entry["current_head_pose"][2],
        ]

        # 保持目标位置不变
        converted_entry["target_position"] = entry["target_position"]

        converted_data[timestamp] = converted_entry

    # 保存转换后的数据
    with open(output_path, "w") as f:
        json.dump(converted_data, f, indent=2)


def main():
    # 创建输出文件夹
    output_dir = "euler_angle_relative_movement"
    os.makedirs(output_dir, exist_ok=True)

    # 获取当前目录下所有JSON文件
    current_dir = os.getcwd()
    for filename in os.listdir(current_dir):
        if filename.endswith(".json"):
            input_path = os.path.join(current_dir, filename)
            output_path = os.path.join(output_dir, filename)
            print(f"处理文件: {filename}")
            process_json_file(input_path, output_path)

    print("所有文件处理完成！")


if __name__ == "__main__":
    main()
