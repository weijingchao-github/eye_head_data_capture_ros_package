import csv
import os


def downsample_csv_file(input_file, output_file, interval=2):
    """
    读取CSV文件并间隔指定行数保存数据

    参数:
        input_file: 输入CSV文件路径
        output_file: 输出CSV文件路径
        interval: 间隔行数，默认为2（即保留1行，跳过1行）
    """
    with open(input_file, "r", newline="") as infile, open(
        output_file, "w", newline=""
    ) as outfile:

        reader = csv.reader(infile)
        writer = csv.writer(outfile)

        # 读取并写入表头
        header = next(reader)
        writer.writerow(header)

        # 间隔一行保存数据（保留第1、3、5...行）
        for index, row in enumerate(reader):
            # 只保存索引为偶数的行（0, 2, 4...）
            if index % interval == 0:
                writer.writerow(row)


def downsample_eye_and_head_pose_sequence(data_folder_path):
    input_csv = os.path.join(data_folder_path, "eye_head_pose_sequence_60hz.csv")
    output_csv = os.path.join(data_folder_path, "eye_head_pose_sequence_30hz.csv")
    downsample_csv_file(input_csv, output_csv)
    print("Stage 2: Downsample eye and head pose sequence from 60Hz to 30Hz Finished.")


if __name__ == "__main__":
    # 输入和输出文件路径
    input_csv = "/home/wjc/Storage/humanoid_head/eye_head_data_capture/src/yesense/scripts/captured_data_3/20250707_105422/eye_head_pose_sequence_60hz.csv"  # 替换为你的输入文件路径
    output_csv = "/home/wjc/Storage/humanoid_head/eye_head_data_capture/src/yesense/scripts/captured_data_3/20250707_105422/eye_head_pose_sequence_30hz.csv"  # 替换为你的输出文件路径

    # 调用函数，间隔1行保存（保留1行，跳过1行）
    downsample_csv_file(input_csv, output_csv)
    print(f"已完成处理，结果保存至 {output_csv}")
