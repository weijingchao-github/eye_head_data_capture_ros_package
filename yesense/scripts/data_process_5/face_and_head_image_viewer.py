import os

import cv2
import numpy as np
import pandas as pd


def get_image_sorted_timestamps(folder_path):
    """获取文件夹中所有图片文件的时间戳文件名，并按时间戳排序"""
    if not os.path.exists(folder_path):
        print(f"错误：文件夹 {folder_path} 不存在！")
        return []

    files = os.listdir(folder_path)

    # 过滤出图片文件（常见图片格式）
    image_extensions = [".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"]
    image_files = [
        os.path.splitext(f)[0]
        for f in files
        if os.path.splitext(f)[1].lower() in image_extensions
    ]

    # 按时间戳（文件名）排序
    image_files.sort()
    return image_files


def main(data_folder_path):
    face_image_folder = os.path.join(data_folder_path, "face_image")
    head_color_image_folder = os.path.join(data_folder_path, "head_color_image")
    eye_head_pose_sequence_csv_file_path = os.path.join(
        data_folder_path, "eye_head_pose_sequence_30hz.csv"
    )
    eye_head_pose_sequence_df = pd.read_csv(
        eye_head_pose_sequence_csv_file_path, dtype={"timestamp": str}
    )
    image_sorted_timestamps = get_image_sorted_timestamps(face_image_folder)
    print("面部摄像头与头顶摄像头视角图片 (按A键切换上一张,按D键切换下一张,ESC退出)")
    # # 创建显示窗口
    # window_name = (
    #     "面部摄像头与头顶摄像头视角图片 (按A键切换上一张,按D键切换下一张,ESC退出)"
    # )
    # cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    current_index = 0
    show_record_direction = "forward"
    df_length = len(eye_head_pose_sequence_df)
    print(f"共 {df_length} 条记录")
    print("操作提示: 按A键切换到上一张图片,按D键切换到下一张图片,按ESC键退出程序")
    # 第一条记录一定带图片，最后一条记录不一定带图片
    while True:
        if current_index == 0:
            print("当前显示第一条记录:")
        if current_index == df_length - 1:
            print("当前显示最后一条记录:")
        if current_index == df_length:
            print("最后一条记录是插值之后的结果,不带图片,回退到上一张图片")
            current_index -= 2
            while (
                eye_head_pose_sequence_df.iloc[current_index]["timestamp"]
                not in image_sorted_timestamps
            ):
                current_index -= 1
            continue
        # 获取当前记录
        current_eye_head_pose_record = eye_head_pose_sequence_df.iloc[current_index]
        current_timestamp = current_eye_head_pose_record["timestamp"]
        timestamp = current_timestamp
        x = round(current_eye_head_pose_record["x"], 3)
        y = round(current_eye_head_pose_record["y"], 3)
        z = round(current_eye_head_pose_record["z"], 3)
        w = round(current_eye_head_pose_record["w"], 3)
        yaw = round(current_eye_head_pose_record["yaw"], 1)
        pitch = round(current_eye_head_pose_record["pitch"], 1)
        print(
            f"index: {current_index}, timestamp: {timestamp}, x: {x:.3f}, y: {y:.3f}, z: {z:.3f}, w: {w:.3f}, yaw: {yaw:.1f}, pitch: {pitch:.1f}"
        )

        if current_timestamp not in image_sorted_timestamps:
            if show_record_direction == "forward":
                current_index += 1
            elif show_record_direction == "backward":
                current_index -= 1
            continue

        # 读取两张图片
        face_image_path = os.path.join(face_image_folder, current_timestamp + ".jpg")
        head_color_image_path = os.path.join(
            head_color_image_folder, current_timestamp + ".jpg"
        )

        face_image = cv2.imread(face_image_path)
        head_color_image = cv2.imread(head_color_image_path)

        # 将两张图片并排拼接
        combined_img = np.hstack((face_image, head_color_image))

        # 放大两倍显示（使用INTER_CUBIC插值方法保持清晰度）
        scale_factor = 0.7
        new_width = int(combined_img.shape[1] * scale_factor)
        new_height = int(combined_img.shape[0] * scale_factor)
        combined_img = cv2.resize(
            combined_img, (new_width, new_height), interpolation=cv2.INTER_CUBIC
        )

        # 在图片上添加文件名（时间戳）- 字体也相应放大
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 1.2  # 字体大小随图片放大而调整
        font_color = (0, 255, 0)  # 绿色
        thickness = 2  # 字体粗细随图片放大而调整
        cv2.putText(
            combined_img,
            current_timestamp,
            (30, 60),
            font,
            font_scale,
            font_color,
            thickness,
            cv2.LINE_AA,
        )

        # 显示拼接后的图片
        cv2.imshow("face_and_head_color_image", combined_img)

        # 等待键盘事件
        key = cv2.waitKey(0)

        # 按D键（小写d:100 或大写D:68）切换到下一张
        if key in [100, 68]:  # D键的ASCII码
            if current_index == df_length - 1:
                print("已经是最后一张图片！")
                # print("已经是最后一张图片，回退到上一张图片")
                show_record_direction = "backward"
            else:
                current_index += 1
                show_record_direction = "forward"

        # 按A键（小写a:97 或大写A:65）切换到上一张
        elif key in [97, 65]:  # A键的ASCII码
            if current_index == 0:
                print("已经是第一张图片！")
                # print("已经是第一张图片，回退到第二张图片")
                show_record_direction = "forward"
            else:
                current_index -= 1
                show_record_direction = "backward"
        # 按ESC键退出
        elif key == 27:
            print("程序已退出")
            break

    # 关闭窗口
    cv2.destroyAllWindows()


def view_face_and_head_image(data_folder_path):
    main(data_folder_path)
    print("Stage3: Find Gaze Shift Start and End Timestamp Finished.")


if __name__ == "__main__":
    data_folder_path = "/home/wjc/Storage/humanoid_head/eye_head_data_capture/src/yesense/scripts/captured_data_3/20250707_105422"
    main(data_folder_path)
