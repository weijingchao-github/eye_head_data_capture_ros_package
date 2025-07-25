import os
import sys

path = os.path.dirname(__file__)
sys.path.insert(0, path)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["GRPC_VERBOSITY"] = "ERROR"
os.environ["GLOG_minloglevel"] = "2"

import json
import math
import pickle

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import zhplot
from eye_gaze_direction_kalman_filter import eye_gaze_direction_kalman_filter
from get_eye_gaze_direction import GetEyeGazeDirection
from get_rectified_head_pose import get_current_rectified_head_pose
from matplotlib.widgets import Slider
from tf.transformations import euler_from_quaternion
from tqdm import tqdm


class HeadPoseDataProcess:
    def __init__(self, data_folder_path, cnt_abort):
        viz_flag = True
        self.cnt_abort = cnt_abort
        self.data_folder_path = data_folder_path
        rectified_and_interpolated_data = self.rectify_and_interpolate_raw_data()
        if viz_flag:
            self.show_quaternion_format_curve(rectified_and_interpolated_data)
            self.show_euler_angles_format_curve(rectified_and_interpolated_data)

    def rectify_and_interpolate_raw_data(self):
        imu_data_df = pd.read_csv(
            os.path.join(self.data_folder_path, "imu_data.csv"),
            dtype={"timestamp": str},
        )
        # 新.csv文件前10帧不要了，前10帧数据时间戳跳动比较大
        cnt_abort = self.cnt_abort
        # 判断有没有跳帧的两个阈值
        # 两帧差在 1.5 * 1/60 之内视为没有跳帧
        # 两帧差在 1.5 * 1/60 到 2.5 * 1/60 之间视为跳一帧，这时用前面的一帧数据和当前读到的这一帧数据插帧
        # 两帧差在 2.5 * 1/60 以上意味着跳了两帧以上，这时认为数据有问题，抛出错误
        time_threshold_1 = 1.5 * 1 / 60
        time_threshold_2 = 2.5 * 1 / 60
        initial_head_pose_quaternion = None
        last_row_data = None
        cnt = 0
        rectified_and_interpolated_data = []
        for row in imu_data_df.itertuples():
            if cnt == 0:
                initial_head_pose_quaternion = (row.x, row.y, row.z, row.w)
            if cnt < cnt_abort:
                cnt += 1
                continue
            current_head_pose_quaternion = (row.x, row.y, row.z, row.w)
            rectified_current_head_pose_quaternion_xyzw = (
                get_current_rectified_head_pose(
                    initial_head_pose_quaternion, current_head_pose_quaternion
                )
            )
            rectified_current_head_pose_quaternion_xyzw = np.array(
                rectified_current_head_pose_quaternion_xyzw
            )
            if last_row_data is not None:
                if (
                    float(row.timestamp) - float(last_row_data["timestamp"])
                ) < time_threshold_1:
                    pass
                elif (
                    time_threshold_1
                    <= (float(row.timestamp) - float(last_row_data["timestamp"]))
                    <= time_threshold_2
                ):
                    last_head_pose_quaternion = np.array(
                        [
                            last_row_data["x"],
                            last_row_data["y"],
                            last_row_data["z"],
                            last_row_data["w"],
                        ]
                    )
                    decimal_length = 7
                    interpolated_head_pose_timestamp = str(
                        round(
                            (float(last_row_data["timestamp"]) + float(row.timestamp))
                            / 2,
                            decimal_length,
                        )
                    )
                    # 补零操作：将timestamp小数部分补零到指定长度
                    if "." in interpolated_head_pose_timestamp:
                        integer_part, decimal_part = (
                            interpolated_head_pose_timestamp.split(".", 1)
                        )
                    else:
                        integer_part = interpolated_head_pose_timestamp
                        decimal_part = ""

                    if decimal_part:
                        # 补0到指定长度（不足则补0，超过则保留原长度，避免截断有效数据）
                        padded_decimal = decimal_part.ljust(decimal_length, "0")
                    else:
                        # 无小数部分时，添加补0后的小数（例如补6位则为"000000"）
                        padded_decimal = "0" * decimal_length
                    interpolated_head_pose_timestamp = (
                        f"{integer_part}.{padded_decimal}"
                    )
                    interpolated_head_pose_quaternion_xyzw = (
                        last_head_pose_quaternion
                        + rectified_current_head_pose_quaternion_xyzw
                    ) / 2
                    rectified_and_interpolated_data.append(
                        {
                            "timestamp": interpolated_head_pose_timestamp,
                            "x": interpolated_head_pose_quaternion_xyzw[0],
                            "y": interpolated_head_pose_quaternion_xyzw[1],
                            "z": interpolated_head_pose_quaternion_xyzw[2],
                            "w": interpolated_head_pose_quaternion_xyzw[3],
                        }
                    )
                elif (
                    float(row.timestamp) - float(last_row_data["timestamp"])
                ) > time_threshold_2:
                    raise Exception("Head Pose Raw Data Lost Too Many Frames")
            data = {
                "timestamp": row.timestamp,
                "x": rectified_current_head_pose_quaternion_xyzw[0],
                "y": rectified_current_head_pose_quaternion_xyzw[1],
                "z": rectified_current_head_pose_quaternion_xyzw[2],
                "w": rectified_current_head_pose_quaternion_xyzw[3],
            }
            rectified_and_interpolated_data.append(data)
            last_row_data = data
        return rectified_and_interpolated_data

    def show_quaternion_format_curve(self, quaternion_data):
        n_points = len(quaternion_data)
        # 提取数据
        indices = list(range(1, len(quaternion_data) + 1))
        x_values = [item["x"] for item in quaternion_data]
        y_values = [item["y"] for item in quaternion_data]
        z_values = [item["z"] for item in quaternion_data]
        w_values = [item["w"] for item in quaternion_data]

        # 初始显示的数据范围
        n_display = 60 * 10  # 每次显示的数据点数量
        start_idx = 0
        end_idx = min(n_display, n_points)

        # 创建画布和四个子图
        fig, axes = plt.subplots(4, 1, figsize=(12, 10), sharex=True)
        fig.subplots_adjust(bottom=0.15)  # 为滚动条腾出空间
        fig.suptitle("四元数随序号变化曲线 (带自适应Y轴)", fontsize=16)

        # 绘制初始数据并设置自适应Y轴
        lines = []
        for ax, values, color, title in zip(
            axes,
            [x_values, y_values, z_values, w_values],
            ["r-", "g-", "b-", "m-"],
            ["四元数 x 分量", "四元数 y 分量", "四元数 z 分量", "四元数 w 分量"],
        ):
            (line,) = ax.plot(
                indices[start_idx:end_idx],
                values[start_idx:end_idx],
                color,
                linewidth=1.5,
            )
            ax.set_ylabel(title.split()[-1], fontsize=10)
            ax.grid(True, linestyle="--", alpha=0.7)
            ax.set_title(title, fontsize=12)
            lines.append(line)

        # 设置x轴标签和刻度
        axes[-1].set_xlabel("数据点序号", fontsize=12)
        axes[-1].set_xlim(indices[start_idx], indices[end_idx - 1])

        # 创建滚动条
        ax_slider = plt.axes([0.15, 0.05, 0.7, 0.03])
        slider = Slider(
            ax=ax_slider,
            label="数据范围",
            valmin=0,
            valmax=n_points - n_display,
            valinit=start_idx,
            valstep=1,
        )

        # 更新函数（带自适应Y轴）
        def update(val):
            start_idx = int(val)
            end_idx = min(start_idx + n_display, n_points)

            # 更新线条数据并自适应Y轴
            for line, values, ax in zip(
                lines, [x_values, y_values, z_values, w_values], axes
            ):
                line.set_xdata(indices[start_idx:end_idx])
                line.set_ydata(values[start_idx:end_idx])

                # 计算当前显示数据的y轴范围，并增加10%的边距
                current_values = values[start_idx:end_idx]
                if len(current_values) > 0:
                    y_min, y_max = min(current_values), max(current_values)
                    margin = (y_max - y_min) * 0.1
                    ax.set_ylim(y_min - margin, y_max + margin)

            # 更新x轴范围
            axes[-1].set_xlim(indices[start_idx], indices[end_idx - 1])

            # 重绘
            fig.canvas.draw_idle()

        # 连接更新函数
        slider.on_changed(update)

        # 添加缩放功能说明
        plt.figtext(
            0.5,
            0.01,
            "使用鼠标滚轮缩放，拖动可平移。使用滚动条浏览大量数据。每个子图Y轴自动适应数据范围。",
            ha="center",
            fontsize=9,
            bbox={"facecolor": "w", "alpha": 0.5, "pad": 5},
        )

        # 显示图形
        plt.show()

    def show_euler_angles_format_curve(self, quaternion_data):
        euler_angles_data = []
        for data in quaternion_data:
            quternion_xyzw = (data["x"], data["y"], data["z"], data["w"])
            roll, pitch, yaw = euler_from_quaternion(quternion_xyzw, axes="rxyz")
            roll_degree = math.degrees(roll)
            pitch_degree = math.degrees(pitch)
            yaw_degree = math.degrees(yaw)
            euler_angles_data_one_piece = {
                "timestamp": data["timestamp"],
                "roll": roll_degree,
                "pitch": pitch_degree,
                "yaw": yaw_degree,
            }
            euler_angles_data.append(euler_angles_data_one_piece)
        n_points = len(euler_angles_data)
        # 提取数据
        indices = list(range(1, len(euler_angles_data) + 1))
        roll_values = [item["roll"] for item in euler_angles_data]
        pitch_values = [item["pitch"] for item in euler_angles_data]
        yaw_values = [item["yaw"] for item in euler_angles_data]

        # 初始显示的数据范围
        n_display = 60 * 10  # 每次显示的数据点数量
        start_idx = 0
        end_idx = min(n_display, n_points)

        # 创建画布和三个子图
        fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
        fig.subplots_adjust(bottom=0.15)  # 为滚动条腾出空间
        fig.suptitle("ZYX欧拉角随序号变化曲线 (带自适应Y轴)", fontsize=16)

        # 绘制初始数据并设置自适应Y轴
        lines = []
        for ax, values, color, title in zip(
            axes,
            [yaw_values, pitch_values, roll_values],
            ["r-", "g-", "b-"],
            ["欧拉角yaw值", "欧拉角pitch值", "欧拉角roll值"],
        ):
            (line,) = ax.plot(
                indices[start_idx:end_idx],
                values[start_idx:end_idx],
                color,
                linewidth=1.5,
            )
            ax.set_ylabel(title.split()[-1], fontsize=10)
            ax.grid(True, linestyle="--", alpha=0.7)
            ax.set_title(title, fontsize=12)
            lines.append(line)

        # 设置x轴标签和刻度
        axes[-1].set_xlabel("数据点序号", fontsize=12)
        axes[-1].set_xlim(indices[start_idx], indices[end_idx - 1])

        # 创建滚动条
        ax_slider = plt.axes([0.15, 0.05, 0.7, 0.03])
        slider = Slider(
            ax=ax_slider,
            label="数据范围",
            valmin=0,
            valmax=n_points - n_display,
            valinit=start_idx,
            valstep=1,
        )

        # 更新函数（带自适应Y轴）
        def update(val):
            start_idx = int(val)
            end_idx = min(start_idx + n_display, n_points)

            # 更新线条数据并自适应Y轴
            for line, values, ax in zip(
                lines, [yaw_values, pitch_values, roll_values], axes
            ):
                line.set_xdata(indices[start_idx:end_idx])
                line.set_ydata(values[start_idx:end_idx])

                # 计算当前显示数据的y轴范围，并增加10%的边距
                current_values = values[start_idx:end_idx]
                if len(current_values) > 0:
                    y_min, y_max = min(current_values), max(current_values)
                    margin = (y_max - y_min) * 0.1
                    ax.set_ylim(y_min - margin, y_max + margin)

            # 更新x轴范围
            axes[-1].set_xlim(indices[start_idx], indices[end_idx - 1])

            # 重绘
            fig.canvas.draw_idle()

        # 连接更新函数
        slider.on_changed(update)

        # 添加缩放功能说明
        plt.figtext(
            0.5,
            0.01,
            "使用鼠标滚轮缩放，拖动可平移。使用滚动条浏览大量数据。每个子图Y轴自动适应数据范围。",
            ha="center",
            fontsize=9,
            bbox={"facecolor": "w", "alpha": 0.5, "pad": 5},
        )

        # 显示图形
        plt.show()


class EyeGazeDirectionDataProcess:
    def __init__(self, data_folder_path, human_parameters, cnt_abort):
        viz_flag = True
        self.cnt_abort = cnt_abort
        self.data_folder_path = data_folder_path
        self.get_eye_gaze_direction_model = GetEyeGazeDirection(human_parameters)
        eye_gaze_direction_raw_data = self.get_eye_gaze_direction_raw_data()
        with open(
            os.path.join(
                os.path.dirname(__file__), "eye_gaze_direction_raw_data.pickle"
            ),
            "wb",
        ) as f:
            pickle.dump(eye_gaze_direction_raw_data, f)
        eye_gaze_direction_kalman_filtered_data = eye_gaze_direction_kalman_filter(
            eye_gaze_direction_raw_data
        )
        if viz_flag:
            # self.show_euler_angles_format_curve(eye_gaze_direction_raw_data)
            self.show_compared_euler_angles_format_curve(
                eye_gaze_direction_raw_data, eye_gaze_direction_kalman_filtered_data
            )

    def get_eye_gaze_direction_raw_data(self):
        imu_data_df = pd.read_csv(
            os.path.join(self.data_folder_path, "imu_data.csv"),
            dtype={"timestamp": str},
        )
        # 新.csv文件前10帧不要了，前10帧数据时间戳跳动比较大
        cnt_abort = self.cnt_abort
        # 判断有没有跳帧的两个阈值
        # 两帧差在 1.5 * 1/60 之内视为没有跳帧
        # 两帧差在 1.5 * 1/60 到 2.5 * 1/60 之间视为跳一帧，这时用前面的一帧数据和当前读到的这一帧数据插帧
        # 两帧差在 2.5 * 1/60 以上意味着跳了两帧以上，这时认为数据有问题，抛出错误
        time_threshold_1 = 1.5 * 1 / 60
        time_threshold_2 = 2.5 * 1 / 60
        last_row_data = None
        cnt = 0
        eye_gaze_direction_data = []

        # 计算总行数用于进度条
        total_rows = len(imu_data_df)
        processed_rows = 0

        # 创建进度条
        progress_bar = tqdm(
            total=total_rows - cnt_abort,
            desc="处理眼动数据",
            unit="行",
            bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]",
        )

        for row in imu_data_df.itertuples():
            if cnt < cnt_abort:
                cnt += 1
                continue

            # 更新进度条
            processed_rows += 1
            progress_bar.update(1)

            current_timestamp = row.timestamp
            face_image = cv2.imread(
                os.path.join(
                    self.data_folder_path, f"face_image/{current_timestamp}.jpg"
                )
            )
            eye_gaze_yaw_radians, eye_gaze_pitch_radians = (
                self.get_eye_gaze_direction_model.pipeline(face_image)
            )
            eye_gaze_yaw_angle = math.degrees(eye_gaze_yaw_radians)
            eye_gaze_pitch_angle = math.degrees(eye_gaze_pitch_radians)
            current_eye_gaze_direction = np.array(
                [eye_gaze_yaw_angle, eye_gaze_pitch_angle]
            )
            if last_row_data is not None:
                if (
                    float(row.timestamp) - float(last_row_data["timestamp"])
                ) < time_threshold_1:
                    pass
                elif (
                    time_threshold_1
                    <= (float(row.timestamp) - float(last_row_data["timestamp"]))
                    <= time_threshold_2
                ):
                    last_eye_gaze_direction = np.array(
                        last_row_data["yaw"], last_row_data["pitch"]
                    )
                    decimal_length = 7
                    interpolated_eye_gaze_timestamp = str(
                        round(
                            (float(last_row_data["timestamp"]) + float(row.timestamp))
                            / 2,
                            decimal_length,
                        )
                    )
                    # 补零操作：将timestamp小数部分补零到指定长度
                    if "." in interpolated_eye_gaze_timestamp:
                        integer_part, decimal_part = (
                            interpolated_eye_gaze_timestamp.split(".", 1)
                        )
                    else:
                        integer_part = interpolated_eye_gaze_timestamp
                        decimal_part = ""

                    if decimal_part:
                        # 补0到指定长度（不足则补0，超过则保留原长度，避免截断有效数据）
                        padded_decimal = decimal_part.ljust(decimal_length, "0")
                    else:
                        # 无小数部分时，添加补0后的小数（例如补6位则为"000000"）
                        padded_decimal = "0" * decimal_length
                    interpolated_eye_gaze_timestamp = f"{integer_part}.{padded_decimal}"
                    interpolated_eye_gaze_direction = (
                        last_eye_gaze_direction + current_eye_gaze_direction
                    ) / 2
                    eye_gaze_direction_data.append(
                        {
                            "timestamp": interpolated_eye_gaze_timestamp,
                            "yaw": interpolated_eye_gaze_direction[0],
                            "pitch": interpolated_eye_gaze_direction[1],
                        }
                    )
                elif (
                    float(row.timestamp) - float(last_row_data["timestamp"])
                ) > time_threshold_2:
                    progress_bar.close()  # 关闭进度条后再抛出异常
                    raise Exception("Eye Gaze Direction Raw Data Lost Too Many Frames")
            data = {
                "timestamp": row.timestamp,
                "yaw": current_eye_gaze_direction[0],
                "pitch": current_eye_gaze_direction[1],
            }
            eye_gaze_direction_data.append(data)
            last_row_data = data

        # 关闭进度条
        progress_bar.close()

        return eye_gaze_direction_data

    def show_euler_angles_format_curve(self, euler_angles_data):
        n_points = len(euler_angles_data)
        # 提取数据
        indices = list(range(1, len(euler_angles_data) + 1))
        pitch_values = [item["pitch"] for item in euler_angles_data]
        yaw_values = [item["yaw"] for item in euler_angles_data]

        # 初始显示的数据范围
        n_display = 60 * 10  # 每次显示的数据点数量
        start_idx = 0
        end_idx = min(n_display, n_points)

        # 创建画布和两个子图
        fig, axes = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
        fig.subplots_adjust(bottom=0.15)  # 为滚动条腾出空间
        fig.suptitle("Yaw, Pitch角随序号变化曲线 (带自适应Y轴)", fontsize=16)

        # 绘制初始数据并设置自适应Y轴
        lines = []
        for ax, values, color, title in zip(
            axes,
            [yaw_values, pitch_values],
            ["r-", "g-"],
            ["欧拉角yaw值", "欧拉角pitch值"],
        ):
            (line,) = ax.plot(
                indices[start_idx:end_idx],
                values[start_idx:end_idx],
                color,
                linewidth=1.5,
            )
            ax.set_ylabel(title.split()[-1], fontsize=10)
            ax.grid(True, linestyle="--", alpha=0.7)
            ax.set_title(title, fontsize=12)
            lines.append(line)

        # 设置x轴标签和刻度
        axes[-1].set_xlabel("数据点序号", fontsize=12)
        axes[-1].set_xlim(indices[start_idx], indices[end_idx - 1])

        # 创建滚动条
        ax_slider = plt.axes([0.15, 0.05, 0.7, 0.03])
        slider = Slider(
            ax=ax_slider,
            label="数据范围",
            valmin=0,
            valmax=n_points - n_display,
            valinit=start_idx,
            valstep=1,
        )

        # 更新函数（带自适应Y轴）
        def update(val):
            start_idx = int(val)
            end_idx = min(start_idx + n_display, n_points)

            # 更新线条数据并自适应Y轴
            for line, values, ax in zip(lines, [yaw_values, pitch_values], axes):
                line.set_xdata(indices[start_idx:end_idx])
                line.set_ydata(values[start_idx:end_idx])

                # 计算当前显示数据的y轴范围，并增加10%的边距
                current_values = values[start_idx:end_idx]
                if len(current_values) > 0:
                    y_min, y_max = min(current_values), max(current_values)
                    margin = (y_max - y_min) * 0.1
                    ax.set_ylim(y_min - margin, y_max + margin)

            # 更新x轴范围
            axes[-1].set_xlim(indices[start_idx], indices[end_idx - 1])

            # 重绘
            fig.canvas.draw_idle()

        # 连接更新函数
        slider.on_changed(update)

        # 添加缩放功能说明
        plt.figtext(
            0.5,
            0.01,
            "使用鼠标滚轮缩放，拖动可平移。使用滚动条浏览大量数据。每个子图Y轴自动适应数据范围。",
            ha="center",
            fontsize=9,
            bbox={"facecolor": "w", "alpha": 0.5, "pad": 5},
        )

        # 显示图形
        plt.show()

    def show_compared_euler_angles_format_curve(
        self, euler_angles_data, kf_euler_angles_data
    ):
        n_points = len(euler_angles_data)
        # 确保两个数据集长度一致
        assert len(euler_angles_data) == len(
            kf_euler_angles_data
        ), "两个数据集长度不一致"

        # 提取数据
        indices = list(range(1, n_points + 1))
        # 原始数据
        pitch_values = [item["pitch"] for item in euler_angles_data]
        yaw_values = [item["yaw"] for item in euler_angles_data]
        # 卡尔曼滤波后的数据
        kf_pitch_values = [item["pitch"] for item in kf_euler_angles_data]
        kf_yaw_values = [item["yaw"] for item in kf_euler_angles_data]

        # 初始显示的数据范围
        n_display = 60 * 10  # 每次显示的数据点数量
        start_idx = 0
        end_idx = min(n_display, n_points)

        # 创建画布和两个子图
        fig, axes = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
        fig.subplots_adjust(bottom=0.15)  # 为滚动条腾出空间
        fig.suptitle("Yaw, Pitch角原始数据 vs 卡尔曼滤波数据对比", fontsize=16)

        # 绘制初始数据并设置自适应Y轴
        lines = []
        for ax, values, kf_values, title in zip(
            axes,
            [yaw_values, pitch_values],
            [kf_yaw_values, kf_pitch_values],
            ["Yaw角对比", "Pitch角对比"],
        ):
            # 原始数据（蓝色实线）
            (line_raw,) = ax.plot(
                indices[start_idx:end_idx],
                values[start_idx:end_idx],
                "b-",
                linewidth=1.0,
                alpha=0.7,
                label="原始数据",
            )
            # 卡尔曼滤波数据（红色实线）
            (line_kf,) = ax.plot(
                indices[start_idx:end_idx],
                kf_values[start_idx:end_idx],
                "r-",
                linewidth=1.5,
                label="卡尔曼滤波",
            )
            ax.set_ylabel(title.split()[0], fontsize=10)
            ax.grid(True, linestyle="--", alpha=0.7)
            ax.set_title(title, fontsize=12)
            ax.legend(loc="upper right")  # 添加图例
            lines.append((line_raw, line_kf))  # 保存两条线的引用

        # 设置x轴标签和刻度
        axes[-1].set_xlabel("数据点序号", fontsize=12)
        axes[-1].set_xlim(indices[start_idx], indices[end_idx - 1])

        # 创建滚动条
        ax_slider = plt.axes([0.15, 0.05, 0.7, 0.03])
        slider = Slider(
            ax=ax_slider,
            label="数据范围",
            valmin=0,
            valmax=n_points - n_display,
            valinit=start_idx,
            valstep=1,
        )

        # 更新函数（带自适应Y轴）
        def update(val):
            start_idx = int(val)
            end_idx = min(start_idx + n_display, n_points)

            # 更新线条数据并自适应Y轴
            for (line_raw, line_kf), values, kf_values, ax in zip(
                lines,
                [yaw_values, pitch_values],
                [kf_yaw_values, kf_pitch_values],
                axes,
            ):
                line_raw.set_xdata(indices[start_idx:end_idx])
                line_raw.set_ydata(values[start_idx:end_idx])
                line_kf.set_xdata(indices[start_idx:end_idx])
                line_kf.set_ydata(kf_values[start_idx:end_idx])

                # 计算当前显示数据的y轴范围（同时考虑原始数据和滤波数据）
                current_values = (
                    values[start_idx:end_idx] + kf_values[start_idx:end_idx]
                )
                if len(current_values) > 0:
                    y_min, y_max = min(current_values), max(current_values)
                    margin = (y_max - y_min) * 0.1
                    ax.set_ylim(y_min - margin, y_max + margin)

            # 更新x轴范围
            axes[-1].set_xlim(indices[start_idx], indices[end_idx - 1])

            # 重绘
            fig.canvas.draw_idle()

        # 连接更新函数
        slider.on_changed(update)

        # 添加缩放功能说明
        plt.figtext(
            0.5,
            0.01,
            "使用鼠标滚轮缩放，拖动可平移。使用滚动条浏览大量数据。蓝色=原始数据，红色=卡尔曼滤波。",
            ha="center",
            fontsize=9,
            bbox={"facecolor": "w", "alpha": 0.5, "pad": 5},
        )

        # 显示图形
        plt.show()


def main():
    data_folder_path = "/home/wjc/Storage/humanoid_head/eye_head_data_capture/src/yesense/scripts/captured_data_3/20250707_105422"
    cnt_abort = 15  # 这个根据每个data_folder的情况赋值，一个是时间戳数据从哪一帧开始连续平稳，另一个是期望刚开始的时候眼颈姿态平稳，没有处在变化中，不然卡尔曼滤波对眼球转动角速度的初值估计有误，影响滤波效果
    human_parameters_path = os.path.join(
        os.path.dirname(__file__), "human_parameters.json"
    )
    with open(human_parameters_path, "r", encoding="utf-8") as f:
        human_parameters = json.load(f)
    # head pose data process
    # HeadPoseDataProcess(data_folder_path, cnt_abort)
    # eye gaze direction data process
    EyeGazeDirectionDataProcess(data_folder_path, human_parameters, cnt_abort)


if __name__ == "__main__":
    main()
