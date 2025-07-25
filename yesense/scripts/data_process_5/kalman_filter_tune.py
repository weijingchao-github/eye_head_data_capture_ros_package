import pickle

import matplotlib.pyplot as plt
import zhplot
from eye_gaze_direction_kalman_filter import eye_gaze_direction_kalman_filter
from matplotlib.widgets import Slider
from tqdm import tqdm


def show_compared_euler_angles_format_curve(euler_angles_data, kf_euler_angles_data):
    n_points = len(euler_angles_data)
    # 确保两个数据集长度一致
    assert len(euler_angles_data) == len(kf_euler_angles_data), "两个数据集长度不一致"

    # 提取数据
    indices = list(range(1, n_points + 1))
    # 原始数据
    pitch_values = [item["pitch"] for item in euler_angles_data]
    yaw_values = [item["yaw"] for item in euler_angles_data]
    # 卡尔曼滤波后的数据
    kf_pitch_values = [item["pitch"] for item in kf_euler_angles_data]
    kf_yaw_values = [item["yaw"] for item in kf_euler_angles_data]

    # 初始显示的数据范围
    n_display = 60 * 2  # 每次显示的数据点数量
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
            current_values = values[start_idx:end_idx] + kf_values[start_idx:end_idx]
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
    with open("eye_gaze_direction_raw_data.pickle", "rb") as f:
        eye_gaze_direction_raw_data = pickle.load(f)
    eye_gaze_direction_kalman_filtered_data = eye_gaze_direction_kalman_filter(
        eye_gaze_direction_raw_data
    )
    show_compared_euler_angles_format_curve(
        eye_gaze_direction_raw_data, eye_gaze_direction_kalman_filtered_data
    )


if __name__ == "__main__":
    main()
