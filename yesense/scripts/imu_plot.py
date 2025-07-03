#!/usr/bin/env python
import matplotlib.pyplot as plt
import numpy as np
import rospy
from matplotlib.animation import FuncAnimation
from sensor_msgs.msg import Imu
from tf.transformations import euler_from_quaternion


class IMUPlotter:
    def __init__(self):
        # 存储数据的列表
        self.timestamps = []
        self.rolls = []
        self.pitches = []
        self.yaws = []

        # ROS初始化
        rospy.init_node("imu_plotter", anonymous=True)
        self.sub = rospy.Subscriber("/imu/data", Imu, self.imu_callback)

        # 记录开始时间
        self.start_time = rospy.Time.now().to_sec()
        self.duration = 1000.0  # 采集10秒数据

        # 设置中文字体支持
        plt.rcParams["font.family"] = ["SimHei", "WenQuanYi Micro Hei", "Heiti TC"]
        plt.rcParams["axes.unicode_minus"] = False  # 解决负号显示问题

        # 创建图形
        self.fig, self.axes = plt.subplots(3, 1, figsize=(10, 8))
        self.fig.suptitle("IMU欧拉角随时间变化")

        # 初始化曲线
        (self.line_roll,) = self.axes[0].plot([], [], "r-", label="Roll")
        (self.line_pitch,) = self.axes[1].plot([], [], "g-", label="Pitch")
        (self.line_yaw,) = self.axes[2].plot([], [], "b-", label="Yaw")

        # 设置坐标轴标签和标题
        for ax, title in zip(self.axes, ["Roll (度)", "Pitch (度)", "Yaw (度)"]):
            ax.set_ylabel(title)
            ax.grid(True)
            ax.legend()

        self.axes[2].set_xlabel("时间 (秒)")

        # 设置动画
        self.ani = FuncAnimation(
            self.fig, self.update_plot, interval=100, cache_frame_data=False
        )

    def imu_callback(self, msg):
        # 计算相对时间（秒）
        current_time = msg.header.stamp.to_sec()
        relative_time = current_time - self.start_time

        # 提取四元数并转换为欧拉角（弧度）
        quaternion = (
            msg.orientation.x,
            msg.orientation.y,
            msg.orientation.z,
            msg.orientation.w,
        )
        print(quaternion)
        roll, pitch, yaw = euler_from_quaternion(quaternion, axes="rxyz")

        # 转换为角度
        roll_deg = np.degrees(roll)
        pitch_deg = np.degrees(pitch)
        yaw_deg = np.degrees(yaw)

        # 存储数据
        self.timestamps.append(relative_time)
        self.rolls.append(roll_deg)
        self.pitches.append(pitch_deg)
        self.yaws.append(yaw_deg)

        # 检查是否达到采集时间
        if relative_time >= self.duration:
            rospy.signal_shutdown("数据采集完成")

    def update_plot(self, frame):
        if not self.timestamps:
            return self.line_roll, self.line_pitch, self.line_yaw

        # 更新曲线数据
        self.line_roll.set_data(self.timestamps, self.rolls)
        self.line_pitch.set_data(self.timestamps, self.pitches)
        self.line_yaw.set_data(self.timestamps, self.yaws)

        # 自动调整坐标轴范围
        for ax, data in zip(self.axes, [self.rolls, self.pitches, self.yaws]):
            if data:
                ax.set_xlim(0, max(self.timestamps))
                ax.set_ylim(min(data) - 5, max(data) + 5)

        return self.line_roll, self.line_pitch, self.line_yaw

    def run(self):
        try:
            plt.tight_layout()
            plt.show()
            rospy.spin()
        except rospy.ROSInterruptException:
            pass
        finally:
            # 保存数据到文件
            if self.timestamps:
                data = np.column_stack(
                    (self.timestamps, self.rolls, self.pitches, self.yaws)
                )
                np.savetxt(
                    "imu_euler_data.csv",
                    data,
                    delimiter=",",
                    header="Time(s),Roll(deg),Pitch(deg),Yaw(deg)",
                    comments="",
                )
                print("数据已保存到 imu_euler_data.csv")

                # 绘制最终静态图
                self.fig, self.axes = plt.subplots(3, 1, figsize=(10, 8))
                self.fig.suptitle("IMU欧拉角随时间变化（最终结果）")

                self.axes[0].plot(self.timestamps, self.rolls, "r-")
                self.axes[1].plot(self.timestamps, self.pitches, "g-")
                self.axes[2].plot(self.timestamps, self.yaws, "b-")

                for ax, title, data in zip(
                    self.axes,
                    ["Roll (度)", "Pitch (度)", "Yaw (度)"],
                    [self.rolls, self.pitches, self.yaws],
                ):
                    ax.set_ylabel(title)
                    ax.grid(True)
                    ax.set_xlim(0, max(self.timestamps))
                    ax.set_ylim(min(data) - 5, max(data) + 5)

                self.axes[2].set_xlabel("时间 (秒)")
                plt.tight_layout()
                plt.savefig("imu_euler_plot.png")
                print("图表已保存到 imu_euler_plot.png")


if __name__ == "__main__":
    plotter = IMUPlotter()
    plotter.run()
