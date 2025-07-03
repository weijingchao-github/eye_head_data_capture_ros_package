import os
import time
from datetime import datetime

import cv2
import numpy as np
import pandas as pd
import rospy
from cv_bridge import CvBridge
from message_filters import ApproximateTimeSynchronizer, Subscriber
from sensor_msgs.msg import Image, Imu
from tf.transformations import euler_from_quaternion


class DataCollector:
    def __init__(self):
        # 初始化ROS节点
        rospy.init_node("sensor_data_collector", anonymous=True)
        time.sleep(3)
        # 参数设置
        self.duration = 60  # 采集时长（秒）
        self.image_topic = "/camera/color/image_raw"  # 摄像头话题
        self.imu_topic = "/imu/data"  # IMU话题
        save_folder = "captured_data"
        self.save_dir = os.path.join(
            os.path.join(os.path.dirname(__file__), save_folder),
            datetime.now().strftime("%Y%m%d_%H%M%S"),
        )  # 保存目录
        self.queue_size = 20  # 同步队列大小
        self.slop = 1 / 200 / 2  # 时间容差（秒）

        # 创建保存目录
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)
            os.makedirs(os.path.join(self.save_dir, "images"))

        # 数据存储
        self.bridge = CvBridge()
        self.synced_image_data = []
        self.synced_imu_data = []

        # 创建消息过滤器订阅者
        self.image_sub = Subscriber(self.image_topic, Image)
        self.imu_sub = Subscriber(self.imu_topic, Imu)

        # 设置近似时间同步器
        self.ts = ApproximateTimeSynchronizer(
            [self.image_sub, self.imu_sub], queue_size=self.queue_size, slop=self.slop
        )
        self.ts.registerCallback(self.sync_callback)

        # 记录开始时间
        self.start_time = rospy.Time.now().to_sec()
        rospy.loginfo(f"开始数据采集，将持续 {self.duration} 秒")
        rospy.loginfo(f"数据将保存到: {self.save_dir}")

    def sync_callback(self, image_msg, imu_msg):
        # 计算相对时间
        current_time = rospy.Time.now().to_sec()
        relative_time = current_time - self.start_time

        # 检查是否在采集时间内
        if relative_time <= self.duration:
            # 处理图像
            timestamp = image_msg.header.stamp.to_sec()
            cv_image = self.bridge.imgmsg_to_cv2(image_msg, "bgr8")

            # 处理IMU数据
            quaternion = (
                imu_msg.orientation.x,
                imu_msg.orientation.y,
                imu_msg.orientation.z,
                imu_msg.orientation.w,
            )
            roll, pitch, yaw = euler_from_quaternion(quaternion)

            # 保存同步数据
            self.synced_imu_data.append(
                {
                    "timestamp": timestamp,
                    "roll": roll,
                    "pitch": pitch,
                    "yaw": yaw,
                }
            )
            self.synced_image_data.append({"timestamp": timestamp, "image": cv_image})

    def save_data_to_csv(self):
        if not self.synced_imu_data:
            print("没有同步的数据可保存！")
            return

        # 保存同步后的数据到CSV
        print("开始保存数据")
        csv_filename = os.path.join(self.save_dir, "imu_data.csv")
        df = pd.DataFrame(self.synced_imu_data)
        df.to_csv(csv_filename, index=False)

        for image_data in self.synced_image_data:
            timestamp = image_data["timestamp"]
            image = image_data["image"]
            cv2.imwrite(
                os.path.join(self.save_dir, "images") + f"/{timestamp}.jpg", image
            )

    def run(self):
        # 等待采集完成
        while rospy.Time.now().to_sec() - self.start_time < self.duration:
            rospy.sleep(1)

        # 保存数据
        self.save_data_to_csv()

        # 关闭节点
        print("数据采集完成")


if __name__ == "__main__":
    collector = DataCollector()
    collector.run()
