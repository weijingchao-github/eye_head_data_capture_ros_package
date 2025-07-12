import os
import pickle
import time
from datetime import datetime

import cv2
import pandas as pd
import rospy
from cv_bridge import CvBridge
from message_filters import ApproximateTimeSynchronizer, Subscriber
from sensor_msgs.msg import CameraInfo, Image, Imu


class DataCollector:
    def __init__(self):
        # 初始化ROS节点
        rospy.init_node("sensor_data_collector", anonymous=True)
        time.sleep(3)
        # 参数设置
        self.duration = 60  # 采集时长（秒）
        self.head_color_image_topic = "/camera_head/color/image_raw"  # 摄像头话题
        self.head_color_camera_info_topic = "/camera_head/color/camera_info"
        self.head_depth_image_topic = "/camera_head/aligned_depth_to_color/image_raw"
        self.face_image_topic = "/camera_face/color/image_raw"  # 摄像头话题
        self.imu_topic = "/imu/data"  # IMU话题
        save_folder = "captured_data_3"
        self.save_dir = os.path.join(
            os.path.join(os.path.dirname(__file__), save_folder),
            datetime.now().strftime("%Y%m%d_%H%M%S"),
        )  # 保存目录
        self.queue_size = 30  # 同步队列大小
        self.slop = 1 / 60  # 时间容差（秒）

        # 创建保存目录
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)
            os.makedirs(os.path.join(self.save_dir, "head_color_image"))
            os.makedirs(os.path.join(self.save_dir, "head_depth_image"))
            os.makedirs(os.path.join(self.save_dir, "face_image"))

        # 数据存储
        self.bridge = CvBridge()
        self.synced_head_color_image_data = []
        self.synced_head_depth_image_data = []
        self.synced_face_image_data = []
        self.synced_imu_data = []

        # 创建消息过滤器订阅者
        self.head_color_image_sub = Subscriber(self.head_color_image_topic, Image)
        self.head_depth_image_sub = Subscriber(self.head_depth_image_topic, Image)
        self.face_image_sub = Subscriber(self.face_image_topic, Image)
        self.imu_sub = Subscriber(self.imu_topic, Imu)

        # 设置近似时间同步器
        self.ts = ApproximateTimeSynchronizer(
            [
                self.head_color_image_sub,
                self.head_depth_image_sub,
                self.face_image_sub,
                self.imu_sub,
            ],
            queue_size=self.queue_size,
            slop=self.slop,
        )
        self.ts.registerCallback(self.sync_callback)

        # 记录开始时间
        self.start_time = rospy.Time.now().to_sec()
        rospy.loginfo(f"开始数据采集，将持续 {self.duration} 秒")
        rospy.loginfo(f"数据将保存到: {self.save_dir}")

    def sync_callback(
        self, head_color_image_msg, head_depth_image_msg, face_image_msg, imu_msg
    ):
        # 计算相对时间
        current_time = rospy.Time.now().to_sec()
        relative_time = current_time - self.start_time
        # 检查是否在采集时间内
        if relative_time <= self.duration:
            # 处理图像
            timestamp = head_color_image_msg.header.stamp.to_sec()
            timestamp_str = str(timestamp)
            if "." in timestamp_str:
                integer_part, decimal_part = timestamp_str.split(".", 1)
            else:
                integer_part = timestamp_str
                decimal_part = ""
            # 补零操作：将小数部分补零到指定长度
            pad_length = 7
            if decimal_part:
                # 补0到指定长度（不足则补0，超过则保留原长度，避免截断有效数据）
                padded_decimal = decimal_part.ljust(pad_length, "0")
            else:
                # 无小数部分时，添加补0后的小数（例如补6位则为"000000"）
                padded_decimal = "0" * pad_length
            timestamp = f"{integer_part}.{padded_decimal}"

            # 保存同步数据
            self.synced_imu_data.append(
                {
                    "timestamp": timestamp,
                    "quaternion": imu_msg.orientation,
                }
            )
            self.synced_head_color_image_data.append(
                {"timestamp": timestamp, "image": head_color_image_msg}
            )
            self.synced_head_depth_image_data.append(
                {"timestamp": timestamp, "image": head_depth_image_msg}
            )
            self.synced_face_image_data.append(
                {"timestamp": timestamp, "image": face_image_msg}
            )

    def save_data_to_csv(self):
        if not self.synced_imu_data:
            print("没有同步的数据可保存！")
            return

        # 保存同步后的数据到CSV
        print("开始保存数据")
        print("")
        print("")
        print("")
        print("")
        print("")
        print("")
        print("")
        print("")
        print("")
        print("")
        print("")
        print("")
        print("")
        print("")
        print("")
        print("")
        print("")
        imu_data = []
        for synced_imu_data in self.synced_imu_data:
            timestamp = synced_imu_data["timestamp"]
            imu_data.append(
                {
                    "timestamp": timestamp,
                    "x": synced_imu_data["quaternion"].x,
                    "y": synced_imu_data["quaternion"].y,
                    "z": synced_imu_data["quaternion"].z,
                    "w": synced_imu_data["quaternion"].w,
                }
            )
        csv_filename = os.path.join(self.save_dir, "imu_data.csv")
        df = pd.DataFrame(imu_data)
        df.to_csv(csv_filename, index=False)

        for image_data in self.synced_head_color_image_data:
            timestamp = image_data["timestamp"]
            image = image_data["image"]
            image = self.bridge.imgmsg_to_cv2(image, "bgr8")
            cv2.imwrite(
                os.path.join(self.save_dir, "head_color_image") + f"/{timestamp}.jpg",
                image,
            )

        for image_data in self.synced_head_depth_image_data:
            timestamp = image_data["timestamp"]
            image = image_data["image"]
            image = self.bridge.imgmsg_to_cv2(image, desired_encoding="passthrough")
            cv2.imwrite(
                os.path.join(self.save_dir, "head_depth_image") + f"/{timestamp}.png",
                image,
            )

        for image_data in self.synced_face_image_data:
            timestamp = image_data["timestamp"]
            image = image_data["image"]
            image = self.bridge.imgmsg_to_cv2(image, "bgr8")
            image = cv2.flip(image, 0)
            cv2.imwrite(
                os.path.join(self.save_dir, "face_image") + f"/{timestamp}.jpg", image
            )

        # 保存深度相机内参和深度比例
        camera_info = rospy.wait_for_message(
            self.head_color_camera_info_topic, CameraInfo, timeout=5
        )
        with open(os.path.join(self.save_dir, "color_camera_info.pkl"), "wb") as f:
            intrinsics = {
                "fx": camera_info.K[0],
                "fy": camera_info.K[4],
                "cx": camera_info.K[2],
                "cy": camera_info.K[5],
            }
            depth_scale = rospy.get_param("/camera_head/depth/scale", 0.001)
            pickle.dump({"intrinsics": intrinsics, "depth_scale": depth_scale}, f)

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
