import os
import pickle

import cv2
import numpy as np
import rospy
from sensor_msgs.msg import CameraInfo


class DataLoader:
    def __init__(self):
        data_dir = "/home/wjc/Storage/humanoid_head/eye_head_data_capture/src/yesense/scripts/captured_data_3/20250707_104920"
        timestamp = "1751856573.3819400"
        color_image_path = os.path.join(data_dir, f"head_color_image/{timestamp}.jpg")
        depth_image_path = os.path.join(data_dir, f"head_depth_image/{timestamp}.png")
        camera_info_path = os.path.join(data_dir, "color_camera_info.pkl")
        self.color_image = cv2.imread(color_image_path)
        self.depth_image = cv2.imread(depth_image_path, -1)
        with open(camera_info_path, "rb") as f:
            camera_info = pickle.load(f)
            self.intrinsics = camera_info["intrinsics"]
            self.depth_scale = camera_info["depth_scale"]

    def get_3d_position(self, x, y):
        """根据二维坐标获取三维位置"""
        # 检查坐标是否在图像范围内
        height, width = self.depth_image.shape[:2]
        if x < 0 or x >= width or y < 0 or y >= height:
            raise ValueError(
                f"Coordinates ({x}, {y}) are out of image bounds ({width}, {height})"
            )

        # 获取指定像素的深度值
        depth = self.depth_image[y, x] * self.depth_scale  # 转换为米

        # 使用内参计算3D坐标
        fx = self.intrinsics["fx"]
        fy = self.intrinsics["fy"]
        cx = self.intrinsics["cx"]
        cy = self.intrinsics["cy"]

        # 计算相机坐标系下的3D坐标
        z = depth
        x_camera = (x - cx) * z / fx
        y_camera = (y - cy) * z / fy

        return [x_camera, y_camera, z]  # [x, y, z] in meters

    def visualize_depth(self):
        """可视化深度图"""
        # 将深度图转换为8位以进行显示
        depth_colormap = cv2.applyColorMap(
            cv2.convertScaleAbs(self.depth_image, alpha=0.03), cv2.COLORMAP_JET
        )
        return depth_colormap

    def select_point_and_get_3d(self):
        """通过鼠标点击选择点并获取3D坐标"""
        # 创建窗口并设置鼠标回调
        cv2.namedWindow("Color Image")
        cv2.setMouseCallback("Color Image", self.mouse_callback)

        # 显示图像
        cv2.imshow("Color Image", self.color_image)

        print("点击图像中的点获取3D坐标，按ESC键退出")
        while True:
            key = cv2.waitKey(1) & 0xFF
            if key == 27:  # ESC键
                break

        cv2.destroyAllWindows()

    def mouse_callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            # 获取3D坐标
            point = self.get_3d_position(x, y)

            # 在图像上标记点击位置
            # image = self.color_image.copy()
            # cv2.circle(image, (x, y), 5, (0, 255, 0), -1)
            cv2.circle(self.color_image, (x, y), 5, (0, 255, 0), -1)
            cv2.imshow("Color Image", self.color_image)
            # cv2.putText(
            #     image,
            #     f"X: {point[0]:.3f}m",
            #     (x + 10, y - 10),
            #     cv2.FONT_HERSHEY_SIMPLEX,
            #     0.5,
            #     (0, 255, 0),
            #     1,
            # )
            # cv2.putText(
            #     image,
            #     f"Y: {point[1]:.3f}m",
            #     (x + 10, y + 10),
            #     cv2.FONT_HERSHEY_SIMPLEX,
            #     0.5,
            #     (0, 255, 0),
            #     1,
            # )
            # cv2.putText(
            #     image,
            #     f"Z: {point[2]:.3f}m",
            #     (x + 10, y + 30),
            #     cv2.FONT_HERSHEY_SIMPLEX,
            #     0.5,
            #     (0, 255, 0),
            #     1,
            # )

            # # 显示结果
            # cv2.imshow("Color Image", image)

            print(f"点击位置: ({x}, {y})")
            print(f"3D坐标: X={point[0]:.3f}m, Y={point[1]:.3f}m, Z={point[2]:.3f}m")


def main():
    loader = DataLoader()
    loader.select_point_and_get_3d()


if __name__ == "__main__":
    main()
