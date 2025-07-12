import re
import subprocess

import cv2
import numpy as np
from sensor_msgs.msg import CameraInfo


class GetTargetPosition:
    def __init__(
        self, head_camera_color_image, head_camera_depth_image, camera_info, monitor
    ):
        self.color_image = head_camera_color_image
        self.depth_image = head_camera_depth_image
        self.intrinsics = camera_info["intrinsics"]
        self.depth_scale = camera_info["depth_scale"]
        self.monitor = monitor
        self.target_position_in_head_camera_frame = []

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

    def select_point_and_get_3d(self):
        """通过鼠标点击选择点并获取3D坐标"""
        # 创建窗口并设置鼠标回调
        cv2.namedWindow("Color Image", cv2.WND_PROP_FULLSCREEN)
        # Ubuntu系统下使用xrandr获取显示器信息

        if subprocess.call(["which", "xrandr"]) == 0:
            try:
                output = subprocess.check_output(["xrandr"]).decode()
                monitors = []
                for line in output.split("\n"):
                    if " connected" in line:
                        monitors.append(line.split()[0])

                if 0 <= self.monitor < len(monitors):
                    # 使用wmctrl设置窗口到指定显示器
                    if subprocess.call(["which", "wmctrl"]) == 0:
                        cv2.imshow("Image", self.color_image)
                        cv2.waitKey(100)  # 确保窗口已创建

                        # 获取窗口ID
                        window_id_output = subprocess.check_output(
                            ["wmctrl", "-l"]
                        ).decode()
                        window_id = None
                        for line in window_id_output.split("\n"):
                            if "Image" in line:
                                window_id = line.split()[0]
                                break

                        if window_id:
                            # 获取显示器几何信息
                            geometry = subprocess.check_output(
                                ["xrandr", "--query"]
                            ).decode()
                            match = re.search(
                                rf"{monitors[self.monitor]}\s+connected\s+(\d+)x(\d+)",
                                geometry,
                            )
                            if match:
                                width, height = int(match.group(1)), int(match.group(2))
                                # 将窗口移动到指定显示器
                                subprocess.call(
                                    [
                                        "wmctrl",
                                        "-i",
                                        "-r",
                                        window_id,
                                        "-e",
                                        f"0,0,0,{width},{height}",
                                    ]
                                )
                else:
                    print(f"警告: 指定的显示器 {self.monitor} 不存在，使用主显示器")
            except Exception as e:
                print(f"警告: 设置显示器位置时出错: {e}，使用主显示器")
        cv2.setWindowProperty(
            "Color Image", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN
        )
        cv2.setMouseCallback("Color Image", self.mouse_callback)

        # 显示图像
        cv2.imshow("Color Image", self.color_image)

        print("点击图像中的点获取3D坐标")
        while not self.target_position_in_head_camera_frame:
            key = cv2.waitKey(100)
            if key == 27:  # ESC键
                break

        # 等待1秒后自动关闭窗口
        cv2.waitKey(1000)
        cv2.destroyAllWindows()
        return self.target_position_in_head_camera_frame

    def mouse_callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            # 获取3D坐标
            point = self.get_3d_position(x, y)

            # 在图像上标记点击位置
            # image = self.color_image.copy()
            # cv2.circle(image, (x, y), 5, (0, 255, 0), -1)
            cv2.circle(self.color_image, (x, y), 1, (0, 255, 0), -1)
            cv2.imshow("Color Image", self.color_image)

            # # 显示结果
            # cv2.imshow("Color Image", image)
            print(f"点击位置: ({x}, {y})")
            print(f"3D坐标: X={point[0]:.3f}m, Y={point[1]:.3f}m, Z={point[2]:.3f}m")
            self.target_position_in_head_camera_frame = point
