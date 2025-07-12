import re
import subprocess

import cv2
import numpy as np


def select_points_on_image(image_path, monitor=0):
    """
    在指定显示器上显示图像并选择10个点，返回点的坐标

    参数:
        image_path: 图像文件路径
        monitor: 显示器编号，0表示主显示器，1表示第二个显示器，依此类推

    返回:
        np.array: 包含10个点坐标的数组，形状为(10, 2)
    """
    # 读取图像
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"无法读取图像: {image_path}")

    # 复制原图用于显示
    display_image = image.copy()

    # 创建窗口
    cv2.namedWindow("Image", cv2.WND_PROP_FULLSCREEN)

    # Ubuntu系统下使用xrandr获取显示器信息
    if subprocess.call(["which", "xrandr"]) == 0:
        try:
            output = subprocess.check_output(["xrandr"]).decode()
            monitors = []
            for line in output.split("\n"):
                if " connected" in line:
                    monitors.append(line.split()[0])

            if 0 <= monitor < len(monitors):
                # 使用wmctrl设置窗口到指定显示器
                if subprocess.call(["which", "wmctrl"]) == 0:
                    cv2.imshow("Image", display_image)
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
                            rf"{monitors[monitor]}\s+connected\s+(\d+)x(\d+)", geometry
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
                print(f"警告: 指定的显示器 {monitor} 不存在，使用主显示器")
        except Exception as e:
            print(f"警告: 设置显示器位置时出错: {e}，使用主显示器")

    # 设置为全屏
    cv2.setWindowProperty("Image", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    # 存储点击的点
    points = []

    # 缩放和平移相关变量
    scale = 1.0
    offset_x, offset_y = 0, 0
    original_image = image.copy()
    height, width = image.shape[:2]

    # 鼠标拖动相关变量
    drag_start = None
    is_dragging = False
    click_threshold = 5  # 判定为点击的最大像素距离

    # 坐标校准变量
    cursor_offset_x = 0
    cursor_offset_y = 0

    def mouse_callback(event, x, y, flags, param):
        nonlocal points, display_image, scale, offset_x, offset_y, drag_start, is_dragging, cursor_offset_x, cursor_offset_y

        # 应用光标偏移校正
        corrected_x = x - cursor_offset_x
        corrected_y = y - cursor_offset_y

        # 计算实际坐标（考虑缩放和平移）
        actual_x = int((corrected_x - offset_x) / scale)
        actual_y = int((corrected_y - offset_y) / scale)

        if event == cv2.EVENT_LBUTTONDOWN:
            # 左键按下，记录起始位置
            drag_start = (corrected_x, corrected_y)
            is_dragging = False

        elif event == cv2.EVENT_MOUSEMOVE:
            # 鼠标移动
            if drag_start:
                # 计算移动距离
                dx = corrected_x - drag_start[0]
                dy = corrected_y - drag_start[1]
                distance = np.sqrt(dx * dx + dy * dy)

                # 如果移动距离超过阈值，认为是拖动
                if distance > click_threshold:
                    is_dragging = True

        elif event == cv2.EVENT_LBUTTONUP:
            # 左键释放
            if drag_start and not is_dragging:
                # 不是拖动，视为点击
                if len(points) < 10:
                    points.append([actual_x, actual_y])
                    update_display()

            # 重置拖动状态
            drag_start = None
            is_dragging = False

        elif event == cv2.EVENT_RBUTTONDOWN:
            # 右键按下，开始拖动
            drag_start = (corrected_x, corrected_y)

        elif event == cv2.EVENT_RBUTTONUP:
            # 右键释放，结束拖动
            drag_start = None

        elif event == cv2.EVENT_MOUSEMOVE and drag_start:
            # 拖动中
            dx, dy = corrected_x - drag_start[0], corrected_y - drag_start[1]
            offset_x += dx
            offset_y += dy
            drag_start = (corrected_x, corrected_y)
            update_display()

        elif event == cv2.EVENT_MOUSEWHEEL:
            # 鼠标滚轮缩放
            if flags > 0:  # 向上滚动，放大
                scale *= 1.1
            else:  # 向下滚动，缩小
                scale /= 1.1

            # 确保缩放比例在合理范围内
            scale = max(0.1, min(scale, 10.0))
            update_display()

    def update_display():
        nonlocal display_image
        # 缩放图像
        new_width = int(width * scale)
        new_height = int(height * scale)
        scaled_image = cv2.resize(original_image, (new_width, new_height))

        # 创建空白画布
        display_image = np.zeros_like(scaled_image)

        # 计算显示区域
        h, w = display_image.shape[:2]
        x1 = max(0, offset_x)
        y1 = max(0, offset_y)
        x2 = min(w, offset_x + new_width)
        y2 = min(h, offset_y + new_height)
        img_x1 = max(0, -offset_x)
        img_y1 = max(0, -offset_y)
        img_x2 = img_x1 + (x2 - x1)
        img_y2 = img_y1 + (y2 - y1)

        # 将缩放后的图像放置到显示区域
        if x2 > x1 and y2 > y1 and img_x2 > img_x1 and img_y2 > img_y1:
            display_image[y1:y2, x1:x2] = scaled_image[img_y1:img_y2, img_x1:img_x2]

        # 绘制已选点
        for i, (px, py) in enumerate(points):
            display_x = int(px * scale + offset_x)
            display_y = int(py * scale + offset_y)
            if 0 <= display_x < w and 0 <= display_y < h:
                # 使用更精确的方式绘制点，以中心点为准
                cv2.circle(display_image, (display_x, display_y), 2, (0, 0, 255), -1)
                cv2.putText(
                    display_image,
                    str(i + 1),
                    (display_x + 10, display_y + 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 0, 255),
                    2,
                )

        # 显示图像
        cv2.imshow("Image", display_image)

    # 设置鼠标回调函数
    cv2.setMouseCallback("Image", mouse_callback)

    # 显示初始图像
    update_display()

    # 等待用户点击10个点
    while len(points) < 10:
        key = cv2.waitKey(100)
        if key == 27:  # ESC键退出
            break

    # 等待1秒后自动关闭窗口
    cv2.waitKey(1000)
    cv2.destroyAllWindows()

    # 返回点的坐标数组
    return np.array(points)


# 使用示例
if __name__ == "__main__":
    image_path = "/home/wjc/Storage/humanoid_head/eye_head_data_capture/src/yesense/scripts/data_process_2/aligned_image.jpg"  # 替换为你的图像路径
    monitor_number = 0  # 替换为你想要显示的显示器编号
    try:
        points = select_points_on_image(image_path, monitor=monitor_number)
        print("选择的点坐标:")
        print(points)
    except Exception as e:
        print(f"发生错误: {e}")
    except Exception as e:
        print(f"发生错误: {e}")
