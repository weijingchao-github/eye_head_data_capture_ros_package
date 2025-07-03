import os
import re

import cv2
import numpy as np


def get_image_files(folder_path):
    """获取文件夹中所有jpg格式的图片文件"""
    image_files = []
    # 检查文件夹是否存在
    if not os.path.exists(folder_path):
        print(f"错误：文件夹 '{folder_path}' 不存在")
        return []
    # 遍历文件夹中的所有文件
    for filename in os.listdir(folder_path):
        if filename.endswith(".jpg"):
            image_files.append(filename)
    return image_files


def extract_timestamp(filename):
    """从文件名中提取时间戳"""
    try:
        # 使用正则表达式匹配时间戳格式
        match = re.search(r"(\d+\.\d+)", filename)
        if match:
            return float(match.group(1))
        else:
            # 如果文件名中没有时间戳，返回0
            print(f"警告：无法从文件名 '{filename}' 中提取时间戳")
            return 0
    except ValueError:
        print(f"警告：文件名 '{filename}' 中的时间戳格式不正确")
        return 0


def sort_images_by_timestamp(image_files):
    """按时间戳对图片文件进行排序"""
    # 根据提取的时间戳对图片文件列表进行排序
    return sorted(image_files, key=lambda x: extract_timestamp(x))


def display_images(folder_path, image_files, display_time=1000):
    """按顺序显示图片"""
    # 检查是否有图片可显示
    if not image_files:
        print("没有找到图片文件")
        return
    # 创建窗口
    cv2.namedWindow("Images", cv2.WINDOW_NORMAL)
    # 按顺序显示图片
    for filename in image_files:
        print(filename)
        file_path = os.path.join(folder_path, filename)
        try:
            # 读取图片
            img = cv2.imread(file_path)
            if img is None:
                print(f"错误：无法读取图片 '{filename}'")
                continue
            # # 获取图片的时间戳
            # timestamp = extract_timestamp(filename)
            # # 在图片上添加时间戳文本
            # cv2.putText(
            #     img,
            #     f"Timestamp: {timestamp}",
            #     (10, 30),
            #     cv2.FONT_HERSHEY_SIMPLEX,
            #     0.7,
            #     (0, 255, 0),
            #     2,
            # )
            # 显示图片
            cv2.imshow("Images", img)
            # 等待按键事件
            key = cv2.waitKey(display_time)
            # 如果按下ESC键或q键，则退出循环
            if key == 27 or key == ord("q"):
                break
        except Exception as e:
            print(f"错误：显示图片 '{filename}' 时发生异常: {e}")
    # 关闭所有窗口
    cv2.destroyAllWindows()


def main():
    """主函数"""
    # 设置图片文件夹路径
    folder_path = "captured_data/20250528_143603/images"  # 请替换为实际的图片文件夹路径

    # 获取图片文件
    image_files = get_image_files(folder_path)
    if not image_files:
        print("没有找到jpg格式的图片文件")
        return

    # 按时间戳排序
    sorted_images = sort_images_by_timestamp(image_files)
    print(f"找到 {len(sorted_images)} 张图片，已按时间戳排序")

    # 显示图片（每张图片显示1秒）
    display_images(folder_path, sorted_images, display_time=17)


if __name__ == "__main__":
    main()
