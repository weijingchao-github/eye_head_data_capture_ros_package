import json
import os

from spatialmath import SE3, UnitQuaternion


def main():
    target_position_in_head_color_camera = (-1, -1, -1)
    quat_rotation = (
        0,
        0,
        0.7071,
        0.7071,
    )  # x, y, z, w格式的头部姿态IMU四元数，在RealSense IMU坐标系下
    # 通过运行rosrun tf tf_echo camera_imu_optical_frame camera_depth_optical_frame
    # 发现IMU坐标系和彩色摄像头坐标系只有三维位置偏移，没有旋转偏移
    # IMU坐标系目前调研的和彩色摄像头坐标系基坐标形式是一样的，后面通过实验来确定这句话对不对，然后改这行注释
    quaternion = UnitQuaternion(quat_rotation)
    # ROS坐标系到相机坐标系
    tf_ROS_to_camera = SE3.Rz(90, "deg") * SE3.Ry(90, "deg")
    with open(
        os.path.join(os.path.dirname(__file__), "human_parameters.json"),
        "r",
        encoding="utf-8",
    ) as f:
        human_parameters = json.load(f)
    # 彩色摄像头与base坐标系在相机坐标系描述下的固定位置偏差(x, y, z)
    head_color_camera_position = [
        -human_parameters["head_color_camera_to_base"]["y"],
        -human_parameters["head_color_camera_to_base"]["z"],
        human_parameters["head_color_camera_to_base"]["x"],
    ]
    tf_base_to_head_color_camera = (
        SE3(head_color_camera_position) * quaternion.SE3() * tf_ROS_to_camera
    )
    tf_head_color_camera_to_base = tf_base_to_head_color_camera.inv()
    target_position_in_base = (
        tf_head_color_camera_to_base * target_position_in_head_color_camera
    )
    print(target_position_in_base)


if __name__ == "__main__":
    main()
