import numpy as np
from spatialmath import SE3, UnitQuaternion


def calculate_target_position_in_base_frame(
    target_position_in_head_camera_frame,
    current_head_camera_pose_quaternion_xyzw,
    human_parameters,
):
    current_head_camera_pose_quaternion_wxyz = (
        current_head_camera_pose_quaternion_xyzw[-1],
        current_head_camera_pose_quaternion_xyzw[0],
        current_head_camera_pose_quaternion_xyzw[1],
        current_head_camera_pose_quaternion_xyzw[2],
    )
    current_head_camera_pose_quaternion = UnitQuaternion(
        current_head_camera_pose_quaternion_wxyz
    )
    # 彩色摄像头与base坐标系在NWU坐标系描述下的固定位置偏差(x, y, z)
    head_camera_position_to_base_in_nwu = [
        human_parameters["head_camera_position_to_base"]["x"],
        human_parameters["head_camera_position_to_base"]["y"],
        human_parameters["head_camera_position_to_base"]["z"],
    ]
    tf_base_frame_to_current_head_camera_pose_frame = (
        current_head_camera_pose_quaternion.SE3()
        * SE3(head_camera_position_to_base_in_nwu)
        * (SE3.Ry(90, "deg") * SE3.Rz(-90, "deg"))
    )
    target_position_in_base_frame = (
        tf_base_frame_to_current_head_camera_pose_frame
        * np.array(target_position_in_head_camera_frame)
    ).reshape(-1)
    return target_position_in_base_frame
