import numpy as np
from spatialmath import SE3, UnitQuaternion


def calculate_target_position_in_base_frame(
    target_position_in_head_camera_frame,
    initial_head_camera_pose_quaternion_xyzw,
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
    base_orign_to_head_camera_position_in_nwu = [
        -head_camera_position_to_base_in_nwu[0],
        -head_camera_position_to_base_in_nwu[1],
        -head_camera_position_to_base_in_nwu[2],
    ]
    initial_head_camera_pose_quaternion_wxyz = [
        initial_head_camera_pose_quaternion_xyzw[-1],
        initial_head_camera_pose_quaternion_xyzw[0],
        initial_head_camera_pose_quaternion_xyzw[1],
        initial_head_camera_pose_quaternion_xyzw[2],
    ]
    initial_head_camera_pose_quaternion = UnitQuaternion(
        initial_head_camera_pose_quaternion_wxyz
    )
    base_orign_in_initial_head_camera_pose_frame = (
        initial_head_camera_pose_quaternion.SE3().inv()
        * np.array(base_orign_to_head_camera_position_in_nwu)
    ).reshape(-1)
    tf_base_frame_to_current_head_camera_pose_frame = (
        current_head_camera_pose_quaternion.SE3()
        * SE3(-base_orign_in_initial_head_camera_pose_frame)
        * (SE3.Ry(90, "deg") * SE3.Rz(-90, "deg"))
    )
    target_position_in_base_frame = (
        tf_base_frame_to_current_head_camera_pose_frame
        * np.array(target_position_in_head_camera_frame)
    ).reshape(-1)
    return target_position_in_base_frame
