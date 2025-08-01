import transforms3d as t3d
from spatialmath import UnitQuaternion


def calculate_rectified_head_pose(
    initial_head_camera_pose_quaternion_xyzw,
    current_head_camera_pose_quaternion_xyzw,
):
    current_head_camera_pose_quaternion_wxyz = (
        current_head_camera_pose_quaternion_xyzw[-1],
        current_head_camera_pose_quaternion_xyzw[0],
        current_head_camera_pose_quaternion_xyzw[1],
        current_head_camera_pose_quaternion_xyzw[2],
    )
    quaternion_current = UnitQuaternion(current_head_camera_pose_quaternion_wxyz)
    initial_head_camera_pose_quaternion_wxyz = (
        initial_head_camera_pose_quaternion_xyzw[-1],
        initial_head_camera_pose_quaternion_xyzw[0],
        initial_head_camera_pose_quaternion_xyzw[1],
        initial_head_camera_pose_quaternion_xyzw[2],
    )
    quaternion_initial = UnitQuaternion(initial_head_camera_pose_quaternion_wxyz)
    quaternion_initial_so3_inv = quaternion_initial.SO3().inv()
    rectified_head_pose_so3 = quaternion_current.SO3() * quaternion_initial_so3_inv
    rectified_head_pose_quaternion_wxyz = t3d.quaternions.mat2quat(
        rectified_head_pose_so3.A
    )  # w, x, y,z
    return rectified_head_pose_quaternion_wxyz


# if __name__ == "__main__":
#     main()
