import transforms3d as t3d
from spatialmath import SO3
from tf.transformations import euler_from_quaternion


def get_current_rectified_head_pose(
    initial_head_camera_pose_quaternion_xyzw,
    current_head_camera_pose_quaternion_xyzw,
):
    roll, pitch, initial_yaw = euler_from_quaternion(
        initial_head_camera_pose_quaternion_xyzw, axes="rxyz"
    )
    # print(f"roll {roll}, pitch {pitch}, yaw {initial_yaw}")
    roll, pitch, current_yaw = euler_from_quaternion(
        current_head_camera_pose_quaternion_xyzw, axes="rxyz"
    )
    rectified_yaw = current_yaw - initial_yaw
    rectified_current_head_pose_so3 = (
        SO3.Rz(rectified_yaw) * SO3.Ry(pitch) * SO3.Rx(roll)
    )
    # 这个返回的是单位四元数吗(经过测试是的)
    rectified_current_head_pose_quaternion_wxyz = t3d.quaternions.mat2quat(
        rectified_current_head_pose_so3.A
    )
    # # 测试
    # roll, pitch, yaw = euler_from_quaternion(
    #     (
    #         rectified_head_pose_quaternion_wxyz[1],
    #         rectified_head_pose_quaternion_wxyz[2],
    #         rectified_head_pose_quaternion_wxyz[3],
    #         rectified_head_pose_quaternion_wxyz[0],
    #     ),
    #     axes="rxyz",
    # )
    # print(f"roll {roll}, pitch {pitch}, yaw {yaw}")
    rectified_current_head_pose_quaternion_xyzw = (
        rectified_current_head_pose_quaternion_wxyz[1],
        rectified_current_head_pose_quaternion_wxyz[2],
        rectified_current_head_pose_quaternion_wxyz[3],
        rectified_current_head_pose_quaternion_wxyz[0],
    )
    return rectified_current_head_pose_quaternion_xyzw


def get_initial_rectified_head_pose(initial_head_camera_pose_quaternion_xyzw):
    roll, pitch, yaw = euler_from_quaternion(
        initial_head_camera_pose_quaternion_xyzw, axes="rxyz"
    )
    rectified_initial_head_pose_so3 = SO3.Ry(pitch) * SO3.Rx(roll)
    rectified_initial_head_pose_quaternion_wxyz = t3d.quaternions.mat2quat(
        rectified_initial_head_pose_so3.A
    )
    rectified_initial_head_pose_quaternion_xyzw = (
        rectified_initial_head_pose_quaternion_wxyz[1],
        rectified_initial_head_pose_quaternion_wxyz[2],
        rectified_initial_head_pose_quaternion_wxyz[3],
        rectified_initial_head_pose_quaternion_wxyz[0],
    )
    return rectified_initial_head_pose_quaternion_xyzw


# if __name__ == "__main__":
#     main()
