import numpy as np
import rospy
from sensor_msgs.msg import Imu


class ImuCoordinateTransformer:
    def __init__(self):
        rospy.Subscriber("/camera_head/imu", Imu, self.do, queue_size=1)
        self.pub = rospy.Publisher("/camear_head/imu_nwu", Imu, queue_size=1)
        # 定义从相机坐标系(RDF)到NWU的旋转矩阵
        self.rotation_matrix = np.array([[0, 0, 1], [-1, 0, 0], [0, -1, 0]])

    def transform_covariance(self, cov_matrix):
        """将协方差矩阵通过旋转矩阵进行变换"""
        # 协方差矩阵变换公式: C_new = R * C_old * R^T
        return self.rotation_matrix @ cov_matrix @ self.rotation_matrix.T

    def do(self, imu_raw):
        nwu_imu_msg = Imu()
        nwu_imu_msg.header = imu_raw.header
        nwu_imu_msg.header.frame_id = "imu_nwu"

        # 1. 角速度转换（向量转换：遵循旋转矩阵）
        # NWU_X = 相机_Z；NWU_Y = -相机_X；NWU_Z = -相机_Y
        nwu_imu_msg.angular_velocity.x = imu_raw.angular_velocity.z
        nwu_imu_msg.angular_velocity.y = -imu_raw.angular_velocity.x
        nwu_imu_msg.angular_velocity.z = -imu_raw.angular_velocity.y

        # 2. 线加速度转换（同向量转换规则）
        nwu_imu_msg.linear_acceleration.x = imu_raw.linear_acceleration.z
        nwu_imu_msg.linear_acceleration.y = -imu_raw.linear_acceleration.x
        nwu_imu_msg.linear_acceleration.z = -imu_raw.linear_acceleration.y

        # 3. 转换角速度协方差矩阵
        angular_cov_old = np.array(imu_raw.angular_velocity_covariance).reshape(3, 3)
        angular_cov_new = self.transform_covariance(angular_cov_old)
        nwu_imu_msg.angular_velocity_covariance = angular_cov_new.flatten().tolist()

        # 4. 转换线加速度协方差矩阵
        linear_cov_old = np.array(imu_raw.linear_acceleration_covariance).reshape(3, 3)
        linear_cov_new = self.transform_covariance(linear_cov_old)
        nwu_imu_msg.linear_acceleration_covariance = linear_cov_new.flatten().tolist()

        # 5. 转换姿态协方差矩阵
        orientation_cov_old = np.array(imu_raw.orientation_covariance).reshape(3, 3)
        orientation_cov_new = self.transform_covariance(orientation_cov_old)
        nwu_imu_msg.orientation_covariance = orientation_cov_new.flatten().tolist()

        self.pub.publish(nwu_imu_msg)


def main():
    rospy.init_node("imu_coordinate_transformer")
    ImuCoordinateTransformer()
    rospy.spin()


if __name__ == "__main__":
    main()
