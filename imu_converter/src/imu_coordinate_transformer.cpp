#include <ros/ros.h>
#include <sensor_msgs/Imu.h>
#include <Eigen/Dense>

class ImuCoordinateTransformer {
private:
    ros::NodeHandle nh_;
    ros::Subscriber imu_sub_;
    ros::Publisher imu_pub_;
    Eigen::Matrix3d rotation_matrix_;

public:
    ImuCoordinateTransformer() {
        // 订阅相机IMU数据
        imu_sub_ = nh_.subscribe("/camera_head/imu", 10, &ImuCoordinateTransformer::doTransform, this);
        // 发布转换后的IMU数据
        imu_pub_ = nh_.advertise<sensor_msgs::Imu>("/camera_head/imu_nwu", 1);
        
        // 初始化从相机坐标系(RDF)到NWU的旋转矩阵
        rotation_matrix_ << 0, 0, 1,
                           -1, 0, 0,
                            0, -1, 0;
    }

    // 协方差矩阵转换函数
    Eigen::Matrix3d transformCovariance(const Eigen::Matrix3d& cov_matrix) {
        // 协方差矩阵变换公式: C_new = R * C_old * R^T
        return rotation_matrix_ * cov_matrix * rotation_matrix_.transpose();
    }

    // 将ROS协方差数组转换为Eigen矩阵
    Eigen::Matrix3d arrayToMatrix(const boost::array<double, 9>& arr) {
        Eigen::Matrix3d mat;
        mat << arr[0], arr[1], arr[2],
               arr[3], arr[4], arr[5],
               arr[6], arr[7], arr[8];
        return mat;
    }

    // 将Eigen矩阵转换为ROS协方差数组
    boost::array<double, 9> matrixToArray(const Eigen::Matrix3d& mat) {
        boost::array<double, 9> arr;
        arr[0] = mat(0, 0); arr[1] = mat(0, 1); arr[2] = mat(0, 2);
        arr[3] = mat(1, 0); arr[4] = mat(1, 1); arr[5] = mat(1, 2);
        arr[6] = mat(2, 0); arr[7] = mat(2, 1); arr[8] = mat(2, 2);
        return arr;
    }

    // 回调函数处理IMU数据转换
    void doTransform(const sensor_msgs::Imu::ConstPtr& imu_raw) {
        sensor_msgs::Imu nwu_imu_msg;
        nwu_imu_msg.header = imu_raw->header;
        nwu_imu_msg.header.frame_id = "imu_nwu";

        // 1. 转换角速度向量
        nwu_imu_msg.angular_velocity.x = imu_raw->angular_velocity.z;
        nwu_imu_msg.angular_velocity.y = -imu_raw->angular_velocity.x;
        nwu_imu_msg.angular_velocity.z = -imu_raw->angular_velocity.y;

        // 2. 转换线加速度向量
        nwu_imu_msg.linear_acceleration.x = imu_raw->linear_acceleration.z;
        nwu_imu_msg.linear_acceleration.y = -imu_raw->linear_acceleration.x;
        nwu_imu_msg.linear_acceleration.z = -imu_raw->linear_acceleration.y;

        // 3. 转换角速度协方差矩阵
        Eigen::Matrix3d angular_cov_old = arrayToMatrix(imu_raw->angular_velocity_covariance);
        Eigen::Matrix3d angular_cov_new = transformCovariance(angular_cov_old);
        nwu_imu_msg.angular_velocity_covariance = matrixToArray(angular_cov_new);

        // 4. 转换线加速度协方差矩阵
        Eigen::Matrix3d linear_cov_old = arrayToMatrix(imu_raw->linear_acceleration_covariance);
        Eigen::Matrix3d linear_cov_new = transformCovariance(linear_cov_old);
        nwu_imu_msg.linear_acceleration_covariance = matrixToArray(linear_cov_new);

        // 5. 转换姿态协方差矩阵
        Eigen::Matrix3d orientation_cov_old = arrayToMatrix(imu_raw->orientation_covariance);
        Eigen::Matrix3d orientation_cov_new = transformCovariance(orientation_cov_old);
        nwu_imu_msg.orientation_covariance = matrixToArray(orientation_cov_new);

        // 发布转换后的IMU消息
        imu_pub_.publish(nwu_imu_msg);
    }
};

int main(int argc, char** argv) {
    ros::init(argc, argv, "imu_coordinate_transformer");
    ImuCoordinateTransformer transformer;
    ros::spin();
    return 0;
}