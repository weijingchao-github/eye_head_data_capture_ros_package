import numpy as np
from filterpy.kalman import KalmanFilter
from scipy import signal


def butterworth_lowpass_filter(
    eye_gaze_direction_raw_data, cutoff_freq=5, fs=60, order=5
):
    """
    巴特沃斯低通滤波器
    :param eye_gaze_direction_raw_data: 原始数据
    :param cutoff_freq: 截止频率
    :param fs: 采样频率
    :param order: 滤波器阶数
    :return: 滤波后的数据
    """
    timestamp_set = []
    raw_yaw_data_set = []
    raw_pitch_data_set = []
    for raw_data in eye_gaze_direction_raw_data:
        timestamp_set.append(raw_data["timestamp"])
        raw_yaw_data_set.append(raw_data["yaw"])
        raw_pitch_data_set.append(raw_data["pitch"])
    nyquist = 0.5 * fs
    normal_cutoff = cutoff_freq / nyquist
    # 设计Butterworth低通滤波器
    b, a = signal.butter(order, normal_cutoff, btype="low", analog=False)
    # 应用滤波器
    processed_yaw_data = signal.filtfilt(b, a, raw_yaw_data_set)
    # 设计Butterworth低通滤波器
    b, a = signal.butter(order, normal_cutoff, btype="low", analog=False)
    # 应用滤波器
    processed_pitch_data = signal.filtfilt(b, a, raw_pitch_data_set)
    eye_gaze_direction_filtered_data = []
    for timetamp, yaw, pitch in zip(
        timestamp_set, processed_yaw_data, processed_pitch_data
    ):
        eye_gaze_direction_filtered_data.append(
            {
                "timestamp": timetamp,
                "yaw": yaw,
                "pitch": pitch,
            }
        )
    return eye_gaze_direction_filtered_data


def kalman_filter(eye_gaze_direction_raw_data):
    kalman_filter = KalmanFilter(dim_x=4, dim_z=2)
    delta_t = 1 / 60
    eye_gaze_direction_kalman_filtered_data = []
    eye_gaze_direction_kalman_filtered_data.append(eye_gaze_direction_raw_data[0])
    initial_eye_gaze_direction = (
        eye_gaze_direction_raw_data[0]["yaw"],
        eye_gaze_direction_raw_data[0]["pitch"],
    )
    kalman_filter.x = np.array([*initial_eye_gaze_direction, 0, 0]).reshape(
        4, 1
    )  # 初始状态矩阵
    kalman_filter.F = np.array(
        [
            [1, 0, delta_t, 0],
            [0, 1, 0, delta_t],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
        ]
    )  # 状态转移矩阵
    kalman_filter.H = np.array(
        [
            [1, 0, 0, 0],
            [0, 1, 0, 0],
        ]
    )  # 观测矩阵
    kalman_filter.P = np.array(
        [
            [16, 0, 0, 0],
            [0, 16, 0, 0],
            [0, 0, 100, 0],
            [0, 0, 0, 100],
        ]
    )  # 初始误差协方差矩阵
    # kalman_filter.P *= 1000
    # kalman_filter.R *= 10  # state uncertainty
    # kalman_filter.Q[2:, 2:] *= 0.01
    kalman_filter.Q = np.array(
        [
            [5, 0, 0, 0],
            [0, 5, 0, 0],
            [0, 0, 0.5, 0],
            [0, 0, 0, 0.5],
        ]
    )
    kalman_filter.R = np.array(
        [
            [16, 0],
            [0, 16],
        ]
    )
    for raw_data in eye_gaze_direction_raw_data[1:]:
        timestamp = raw_data["timestamp"]
        raw_eye_gaze_direction = (raw_data["yaw"], raw_data["pitch"])
        kalman_filter.predict()
        kalman_filter.update(raw_eye_gaze_direction)
        kf_eye_gaze_direction = kalman_filter.x.reshape(-1)
        eye_gaze_direction_kalman_filtered_data.append(
            {
                "timstamp": timestamp,
                "yaw": kf_eye_gaze_direction[0],
                "pitch": kf_eye_gaze_direction[1],
            }
        )
    return eye_gaze_direction_kalman_filtered_data
