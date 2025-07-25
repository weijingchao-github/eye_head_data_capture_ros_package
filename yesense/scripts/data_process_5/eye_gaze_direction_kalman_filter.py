import numpy as np
from filterpy.kalman import KalmanFilter


def eye_gaze_direction_kalman_filter(eye_gaze_direction_raw_data):
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
