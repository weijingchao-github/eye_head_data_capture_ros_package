import os
import sys

path = os.path.dirname(__file__)
sys.path.insert(0, path)

import math

import cv2
import numpy as np
from L2CSNet import gaze_direction_estimate_one_image
from mediapipe.python._framework_bindings import image as mp_image
from mediapipe.python._framework_bindings import image_frame as mp_image_frame
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision
from PIL import Image


class GetEyeGazeDirection:
    def __init__(self, human_parameters):
        self.pitch_offset = human_parameters["face_camera_pitch_angle_offet"]
        self.gaze_estimation_model = gaze_direction_estimate_one_image.GazeEstimation()
        self.facial_landmarks_detector = self._facial_landmarks_detector_init()
        # Detect blink setup
        self.might_blink_count = 0
        self.eye_ar_thresh = 0.3
        self.eye_ar_consec_frames = 1
        self.blink_flag = False

    def pipeline(self, image_raw):
        # 人脸对齐
        aligned_image = self._align_face(image_raw)
        # 视线方向计算
        eye_gaze_yaw_angle, eye_gaze_pitch_angle = (
            self.gaze_estimation_model.do_gaze_direction_estimate(aligned_image)
        )
        eye_gaze_yaw_angle = -eye_gaze_yaw_angle
        eye_gaze_pitch_angle = -(eye_gaze_pitch_angle - self.pitch_offset)
        return eye_gaze_yaw_angle, eye_gaze_pitch_angle

    def _facial_landmarks_detector_init(self):
        # 3FabRec Model can only detect one face in an image, namely only one face can appear in an image.
        base_options = mp_python.BaseOptions(
            model_asset_path="./google_mediapipe/face_landmarker_v2_with_blendshapes.task"
        )
        options = mp_vision.FaceLandmarkerOptions(base_options=base_options)
        facial_landmarks_detector = mp_vision.FaceLandmarker.create_from_options(
            options
        )
        return facial_landmarks_detector

    def _align_face(self, image_align):
        (
            eye_left_inner_corner_landmark_pixel_coordinate,
            eye_right_inner_corner_landmark_pixel_coordinate,
        ) = self._get_eye_inner_corners_from_facial_landmarks_detect_result(image_align)
        angle = float(
            np.degrees(
                np.arctan2(
                    eye_right_inner_corner_landmark_pixel_coordinate[1]
                    - eye_left_inner_corner_landmark_pixel_coordinate[1],
                    eye_right_inner_corner_landmark_pixel_coordinate[0]
                    - eye_left_inner_corner_landmark_pixel_coordinate[0],
                )
            )
        )
        # Rotate image
        image_aligned = np.array(
            Image.fromarray(image_align).rotate(
                angle, resample=Image.Resampling.BICUBIC
            )
        )  # TODO：写完代码之后show一下检查一下图片旋转的填充方式
        # cv2.imshow("a", image_aligned)
        # cv2.waitKey(0)
        return image_aligned

    def _get_eye_inner_corners_from_facial_landmarks_detect_result(self, image_fld):
        # 这个模型里包含了目标检测,所以重复检测了,但是这部分暂时去不掉.
        def normalized_to_pixel_coordinates(x, y, image_height, image_width):
            x_px = min(math.floor(x * image_width), image_width - 1)
            y_px = min(math.floor(y * image_height), image_height - 1)
            return x_px, y_px

        image_height, image_width = image_fld.shape[0], image_fld.shape[1]
        image_fld = mp_image.Image(
            image_format=mp_image_frame.ImageFormat.SRGB,
            data=image_fld.astype(np.uint8),
        )
        landmarks_detect_result = self.facial_landmarks_detector.detect(image_fld)
        if len(landmarks_detect_result.face_landmarks) == 0:
            raise Exception("Facial landmarks detect fails.")
        facial_landmarks = landmarks_detect_result.face_landmarks[0]
        eye_left_inner_corner_landmark_pixel_coordinate = (
            normalized_to_pixel_coordinates(
                facial_landmarks[133].x,
                facial_landmarks[133].y,
                image_height,
                image_width,
            )
        )
        eye_right_inner_corner_landmark_pixel_coordinate = (
            normalized_to_pixel_coordinates(
                facial_landmarks[362].x,
                facial_landmarks[362].y,
                image_height,
                image_width,
            )
        )
        return (
            eye_left_inner_corner_landmark_pixel_coordinate,
            eye_right_inner_corner_landmark_pixel_coordinate,
        )

    def _detect_blink(self, facial_landmarks, image):
        def normalized_to_pixel_coordinates(x, y, image_height, image_width):
            x_px = min(math.floor(x * image_width), image_width - 1)
            y_px = min(math.floor(y * image_height), image_height - 1)
            return x_px, y_px

        image_height, image_width = image.shape[0], image.shape[1]
        left_eye_landmarks = {
            "left": normalized_to_pixel_coordinates(
                facial_landmarks[263].x,
                facial_landmarks[263].y,
                image_height,
                image_width,
            ),
            "right": normalized_to_pixel_coordinates(
                facial_landmarks[362].x,
                facial_landmarks[362].y,
                image_height,
                image_width,
            ),
            "top": normalized_to_pixel_coordinates(
                facial_landmarks[386].x,
                facial_landmarks[386].y,
                image_height,
                image_width,
            ),
            "bottom": normalized_to_pixel_coordinates(
                facial_landmarks[374].x,
                facial_landmarks[374].y,
                image_height,
                image_width,
            ),
            "pupil": normalized_to_pixel_coordinates(
                facial_landmarks[473].x,
                facial_landmarks[473].y,
                image_height,
                image_width,
            ),
        }
        right_eye_landmarks = {
            "left": normalized_to_pixel_coordinates(
                facial_landmarks[133].x,
                facial_landmarks[133].y,
                image_height,
                image_width,
            ),
            "right": normalized_to_pixel_coordinates(
                facial_landmarks[33].x,
                facial_landmarks[33].y,
                image_height,
                image_width,
            ),
            "top": normalized_to_pixel_coordinates(
                facial_landmarks[159].x,
                facial_landmarks[159].y,
                image_height,
                image_width,
            ),
            "bottom": normalized_to_pixel_coordinates(
                facial_landmarks[145].x,
                facial_landmarks[145].y,
                image_height,
                image_width,
            ),
            "pupil": normalized_to_pixel_coordinates(
                facial_landmarks[468].x,
                facial_landmarks[468].y,
                image_height,
                image_width,
            ),
        }
        target_face_facial_landmarks = (left_eye_landmarks, right_eye_landmarks)

        left_eye_aspect_ratio = abs(
            (
                target_face_facial_landmarks[0]["bottom"][1]
                - target_face_facial_landmarks[0]["top"][1]
            )
            / (
                target_face_facial_landmarks[0]["left"][0]
                - target_face_facial_landmarks[0]["right"][0]
            )
        )
        right_eye_aspect_ratio = abs(
            (
                target_face_facial_landmarks[1]["bottom"][1]
                - target_face_facial_landmarks[1]["top"][1]
            )
            / (
                target_face_facial_landmarks[1]["left"][0]
                - target_face_facial_landmarks[1]["right"][0]
            )
        )

        eye_aspect_ratio = (left_eye_aspect_ratio + right_eye_aspect_ratio) / 2
        # print(eye_aspect_ratio)

        # 两种方案:
        # 一种是设置几秒之内只能眨一次眼,如设置5秒,接收到一次眨眼检测结果后,5秒钟之内再接收到眨眼不会再作为一次眨眼动作
        # 另一种是闭眼、抬眼结束后再闭眼，才算是一次新的眨眼,对于这个流程的识别放在后处理模块,不在视觉模块中实现
        # 这里的实现采用方案二
        if eye_aspect_ratio < self.eye_ar_thresh:
            self.might_blink_count += 1
            if self.might_blink_count >= self.eye_ar_consec_frames:
                self.blink_flag = True
        else:
            self.might_blink_count = 0
            self.blink_flag = False
        # return eye_aspect_ratio
