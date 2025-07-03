import json
import math
import os

import cv2
import numpy as np
from mediapipe.python._framework_bindings import image as mp_image
from mediapipe.python._framework_bindings import image_frame as mp_image_frame
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision
from PIL import Image


class ImageProcess:
    def __init__(self):
        # model init
        self.facial_landmarks_detector = self._facial_landmarks_detector_init()
        # Detect blink setup
        self.might_blink_count = 0
        self.eye_ar_thresh = 0.3
        self.eye_ar_consec_frames = 1
        self.blink_flag = False
        with open(
            os.path.join(os.path.dirname(__file__), "human_parameters.json"),
            "r",
            encoding="utf-8",
        ) as f:
            human_parameters = json.load(f)
        self.eye_yaw_scale = human_parameters["eye_yaw_scale_640_480"]
        self.eye_pitch_scale = human_parameters["eye_pitch_scale_640_480"]
        self.eye_ball_radius = human_parameters["eye_ball_radius"]

    def pipeline(self, image_raw):
        # 人脸特征点检测(里面包含了人脸检测)
        eye_corners_facial_landmarks = self._detect_facial_landmarks(
            image_raw, [33, 263]
        )  # 33和263是人双眼远离鼻梁的两个眼角
        if eye_corners_facial_landmarks is None:
            print("Facial landmarks detect error!")
            return None
        # 人脸对齐
        aligned_image = self._align_face(image_raw, eye_corners_facial_landmarks[0])
        # 人脸特征点检测
        facial_landmarks = self._detect_facial_landmarks(aligned_image)
        if facial_landmarks is None:
            print("Facial landmarks detect error!")
            return None
        facial_landmarks = facial_landmarks[1]
        # 眨眼检测
        self._detect_blink(facial_landmarks, aligned_image)
        if self.blink_flag:
            print("Blinking.")
            return None
        # 眼睛位置计算
        eye_left_pupil_position, eye_right_pupil_position = (
            self._calculate_eye_position(facial_landmarks, aligned_image)
        )
        # 视线方向计算
        eye_gaze_yaw_angle, eye_gaze_pitch_angle = self._calculate_gaze_direction(
            eye_left_pupil_position, eye_right_pupil_position
        )
        return eye_gaze_yaw_angle, eye_gaze_pitch_angle

    def _calculate_gaze_direction(
        self, eye_left_pupil_position, eye_right_pupil_position
    ):
        eye_left_gaze_yaw_angle = math.asin(
            eye_left_pupil_position[0] * self.eye_yaw_scale / self.eye_ball_radius
        )
        eye_left_gaze_pitch_angle = math.asin(
            eye_left_pupil_position[1] * self.eye_pitch_scale / self.eye_ball_radius
        )
        eye_right_gaze_yaw_angle = math.asin(
            eye_right_pupil_position[0] * self.eye_yaw_scale / self.eye_ball_radius
        )
        eye_right_gaze_pitch_angle = math.asin(
            eye_right_pupil_position[1] * self.eye_pitch_scale / self.eye_ball_radius
        )
        eye_gaze_yaw_angle = (eye_left_gaze_yaw_angle + eye_right_gaze_yaw_angle) / 2
        eye_gaze_pitch_angle = (
            eye_left_gaze_pitch_angle + eye_right_gaze_pitch_angle
        ) / 2
        return eye_gaze_yaw_angle, eye_gaze_pitch_angle

    def _calculate_eye_position(self, facial_landmarks, image):
        def normalized_to_pixel_coordinates(x, y, image_height, image_width):
            x_px = min(math.floor(x * image_width), image_width - 1)
            y_px = min(math.floor(y * image_height), image_height - 1)
            return x_px, y_px

        image_height, image_width = image.shape
        eye_left_landmarks = {
            "left_corner": normalized_to_pixel_coordinates(
                facial_landmarks[33].x,
                facial_landmarks[33].y,
                image_height,
                image_width,
            ),
            "right_corner": normalized_to_pixel_coordinates(
                facial_landmarks[133].x,
                facial_landmarks[133].y,
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
        eye_right_landmarks = {
            "left_corner": normalized_to_pixel_coordinates(
                facial_landmarks[362].x,
                facial_landmarks[362].y,
                image_height,
                image_width,
            ),
            "right_corner": normalized_to_pixel_coordinates(
                facial_landmarks[263].x,
                facial_landmarks[263].y,
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
        eye_left_pupil_x_position = (
            eye_left_landmarks["pupil"][0]
            - (
                eye_left_landmarks["right_corner"][0]
                - eye_left_landmarks["left_corner"][0]
            )
            / 2
        )
        eye_left_pupil_y_position = -(
            eye_left_landmarks["pupil"][1] - eye_left_landmarks["left_corner"][1]
        )
        eye_right_pupil_x_position = (
            eye_right_landmarks["pupil"][0]
            - (
                eye_right_landmarks["right_corner"][0]
                - eye_right_landmarks["left_corner"][0]
            )
            / 2
        )
        eye_right_pupil_y_position = -(
            eye_right_landmarks["pupil"][1] - eye_right_landmarks["left_corner"][1]
        )
        eye_left_pupil_position = (eye_left_pupil_x_position, eye_left_pupil_y_position)
        eye_right_pupil_position = (
            eye_right_pupil_x_position,
            eye_right_pupil_y_position,
        )
        return eye_left_pupil_position, eye_right_pupil_position

    def _plot_facial_landmarks(
        self, detected_faces, facial_landmarks, image_facial_landmarks_plot
    ):
        if not isinstance(detected_faces, list):
            detected_faces = [
                detected_faces,
            ]
        if not isinstance(facial_landmarks, list):
            facial_landmarks = [
                facial_landmarks,
            ]
        for detected_face, facial_lms in zip(detected_faces, facial_landmarks):
            left_eye_landmarks, right_eye_landmarks = facial_lms
            x, y, w, h = detected_face["xywh"]
            offset = np.array([x - w / 2, y - h / 2])
            for landmarks_xy_pair in left_eye_landmarks.values():
                cv2.circle(
                    image_facial_landmarks_plot,
                    (landmarks_xy_pair + offset).astype(int),
                    radius=0,
                    color=(0, 0, 255),
                    thickness=-1,
                    # lineType=cv2.LINE_AA,
                )  # color: BGR
            for landmarks_xy_pair in right_eye_landmarks.values():
                cv2.circle(
                    image_facial_landmarks_plot,
                    (landmarks_xy_pair + offset).astype(int),
                    radius=0,
                    color=(0, 0, 255),
                    thickness=-1,
                    # lineType=cv2.LINE_AA,
                )  # color: BGR

        return image_facial_landmarks_plot

    def _detect_blink(self, facial_landmarks, image):
        def normalized_to_pixel_coordinates(x, y, image_height, image_width):
            x_px = min(math.floor(x * image_width), image_width - 1)
            y_px = min(math.floor(y * image_height), image_height - 1)
            return x_px, y_px

        image_height, image_width = image.shape
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

    def _align_face(self, image_align, facial_landmarks):
        image_height, image_width = image_align.shape[:2]
        eye_left = facial_landmarks[0]
        eye_right = facial_landmarks[1]
        angle = float(
            np.degrees(
                np.arctan2(eye_right[1] - eye_left[1], eye_right[0] - eye_left[0])
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
        # # Rotate facial area
        # x, y, w, h = detected_face["xywh"]
        # direction = 1 if angle >= 0 else -1
        # angle = abs(angle) % 360
        # if angle == 0:
        #     rotated_x1 = max(0, int(x - w / 2))
        #     rotated_y1 = max(0, int(y - h / 2))
        #     rotated_x2 = min(int(x + w / 2), int(image_width - 1))
        #     rotated_y2 = min(int(y + h / 2), int(image_height - 1))
        # else:
        #     angle = math.radians(angle)
        #     # Translate the facial area to the center of the image
        #     x_image_center_origin = x - image_width / 2
        #     y_image_center_origin = y - image_height / 2
        #     # Rotate the facial area
        #     x_new = x_image_center_origin * np.cos(
        #         angle
        #     ) + y_image_center_origin * direction * np.sin(angle)
        #     y_new = -x_image_center_origin * direction * np.sin(
        #         angle
        #     ) + y_image_center_origin * np.cos(angle)
        #     # Translate the facial area back to the original position
        #     x_new = x_new + image_width / 2
        #     y_new = y_new + image_height / 2

        #     rotated_x1 = max(0, int(x_new - w / 2))
        #     rotated_y1 = max(0, int(y_new - h / 2))
        #     rotated_x2 = min(int(x_new + w / 2), int(image_width - 1))
        #     rotated_y2 = min(int(y_new + h / 2), int(image_height - 1))
        # aligned_face_image = image_aligned[rotated_y1:rotated_y2, rotated_x1:rotated_x2]

        # # if self.debug_face_align:
        # #     for aligned_face_image in aligned_face_images:
        # #         self._image_show(aligned_face_image, "aligned_face_image", 2000)

        # return aligned_face_images

    def _detect_facial_landmarks(self, image_fld, num_list=[]):
        # 这个模型里包含了目标检测,所以重复检测了,但是这部分暂时去不掉.
        def normalized_to_pixel_coordinates(x, y, image_height, image_width):
            x_px = min(math.floor(x * image_width), image_width - 1)
            y_px = min(math.floor(y * image_height), image_height - 1)
            return x_px, y_px

        image_height, image_width = image_fld.shape[0], image_fld.shape[1]
        # image_fld_display = copy.deepcopy(image_fld)
        image_fld = mp_image.Image(
            image_format=mp_image_frame.ImageFormat.SRGB,
            data=image_fld.astype(np.uint8),
        )
        landmarks_detect_result = self.facial_landmarks_detector.detect(image_fld)
        if (
            len(landmarks_detect_result.face_landmarks) == 0
            or len(landmarks_detect_result.face_landmarks) > 1
        ):
            # return None, image_fld_display
            return None
        facial_landmarks = landmarks_detect_result.face_landmarks[0]
        landmarks_detect_result = []
        for num in num_list:
            landmark = normalized_to_pixel_coordinates(
                facial_landmarks[num].x,
                facial_landmarks[num].y,
                image_height,
                image_width,
            )
            landmarks_detect_result.append(landmark)
        return landmarks_detect_result, facial_landmarks

    def _facial_landmarks_detector_init(self):
        # 3FabRec Model can only detect one face in an image, namely only one face can appear in an image.
        base_options = mp_python.BaseOptions(
            model_asset_path=os.path.join(
                os.path.dirname(__file__),
                "google_mediapipe/face_landmarker_v2_with_blendshapes.task",
            )
        )
        options = mp_vision.FaceLandmarkerOptions(base_options=base_options)
        facial_landmarks_detector = mp_vision.FaceLandmarker.create_from_options(
            options
        )
        return facial_landmarks_detector


def main():
    image_path = ""
    image = cv2.imread(image_path)
    image_processor = ImageProcess()
    gaze_direction = image_processor.pipeline(image)
    if gaze_direction is not None:
        eye_gaze_yaw_angle, eye_gaze_pitch_angle = gaze_direction
        print(eye_gaze_yaw_angle, eye_gaze_pitch_angle)


if __name__ == "__main__":
    main()
