import os
import sys

path = os.path.dirname(__file__)
sys.path.insert(0, path)

import copy
from typing import Union

import cv2
import deepface.modules.detection as deepface_detection
import numpy as np
import rospy
import torch
from cv_bridge import CvBridge
from l2cs import utils, vis
from sensor_msgs.msg import Image

# record video
enable_record_video = True
if enable_record_video:
    fourcc = cv2.VideoWriter.fourcc(*"XVID")
    fps = 30
    output_video_path = os.path.join(
        os.path.dirname(__file__), "output_video_color.avi"
    )
    img_width = 640
    img_height = 480
    video_writer = cv2.VideoWriter(
        output_video_path, fourcc, fps, (img_width, img_height)
    )


class L2CSNet:
    def __init__(self):
        # model init
        model_weights_path = os.path.join(
            os.path.dirname(__file__), "models/L2CSNet_gaze360.pkl"
        )
        self.gaze_direction_estimation_model = utils.getArch("ResNet50", 90)
        self.device = torch.device("cuda:0")
        self.gaze_direction_estimation_model.load_state_dict(
            torch.load(model_weights_path, map_location=self.device)
        )
        self.gaze_direction_estimation_model.to(self.device)
        self.gaze_direction_estimation_model.eval()
        # face detector init
        self.face_detector_model = deepface_detection.deepface_face_detector_model_init(
            detector_backend="yolov8"
        )
        # others
        self.enable_visualization = True
        self.softmax = torch.nn.Softmax(dim=1)
        self.idx_tensor = [idx for idx in range(90)]
        self.idx_tensor = torch.FloatTensor(self.idx_tensor).to(self.device)
        # ROS init
        self.bridge = CvBridge()
        rospy.Subscriber(
            "/camera/color/image_raw",
            Image,
            self._do_gaze_direction_estimate,
            queue_size=1,
        )

    def _do_gaze_direction_estimate(self, image_color):
        image_color = self.bridge.imgmsg_to_cv2(image_color, desired_encoding="bgr8")
        image_color_plot = copy.deepcopy(image_color)
        image_height, image_width, _ = image_color.shape
        faces_detect_result_xywh = self._detect_face(image_color)
        if len(faces_detect_result_xywh) != 0:
            face_images_resized = []
            bboxes_xyxy = []
            for face_detect_result_xywh in faces_detect_result_xywh:
                # crop face image
                x, y, w, h = face_detect_result_xywh
                x_min = int(max(0, int(x - w / 2)))
                y_min = int(max(0, int(y - h / 2)))
                x_max = int(min(image_width - 1, int(x + w / 2)))
                y_max = int(min(image_height - 1, int(y + h / 2)))
                bbox_xyxy = {"xy1": (x_min, y_min), "xy2": (x_max, y_max)}
                face_image = image_color[y_min:y_max, x_min:x_max]
                face_image = cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB)
                face_image_resized = cv2.resize(face_image, (224, 224))
                face_images_resized.append(face_image_resized)
                bboxes_xyxy.append(bbox_xyxy)
            pitch_pred_array, yaw_pred_array = self._predict_gaze(
                np.stack(face_images_resized)
            )
            if self.enable_visualization:
                for i, bbox_xyxy in enumerate(bboxes_xyxy):
                    # draw bbox
                    cv2.rectangle(
                        image_color_plot,
                        bbox_xyxy["xy1"],
                        bbox_xyxy["xy2"],
                        (0, 255, 0),
                        thickness=2,
                        lineType=cv2.LINE_AA,
                    )
                    # draw gaze direction estimation arrow
                    x_min = bbox_xyxy["xy1"][0]
                    y_min = bbox_xyxy["xy1"][1]
                    x_max = bbox_xyxy["xy2"][0]
                    y_max = bbox_xyxy["xy2"][1]
                    bbox_width = x_max - x_min
                    bbox_height = y_max - y_min
                    vis.draw_gaze(
                        x_min,
                        y_min,
                        bbox_width,
                        bbox_height,
                        image_color_plot,
                        (pitch_pred_array[i], yaw_pred_array[i]),
                        color=(0, 0, 255),
                    )
        else:
            print("No face detected in the image.")
        if enable_record_video:
            video_writer.write(image_color_plot)
        if self.enable_visualization:
            cv2.imshow("gaze_direction_estimate", image_color_plot)
            cv2.waitKey(1)

    def _detect_face(self, image_detect):
        faces_detect_result = self.face_detector_model.detect_faces(image_detect)
        faces_detect_result_xywh = []
        for face_detect_result in faces_detect_result:
            faces_detect_result_xywh.append(face_detect_result["xywh"])
        return faces_detect_result_xywh

    def _predict_gaze(self, frame: Union[np.ndarray, torch.Tensor]):
        # Prepare input
        if isinstance(frame, np.ndarray):
            img = utils.prep_input_numpy(frame, self.device)
        elif isinstance(frame, torch.Tensor):
            img = frame
        else:
            raise RuntimeError("Invalid dtype for input")

        # Predict
        gaze_pitch, gaze_yaw = self.gaze_direction_estimation_model(img)
        pitch_predicted = self.softmax(gaze_pitch)
        yaw_predicted = self.softmax(gaze_yaw)

        # Get continuous predictions in degrees.
        pitch_predicted = (
            torch.sum(pitch_predicted.data * self.idx_tensor, dim=1) * 4 - 180
        )
        yaw_predicted = torch.sum(yaw_predicted.data * self.idx_tensor, dim=1) * 4 - 180

        pitch_predicted = pitch_predicted.cpu().detach().numpy() * np.pi / 180.0
        yaw_predicted = yaw_predicted.cpu().detach().numpy() * np.pi / 180.0

        return pitch_predicted, yaw_predicted


def main():
    rospy.init_node("gaze_direction_estimate")
    L2CSNet()
    if enable_record_video:
        try:
            rospy.spin()
        finally:
            video_writer.release()
    else:
        rospy.spin()


if __name__ == "__main__":
    main()
