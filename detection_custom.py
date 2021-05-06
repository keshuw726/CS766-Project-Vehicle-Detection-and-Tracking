#================================================================
#
#   File name   : detection_custom.py
#   Author      : PyLessons
#   Created date: 2020-09-17
#   Website     : https://pylessons.com/
#   GitHub      : https://github.com/pythonlessons/TensorFlow-2.x-YOLOv3
#   Description : object detection image and video example
#
#================================================================
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import cv2
import numpy as np
import tensorflow as tf
from yolov3.utils import detect_image, detect_realtime, detect_video, Load_Yolo_model, detect_video_realtime_mp
from yolov3.configs import *

image_path   = "./IMAGES/plate_2.jpg"
video_path   = "./IMAGES/test.mp4"
video_path   = "./IMAGES/cars.mp4"
image_path   = "./IMAGES/000138.png"
image_path   = "./IMAGES/000517.png"
image_path   = "./IMAGES/000808.png"
image_path   = "./IMAGES/001054.png"
video_path   = "./IMAGES/cars_video_1.mp4"

yolo = Load_Yolo_model()
# detect_image(yolo, image_path, "./IMAGES/plate_1_detect.jpg", input_size=YOLO_INPUT_SIZE, show=True, CLASSES=TRAIN_CLASSES, rectangle_colors=(255,0,0))
detect_image(yolo, image_path, "./IMAGES/001054_pred_.jpg", input_size=YOLO_INPUT_SIZE, show=True, CLASSES=TRAIN_CLASSES, rectangle_colors=(255,0,0))
# detect_video(yolo, video_path, './IMAGES/cars_video_1_detection.mp4', input_size=YOLO_INPUT_SIZE, show=True, CLASSES=TRAIN_CLASSES, score_threshold=0.25, iou_threshold=0.3, rectangle_colors=(255,0,0))
#detect_realtime(yolo, '', input_size=YOLO_INPUT_SIZE, show=True, CLASSES=TRAIN_CLASSES, rectangle_colors=(255, 0, 0))

#detect_video_realtime_mp(video_path, "Output.mp4", input_size=YOLO_INPUT_SIZE, show=True, CLASSES=TRAIN_CLASSES, rectangle_colors=(255,0,0), realtime=False)
