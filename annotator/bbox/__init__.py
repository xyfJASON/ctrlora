"""Adapted from https://github.com/salesforce/UniControl/blob/main/app/gradio_bbox2image.py."""

import os
import shutil

import cvlib as cv
import numpy as np

from annotator.util import HWC3, annotator_ckpts_path


color_dict = {
    'background': (0, 0, 100),
    'person': (255, 0, 0),
    'bicycle': (0, 255, 0),
    'car': (0, 0, 255),
    'motorcycle': (255, 255, 0),
    'airplane': (255, 0, 255),
    'bus': (0, 255, 255),
    'train': (128, 128, 0),
    'truck': (128, 0, 128),
    'boat': (0, 128, 128),
    'traffic light': (128, 128, 128),
    'fire hydrant': (64, 0, 0),
    'stop sign': (0, 64, 0),
    'parking meter': (0, 0, 64),
    'bench': (64, 64, 0),
    'bird': (64, 0, 64),
    'cat': (0, 64, 64),
    'dog': (192, 192, 192),
    'horse': (32, 32, 32),
    'sheep': (96, 96, 96),
    'cow': (160, 160, 160),
    'elephant': (224, 224, 224),
    'bear': (32, 0, 0),
    'zebra': (0, 32, 0),
    'giraffe': (0, 0, 32),
    'backpack': (32, 32, 0),
    'umbrella': (32, 0, 32),
    'handbag': (0, 32, 32),
    'tie': (96, 0, 0),
    'suitcase': (0, 96, 0),
    'frisbee': (0, 0, 96),
    'skis': (96, 96, 0),
    'snowboard': (96, 0, 96),
    'sports ball': (0, 96, 96),
    'kite': (160, 0, 0),
    'baseball bat': (0, 160, 0),
    'baseball glove': (0, 0, 160),
    'skateboard': (160, 160, 0),
    'surfboard': (160, 0, 160),
    'tennis racket': (0, 160, 160),
    'bottle': (224, 0, 0),
    'wine glass': (0, 224, 0),
    'cup': (0, 0, 224),
    'fork': (224, 224, 0),
    'knife': (224, 0, 224),
    'spoon': (0, 224, 224),
    'bowl': (64, 64, 64),
    'banana': (128, 64, 64),
    'apple': (64, 128, 64),
    'sandwich': (64, 64, 128),
    'orange': (128, 128, 64),
    'broccoli': (128, 64, 128),
    'carrot': (64, 128, 128),
    'hot dog': (192, 64, 64),
    'pizza': (64, 192, 64),
    'donut': (64, 64, 192),
    'cake': (192, 192, 64),
    'chair': (192, 64, 192),
    'couch': (64, 192, 192),
    'potted plant': (96, 32, 32),
    'bed': (32, 96, 32),
    'dining table': (32, 32, 96),
    'toilet': (96, 96, 32),
    'tv': (96, 32, 96),
    'laptop': (32, 96, 96),
    'mouse': (160, 32, 32),
    'remote': (32, 160, 32),
    'keyboard': (32, 32, 160),
    'cell phone': (160, 160, 32),
    'microwave': (160, 32, 160),
    'oven': (32, 160, 160),
    'toaster': (224, 32, 32),
    'sink': (32, 224, 32),
    'refrigerator': (32, 32, 224),
    'book': (224, 224, 32),
    'clock': (224, 32, 224),
    'vase': (32, 224, 224),
    'scissors': (64, 96, 96),
    'teddy bear': (96, 64, 96),
    'hair drier': (96, 96, 64),
    'toothbrush': (160, 96, 96)
}


class BBoxDetector:

    def __init__(self):
        remote_model_path = "https://github.com/AlexeyAB/darknet/releases/download/darknet_yolo_v3_optimal/yolov4.weights"
        modelpath = os.path.join(annotator_ckpts_path, "yolov4.weights")
        if not os.path.exists(modelpath):
            from basicsr.utils.download_util import load_file_from_url
            load_file_from_url(remote_model_path, model_dir=annotator_ckpts_path)

        yolov3_classes_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'yolov3_classes.txt')
        yolov4_cfg_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'yolov4.cfg')

        cvlib_yolo_directory = os.path.expanduser('~/.cvlib/object_detection/yolo/yolov3')
        os.makedirs(cvlib_yolo_directory, exist_ok=True)
        shutil.copy(yolov4_cfg_path, cvlib_yolo_directory)
        shutil.copy(yolov3_classes_path, cvlib_yolo_directory)
        shutil.copy(modelpath, cvlib_yolo_directory)

    def __call__(self, input_image, confidence=0.4, nms_thresh=0.5):
        input_image = HWC3(input_image)
        bbox, label, conf = cv.detect_common_objects(input_image, confidence=confidence, nms_thresh=nms_thresh)
        mask = np.zeros((input_image.shape), np.uint8)
        if len(bbox) > 0:
            order_area = np.zeros(len(bbox))
            area_all = 0
            for idx_mask, box in enumerate(bbox):
                x_1, y_1, x_2, y_2 = box

                x_1 = 0 if x_1 < 0 else x_1
                y_1 = 0 if y_1 < 0 else y_1
                x_2 = input_image.shape[1] if x_2 < 0 else x_2
                y_2 = input_image.shape[0] if y_2 < 0 else y_2

                area = (x_2 - x_1) * (y_2 - y_1)
                order_area[idx_mask] = area
                area_all += area
            ordered_area = np.argsort(-order_area)

            for idx_mask in ordered_area:
                box = bbox[idx_mask]
                x_1, y_1, x_2, y_2 = box
                x_1 = 0 if x_1 < 0 else x_1
                y_1 = 0 if y_1 < 0 else y_1
                x_2 = input_image.shape[1] if x_2 < 0 else x_2
                y_2 = input_image.shape[0] if y_2 < 0 else y_2

                mask[y_1:y_2, x_1:x_2, :] = color_dict[label[idx_mask]]

        return mask
