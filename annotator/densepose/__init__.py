"""
Adapted from https://github.com/Flode-Labs/vid2densepose/blob/main/main.py
"""
import os
import urllib.request

import cv2
import numpy as np
import torch
from densepose import add_densepose_config
from densepose.vis.densepose_results import DensePoseResultsFineSegmentationVisualizer as Visualizer
from densepose.vis.extractor import DensePoseResultExtractor
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor


class DenseposeDetector:

    def __init__(self):
        cfg = get_cfg()
        add_densepose_config(cfg)
        # cfg.merge_from_file(os.path.join(os.path.dirname(__file__), "densepose_rcnn_R_50_FPN_s1x.yaml"))
        cfg.merge_from_file(os.path.join(os.path.dirname(__file__), "densepose_rcnn_R_101_FPN_DL_s1x.yaml"))
        # cfg.MODEL.WEIGHTS = "./annotator/ckpts/model_final_162be9.pkl"
        cfg.MODEL.WEIGHTS = "./annotator/ckpts/model_final_844d15.pkl"
        if not os.path.exists(cfg.MODEL.WEIGHTS):
            # urllib.request.urlretrieve("https://dl.fbaipublicfiles.com/densepose/densepose_rcnn_R_50_FPN_s1x/165712039/model_final_162be9.pkl", cfg.MODEL.WEIGHTS)
            urllib.request.urlretrieve("https://dl.fbaipublicfiles.com/densepose/densepose_rcnn_R_101_FPN_DL_s1x/165712116/model_final_844d15.pkl", cfg.MODEL.WEIGHTS)
        cfg.MODEL.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
        self.predictor = DefaultPredictor(cfg)

    def __call__(self, img):
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        with torch.no_grad():
            outputs = self.predictor(img)["instances"]
        results = DensePoseResultExtractor()(outputs)
        cmap = cv2.COLORMAP_VIRIDIS
        arr = cv2.applyColorMap(np.zeros((img.shape[0], img.shape[1]), dtype=np.uint8), cmap)
        seg = Visualizer(alpha=1, cmap=cmap).visualize(arr, results)

        seg = cv2.cvtColor(seg, cv2.COLOR_BGR2RGB)
        return seg
