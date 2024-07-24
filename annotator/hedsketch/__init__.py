"""
Adapted from https://github.com/salesforce/UniControl/blob/main/app/gradio_hedsketch2image.py#L49-L64
"""

import cv2
import numpy as np

from ..hed import HEDdetector
from ..util import HWC3


class HEDSketchDetector:
    def __init__(self):
        self.hed = HEDdetector()

    def __call__(self, input_image, safe=False):
        edge = self.hed(input_image, safe=safe)
        edge = HWC3(edge)

        retry = 0
        cnt = 0
        while retry == 0:
            threshold_value = np.random.randint(110, 160)
            kernel_size = 3
            binary_image = cv2.threshold(edge, threshold_value, 255, cv2.THRESH_BINARY)[1]
            inverted_image = cv2.bitwise_not(binary_image)
            edge = cv2.GaussianBlur(inverted_image, (kernel_size, kernel_size), 0)
            if np.sum(edge < 5) > 0.005 * edge.shape[0] * edge.shape[1] or cnt == 5:
                retry = 1
            else:
                cnt += 1

        return edge
