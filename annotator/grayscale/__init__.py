import cv2
import numpy as np


class GrayscaleConverter:
    def __call__(self, img):
        return np.stack([cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)] * 3, axis=-1).astype('uint8')
