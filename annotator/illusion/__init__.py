"""
Adapted from https://gitlab.com/-/snippets/3611640
Reference: https://civitai.com/models/137638/controlnet-mysee-light-and-dark-squint-illusions-hidden-symbols-subliminal-text-qr-codes
"""

import cv2
import numpy as np


class IllusionConverter:

    def __call__(self, img):
        img_grayscale = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        img_grayscale = img[:, :, 2]
        # img_grayscale = cv2.GaussianBlur(img_grayscale, (15, 15), 0)

        threshold = 256 // 3
        remap = np.zeros_like(img_grayscale)
        remap[np.where(img_grayscale < threshold)] = 0
        remap[np.where((img_grayscale >= threshold) & (img_grayscale <= 255 - threshold))] = 127
        remap[np.where(img_grayscale > 255 - threshold)] = 255

        large_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
        small_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))

        img_out = cv2.morphologyEx(remap, cv2.MORPH_CLOSE, large_kernel)
        img_out = cv2.morphologyEx(img_out, cv2.MORPH_OPEN, small_kernel)

        img_out = np.stack([img_out] * 3, axis=-1).astype('uint8')
        return img_out
