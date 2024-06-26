"""
Adapted from https://github.com/salesforce/UniControl/blob/main/annotator/grayscale/__init__.py
"""

from skimage import color


class GrayscaleConverter:
    def __call__(self, img):
        return (color.rgb2gray(img) * 255.0).astype('ubyte')
