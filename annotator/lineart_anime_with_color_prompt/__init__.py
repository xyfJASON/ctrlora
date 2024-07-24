import numpy as np

from ..lineart_anime import LineartAnimeDetector
from ..util import HWC3


class LineartAnimeWithColorPromptConverter:
    def __init__(self):
        self.detector = LineartAnimeDetector()

    def __call__(self, img, p=0.10, min_n_patches=5, radius_frac_range=(0.02, 0.04)):
        if LineartAnimeWithColorPromptConverter.is_grayscale(img):
            return

        h, w = img.shape[0], img.shape[1]
        min_radius = int(min(h, w) * radius_frac_range[0])
        max_radius = int(min(h, w) * radius_frac_range[1])

        img_prompt = HWC3(self.detector(img))
        n_patches = 0
        y, x = np.meshgrid(range(h), range(w), indexing='ij')
        while np.random.rand() < (1 - p) or n_patches < min_n_patches:
            r = np.random.choice(range(min_radius, max_radius + 1))
            cy, cx = np.random.randint(h), np.random.randint(w)
            mask = ((y - cy) ** 2 + (x - cx) ** 2) <= (r ** 2)
            img_prompt[mask] = np.mean(img[mask], axis=0).astype('uint8')
            n_patches += 1

        return img_prompt

    @staticmethod
    def is_grayscale(img, threshold=5):
        return np.mean(np.std(img, axis=-1) < threshold) > 0.95
