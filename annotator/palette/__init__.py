import cv2


class PaletteDetector:
    def __call__(self, img):
        size = img.shape[:2]
        # 64x bicubic downsampling
        img = cv2.resize(img, (size[1] // 64, size[0] // 64), interpolation=cv2.INTER_CUBIC)
        # 64x nearest upsampling
        img = cv2.resize(img, (size[1], size[0]), interpolation=cv2.INTER_NEAREST)
        return img
