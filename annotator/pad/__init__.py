import cv2


class Padder:
    def __call__(self, img, top_ratio, bottom_ratio, left_ratio, right_ratio):
        height, width = img.shape[:2]
        padded_img = cv2.copyMakeBorder(
            img,
            top=int(height * top_ratio),
            bottom=int(height * bottom_ratio),
            left=int(width * left_ratio),
            right=int(width * right_ratio),
            borderType=cv2.BORDER_CONSTANT,
            value=[0, 0, 0]
        )
        return padded_img
