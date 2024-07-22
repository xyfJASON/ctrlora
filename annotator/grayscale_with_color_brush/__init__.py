import cv2
import math
import numpy as np
from PIL import Image, ImageDraw


class GrayscaleWithColorBrushConverter:
    def __init__(
            self,
            brush_num: tuple[int, int] = (5, 9),
            brush_n_vertex: tuple[int, int] = (4, 18),
            brush_mean_angle: float = 2 * math.pi / 5,
            brush_angle_range: float = 2 * math.pi / 15,
            brush_width_ratio: tuple[float, float] = (0.02, 0.1),
    ):
        self.brush_num = brush_num
        self.brush_n_vertex = brush_n_vertex
        self.brush_mean_angle = brush_mean_angle
        self.brush_angle_range = brush_angle_range
        self.brush_width_ratio = brush_width_ratio

    def __call__(self, img: np.ndarray):
        if GrayscaleWithColorBrushConverter.is_grayscale(img):
            return
        img_prompt = np.stack([cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)] * 3, axis=-1)
        mask = self.sample_brushes(img.shape[0], img.shape[1])
        img_prompt[mask] = img[mask]
        img_yuv = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
        img_prompt_yuv = cv2.cvtColor(img_prompt, cv2.COLOR_RGB2YUV)
        img_prompt_yuv[:, :, 0] = img_yuv[:, :, 0]
        img_prompt = cv2.cvtColor(img_prompt_yuv, cv2.COLOR_YUV2RGB)
        return img_prompt

    @staticmethod
    def is_grayscale(img, threshold=5):
        return np.mean(np.std(img, axis=-1) < threshold) > 0.95

    def sample_brushes(self, H: int, W: int):
        min_num, max_num = self.brush_num
        min_n_vertex, max_n_vertex = self.brush_n_vertex
        min_width = int(self.brush_width_ratio[0] * min(H, W))
        max_width = int(self.brush_width_ratio[1] * min(H, W))
        n_brush = np.random.randint(min_num, max_num + 1)
        average_radius = math.sqrt(H * H + W * W) / 8
        mask = Image.new('L', (W, H), 255)
        for i in range(n_brush):
            n_vertex = np.random.randint(min_n_vertex, max_n_vertex + 1)
            width = np.random.randint(min_width, max_width + 1)
            min_angle = self.brush_mean_angle - np.random.rand() * self.brush_angle_range
            max_angle = self.brush_mean_angle + np.random.rand() * self.brush_angle_range
            vertex = [(np.random.randint(0, W), np.random.randint(0, H))]
            for j in range(n_vertex):
                angle = np.random.rand() * (max_angle - min_angle) + min_angle
                if j % 2 == 0:
                    angle = 2 * math.pi - angle
                r = np.clip(
                    np.random.normal(loc=average_radius, scale=average_radius // 2),
                    0, 2 * average_radius,
                )
                new_x = min(max(vertex[-1][0] + r * math.cos(angle), 0), W)
                new_y = min(max(vertex[-1][1] + r * math.sin(angle), 0), H)
                vertex.append((new_x, new_y))
            draw = ImageDraw.Draw(mask)
            draw.line(vertex, fill=0, width=width)
            for v in vertex:
                draw.ellipse((v[0] - width // 2,
                              v[1] - width // 2,
                              v[0] + width // 2,
                              v[1] + width // 2), fill=0)
            if np.random.rand() > 0.5:
                mask = mask.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
            if np.random.rand() > 0.5:
                mask = mask.transpose(Image.Transpose.FLIP_TOP_BOTTOM)
        if np.random.rand() > 0.5:
            mask = mask.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
        if np.random.rand() > 0.5:
            mask = mask.transpose(Image.Transpose.FLIP_TOP_BOTTOM)
        mask = np.array(mask)
        mask = mask < 128
        return mask
