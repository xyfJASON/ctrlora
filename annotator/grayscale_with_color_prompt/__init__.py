import cv2
import numpy as np


# class GrayscaleWithColorPromptConverter:
#     def __call__(self, img, p=0.125, min_n_patches=4, max_patch_frac=0.1):
#         if GrayscaleWithColorPromptConverter.is_grayscale(img):
#             return
#
#         h, w = img.shape[0], img.shape[1]
#         min_patch_size = 2
#         max_patch_size = max(round(min(h, w) * max_patch_frac), min_patch_size)
#
#         img_prompt = np.stack([cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)] * 3, axis=-1)
#         n_patches = 0
#         while np.random.rand() < (1 - p) or n_patches < min_n_patches:
#             ps = np.random.choice(range(min_patch_size, max_patch_size + 1))
#             # y = int(np.clip(np.random.normal((h - ps + 1) / 2, (h - ps + 1) / 4), 0, h - ps))
#             # x = int(np.clip(np.random.normal((w - ps + 1) / 2, (w - ps + 1) / 4), 0, w - ps))
#             y = np.random.randint(h - ps + 1)
#             x = np.random.randint(w - ps + 1)
#             # img_prompt[y:y + ps, x:x + ps] = img_prompt[y:y + ps, x:x + ps] * 0.5 + np.mean(img[y:y + ps, x:x + ps], axis=(0, 1)) * 0.5
#             img_prompt[y:y + ps, x:x + ps] = img[y:y + ps, x:x + ps]
#             # img_prompt[y:y + ps, x:x + ps] = np.mean(img[y:y + ps, x:x + ps], axis=(0, 1))
#             n_patches += 1
#
#         img_prompt = img_prompt.astype('uint8')
#         return img_prompt
#
#     @staticmethod
#     def is_grayscale(img, threshold=5):
#         return np.mean(np.std(img, axis=-1) < threshold) > 0.95


class GrayscaleWithColorPromptConverter:
    def __call__(self, img, p=0.04, min_n_patches=5, radius_frac_range=(0.02, 0.05)):
        if GrayscaleWithColorPromptConverter.is_grayscale(img):
            return

        h, w = img.shape[0], img.shape[1]
        min_radius = int(min(h, w) * radius_frac_range[0])
        max_radius = int(min(h, w) * radius_frac_range[1])

        img_prompt = np.stack([cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)] * 3, axis=-1)
        n_patches = 0
        y, x = np.meshgrid(range(h), range(w), indexing='ij')
        while np.random.rand() < (1 - p) or n_patches < min_n_patches:
            r = np.random.choice(range(min_radius, max_radius + 1))
            cy, cx = np.random.randint(h), np.random.randint(w)
            mask = ((y - cy) ** 2 + (x - cx) ** 2) <= (r ** 2)

            img_prompt[mask] = np.mean(img[mask], axis=0).astype('uint8')

            img_yuv = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
            img_prompt_yuv = cv2.cvtColor(img_prompt, cv2.COLOR_RGB2YUV)
            img_prompt_yuv[:, :, 0] = img_yuv[:, :, 0]
            img_prompt = cv2.cvtColor(img_prompt_yuv, cv2.COLOR_YUV2RGB)

            n_patches += 1

        return img_prompt

    @staticmethod
    def is_grayscale(img, threshold=5):
        return np.mean(np.std(img, axis=-1) < threshold) > 0.95
