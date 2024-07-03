import cv2
import numpy as np
from scipy.spatial import KDTree
from sklearn.cluster import KMeans


class Pixelater:

    def __call__(self, img, palette=None, n_colors=16, scale=16, min_size=32, down_interpolation=cv2.INTER_NEAREST):
        ori_height, ori_width = img.shape[:2]
        new_height = max(ori_height // scale, min_size)
        new_width = max(ori_width // scale, min_size)
        img = cv2.resize(img, (new_width, new_height), interpolation=down_interpolation)

        if palette is None:
            palette = Pixelater.get_palette(img, n_colors)

        pixels = img.reshape((-1, 3))
        tree = KDTree(palette)
        _, indexes = tree.query(pixels)
        new_pixels = palette[indexes]

        new_img = new_pixels.reshape((new_height, new_width, 3))
        new_img = cv2.resize(new_img, (ori_width, ori_height), interpolation=cv2.INTER_NEAREST)

        return new_img

    @staticmethod
    def get_palette(img, n_colors):
        pixels = img.reshape((-1, 3))

        kmeans = KMeans(n_clusters=n_colors)
        kmeans.fit(pixels)

        centers = kmeans.cluster_centers_
        palette = np.clip(np.rint(centers), 0, 255).astype(np.uint8)

        return palette
