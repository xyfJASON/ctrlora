import os
import numpy as np
from PIL import Image
from typing import List

from pycocotools.coco import COCO as _COCO
from torch.utils.data import Dataset


class COCO(Dataset):
    def __init__(self, root: str, split: str, cond: str, drop_rate: float = 0.0):
        self.root = os.path.expanduser(root)
        self.split = split
        self.cond = cond
        self.drop_rate = drop_rate
        if not os.path.isdir(os.path.join(root, f'{split}2017-resized')):
            raise NotADirectoryError(f"Directory not found: {os.path.join(root, f'{split}2017-resized')}")
        if not os.path.isdir(os.path.join(root, f'{split}2017-resized-{cond}')):
            raise NotADirectoryError(f"Directory not found: {os.path.join(root, f'{split}2017-resized-{cond}')}")

        self.coco = _COCO(os.path.join(root, f'annotations/captions_{split}2017.json'))
        self.ids = list(sorted(self.coco.imgs.keys()))

    def _load_image(self, idx: int) -> Image.Image:
        path = self.coco.loadImgs(idx)[0]["file_name"]
        return Image.open(os.path.join(self.root, f'{self.split}2017-resized', path)).convert("RGB")

    def _load_target(self, idx: int) -> List[str]:
        anns = self.coco.loadAnns(self.coco.getAnnIds(idx))
        return [ann["caption"] for ann in anns]  # noqa

    def _load_cond(self, idx: int) -> Image.Image:
        path = self.coco.loadImgs(idx)[0]["file_name"]
        return Image.open(os.path.join(self.root, f'{self.split}2017-resized-{self.cond}', path)).convert("RGB")

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, index: int):
        idx = self.ids[index]
        image = self._load_image(idx)       # PIL Image
        target = self._load_target(idx)     # a list of strings
        cond_image = self._load_cond(idx)   # PIL Image

        # Normalize image to [-1, 1].
        image = np.array(image)             # np.uint8, [H, W, C]
        image = (image.astype(np.float32) / 127.5) - 1.0

        # Normalize cond_image to [0, 1].
        if cond_image is not None:
            cond_image = np.array(cond_image)   # np.uint8, [H, W, C]
            cond_image = cond_image.astype(np.float32) / 255.0

        if np.random.rand() < self.drop_rate:
            target[0] = ''
        return dict(jpg=image, txt=target[0], hint=cond_image, task=self.cond)


if __name__ == "__main__":
    dataset = COCO(root="./data/coco-tmp", split="train", cond="jpeg")
    print(dataset[0].keys())
    print(type(dataset[0]['jpg']), dataset[0]['jpg'].shape, dataset[0]['jpg'].dtype, dataset[0]['jpg'].min(), dataset[0]['jpg'].max())
    print(type(dataset[0]['hint']), dataset[0]['hint'].shape, dataset[0]['hint'].dtype, dataset[0]['hint'].min(), dataset[0]['hint'].max())
    print(dataset[0]['txt'])
    print(dataset[0]['task'])
