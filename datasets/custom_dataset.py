import os
import cv2
import json
import numpy as np

from torch.utils.data import Dataset


class CustomDataset(Dataset):
    """Custom dataset.

    Organize your dataset in the following structure:

    root
    ├── prompt.json
    ├── source
    │   ├── 0000.jpg
    │   ├── 0001.jpg
    │   └── ...
    └── target
        ├── 0000.jpg
        ├── 0001.jpg
        └── ...

    Each line in `prompt.json` should be in the following format:
        {"source": "source/0000.jpg", "target": "target/0000.jpg", "prompt": "The quick brown fox jumps over the lazy dog."}

    """
    def __init__(self, root: str, drop_rate: float = 0.0):
        self.root = root
        self.drop_rate = drop_rate

        root = os.path.expanduser(root)
        if not os.path.isfile(os.path.join(root, 'prompt.json')):
            raise FileNotFoundError(f"{os.path.join(root, 'prompt.json')} not found.")
        if not os.path.isdir(os.path.join(root, 'source')):
            raise FileNotFoundError(f"{os.path.join(root, 'source')} not found.")
        if not os.path.isdir(os.path.join(root, 'target')):
            raise FileNotFoundError(f"{os.path.join(root, 'target')} not found.")

        self.data = []
        source_files = set(os.listdir(os.path.join(root, 'source')))
        target_files = set(os.listdir(os.path.join(root, 'target')))
        with open(os.path.join(root, 'prompt.json'), 'rt') as f:
            for line in f:
                data = json.loads(line)
                if data['source'].removeprefix('source/') not in source_files:
                    continue
                if data['target'].removeprefix('target/') not in target_files:
                    continue
                self.data.append(data)
        del source_files, target_files

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item: dict = self.data[idx]

        source_filename = item['source']
        target_filename = item['target']
        prompt = item['prompt']

        if np.random.rand() < self.drop_rate:
            prompt = ''

        source = cv2.imread(os.path.join(self.root, source_filename))
        target = cv2.imread(os.path.join(self.root, target_filename))

        # Do not forget that OpenCV read images in BGR order.
        source = cv2.cvtColor(source, cv2.COLOR_BGR2RGB)
        target = cv2.cvtColor(target, cv2.COLOR_BGR2RGB)

        # Normalize source images to [0, 1].
        source = source.astype(np.float32) / 255.0

        # Normalize target images to [-1, 1].
        target = (target.astype(np.float32) / 127.5) - 1.0

        return dict(jpg=target, txt=prompt, hint=source)


def _test():
    dataset = CustomDataset(root='./training/celebahq-mask')
    print(len(dataset))

    item = dataset[123]
    jpg = item['jpg']
    txt = item['txt']
    hint = item['hint']
    print(txt)
    print(jpg.shape)
    print(hint.shape)


if __name__ == '__main__':
    _test()
