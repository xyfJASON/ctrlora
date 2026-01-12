import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2
import tqdm
import einops
import argparse

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from annotator.util import HWC3, resize_image
from annotator.lineart import LineartDetector


def get_parser():
    parser = argparse.ArgumentParser(description="args")
    parser.add_argument("--sample_dir", type=str, required=True, help='path to the sample directory')
    return parser


class SampleDataset(Dataset):
    def __init__(self, sample_dir: str):
        self.sample_dir = sample_dir

        with open(os.path.join(self.sample_dir, 'prompt.txt'), 'r') as f:
            self.prompts = f.readlines()

    def __len__(self):
        return len(self.prompts)

    def __getitem__(self, item):
        sample = cv2.imread(os.path.join(self.sample_dir, 'sample', f'{item}.png'))[..., ::-1]  # np.uint8, [0, 255]
        sample = torch.from_numpy(sample.copy()).float() / 255.0
        sample = einops.rearrange(sample, 'h w c -> c h w')  # torch.float32, [0, 1]

        img = cv2.imread(os.path.join(self.sample_dir, 'img', f'{item}.png'))[..., ::-1]  # np.uint8, [0, 255]
        img = torch.from_numpy(img.copy()).float() / 255.0
        img = einops.rearrange(img, 'h w c -> c h w')  # torch.float32, [0, 1]

        gt_control = cv2.imread(os.path.join(self.sample_dir, 'control', f'{item}.png'))[..., ::-1]  # np.uint8, [0, 255]
        gt_control = resize_image(HWC3(gt_control), 512)  # np.uint8, [0, 255]
        gt_control = torch.from_numpy(gt_control).float() / 255.0
        gt_control = einops.rearrange(gt_control, 'h w c -> c h w')  # torch.float32, [0, 1]

        prompt = self.prompts[item]

        return sample, img, gt_control, prompt


def main():
    args = get_parser().parse_args()

    dataset = SampleDataset(args.sample_dir)
    dataloader = DataLoader(dataset, batch_size=128, shuffle=False, num_workers=8, pin_memory=True)
    print('Dataset size:', len(dataset))
    print()

    detector = LineartDetector()
    is_coarse = []

    with torch.no_grad():
        for sample, img, gt_control, prompt in tqdm.tqdm(dataloader):

            for im, gc in zip(img, gt_control):
                c_false = detector(((im * 255).to(torch.uint8).permute(1, 2, 0).numpy()), coarse=False)
                c_false = resize_image(HWC3(c_false), 512)  # np.uint8, [0, 255]
                c_false = torch.from_numpy(c_false).float() / 255.0
                c_false = einops.rearrange(c_false, 'h w c -> c h w')  # torch.float32, [0, 1]
                
                c_true = detector(((im * 255).to(torch.uint8).permute(1, 2, 0).numpy()), coarse=True)
                c_true = resize_image(HWC3(c_true), 512)  # np.uint8, [0, 255]
                c_true = torch.from_numpy(c_true).float() / 255.0
                c_true = einops.rearrange(c_true, 'h w c -> c h w')  # torch.float32, [0, 1]

                diff_false = F.mse_loss(c_false, gc)
                diff_true = F.mse_loss(c_true, gc)
                is_coarse.append(str(diff_true.item() < diff_false.item()))

    with open('./scripts/evaluate_lineart_is_coarse.txt', 'w') as f:
        f.write('\n'.join(is_coarse))


if __name__ == "__main__":
    main()
