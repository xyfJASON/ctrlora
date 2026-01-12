import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2
import tqdm
import einops
import argparse
from functools import partial

import torch
from torch.utils.data import Dataset, DataLoader
from torchmetrics import MeanSquaredError
from torchmetrics.image import LearnedPerceptualImagePatchSimilarity
from torchmetrics.image import PeakSignalNoiseRatio
from torchmetrics.image import StructuralSimilarityIndexMeasure
from torchmetrics.multimodal import CLIPScore

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

        gt_control = cv2.imread(os.path.join(self.sample_dir, 'control', f'{item}.png'))[..., ::-1]  # np.uint8, [0, 255]
        gt_control = resize_image(HWC3(gt_control), 512)  # np.uint8, [0, 255]
        gt_control = torch.from_numpy(gt_control).float() / 255.0
        gt_control = einops.rearrange(gt_control, 'h w c -> c h w')  # torch.float32, [0, 1]

        prompt = self.prompts[item]

        return sample, gt_control, prompt


def main():
    args = get_parser().parse_args()

    dataset = SampleDataset(args.sample_dir)
    dataloader = DataLoader(dataset, batch_size=128, shuffle=False, num_workers=8, pin_memory=True)
    print('Dataset size:', len(dataset))
    print()

    mse = MeanSquaredError().cuda()
    lpips = LearnedPerceptualImagePatchSimilarity(normalize=True).cuda()
    psnr = PeakSignalNoiseRatio(data_range=(0, 1)).cuda()
    ssim = StructuralSimilarityIndexMeasure(data_range=(0, 1)).cuda()
    clip_sc = CLIPScore(model_name_or_path="openai/clip-vit-large-patch14").cuda()

    detector = LineartDetector()
    with open('./scripts/evaluate_lineart_is_coarse.txt', 'r') as f:
        is_coarse = f.readlines()
    is_coarse = [True if line.strip() == 'True' else False for line in is_coarse]
    assert len(is_coarse) == len(dataset)

    i = 0
    with torch.no_grad():
        for sample, gt_control, prompt in tqdm.tqdm(dataloader):

            control = []
            for s in sample:
                c = detector(((s * 255).to(torch.uint8).permute(1, 2, 0).numpy()), coarse=is_coarse[i])
                c = resize_image(HWC3(c), 512)  # np.uint8, [0, 255]
                c = torch.from_numpy(c).float() / 255.0
                c = einops.rearrange(c, 'h w c -> c h w')  # torch.float32, [0, 1]
                control.append(c)
                i += 1
            control = torch.stack(control, dim=0)

            sample, control, gt_control = sample.cuda(), control.cuda(), gt_control.cuda()

            mse.update(control, gt_control)
            lpips.update(control, gt_control)
            psnr.update(control, gt_control)
            ssim.update(control, gt_control)
            clip_sc.update(sample, prompt)

    print(f'MSE: {mse.compute().item():.4f}')
    print(f'LPIPS: {lpips.compute().item():.4f}')
    print(f'PSNR: {psnr.compute().item():.4f}')
    print(f'SSIM: {ssim.compute().item():.4f}')
    print(f'CLIP SCORE: {clip_sc.compute().item():.4f}')


if __name__ == "__main__":
    main()
