"""
Requirements: `pip install torchmetrics[image]`
"""

import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2
import tqdm
import einops
import argparse

import torch
from torch.utils.data import Dataset, DataLoader
from torchmetrics import MeanSquaredError
from torchmetrics.image import LearnedPerceptualImagePatchSimilarity
from torchmetrics.image import PeakSignalNoiseRatio
from torchmetrics.image import StructuralSimilarityIndexMeasure
from torchmetrics.multimodal import CLIPScore


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

        control = cv2.imread(os.path.join(self.sample_dir, 'control', f'{item}.png'))[..., ::-1]  # np.uint8, [0, 255]
        control = torch.from_numpy(control.copy()).float() / 255.0
        control = einops.rearrange(control, 'h w c -> c h w')  # torch.float32, [0, 1]

        prompt = self.prompts[item]

        return sample, img, control, prompt


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

    control_mse = MeanSquaredError().cuda()
    control_lpips = LearnedPerceptualImagePatchSimilarity(normalize=True).cuda()
    control_psnr = PeakSignalNoiseRatio(data_range=(0, 1)).cuda()
    control_ssim = StructuralSimilarityIndexMeasure(data_range=(0, 1)).cuda()
    control_clip_sc = CLIPScore(model_name_or_path="openai/clip-vit-large-patch14").cuda()

    with torch.no_grad():
        for sample, img, control, prompt in tqdm.tqdm(dataloader):
            sample, img, control = sample.cuda(), img.cuda(), control.cuda()

            mse.update(sample, img)
            lpips.update(sample, img)
            psnr.update(sample, img)
            ssim.update(sample, img)
            clip_sc.update(sample, prompt)

            control_mse.update(control, img)
            control_lpips.update(control, img)
            control_psnr.update(control, img)
            control_ssim.update(control, img)
            control_clip_sc.update(control, prompt)

    print(f'MSE: {mse.compute().item():.4f}')
    print(f'LPIPS: {lpips.compute().item():.4f}')
    print(f'PSNR: {psnr.compute().item():.4f}')
    print(f'SSIM: {ssim.compute().item():.4f}')
    print(f'CLIP SCORE: {clip_sc.compute().item():.4f}')
    print()
    print(f'CONTROL MSE: {control_mse.compute().item():.4f}')
    print(f'CONTROL LPIPS: {control_lpips.compute().item():.4f}')
    print(f'CONTROL PSNR: {control_psnr.compute().item():.4f}')
    print(f'CONTROL SSIM: {control_ssim.compute().item():.4f}')
    print(f'CONTROL CLIP SCORE: {control_clip_sc.compute().item():.4f}')


if __name__ == "__main__":
    main()
