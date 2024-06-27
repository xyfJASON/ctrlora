import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from share import *

import cv2
import einops
import argparse

import torch
import numpy as np
from torch.utils.data import Subset

from annotator.util import HWC3, resize_image
from cldm.model import create_model, load_state_dict
from cldm.ddim_hacked import DDIMSampler
from cldm.cldm_ctrlora_pretrain import ControlPretrainLDM
from datasets.coco import COCO
from datasets.multigen20m import MultiGen20M
from datasets.custom_dataset import CustomDataset


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="args")
    # Dataset configs
    parser.add_argument("--dataroot", type=str, required=True, help='path to dataset')
    parser.add_argument("--multigen20m", action='store_true', default=False, help='use multigen20m dataset')
    parser.add_argument("--coco", action='store_true', default=False, help='use coco dataset')
    parser.add_argument("--task", type=str, choices=[
        'hed', 'canny', 'seg', 'depth', 'normal', 'openpose', 'hedsketch',
        'bbox', 'outpainting', 'inpainting', 'blur', 'grayscale',
        'jpeg', 'palette', 'pixel', 'pixel2',
    ], help='task name')
    # Model configs
    parser.add_argument("--config", type=str, required=True, help='path to model config file')
    parser.add_argument("--ckpt", type=str, required=True, help='path to trained checkpoint')
    # Sampling configs
    parser.add_argument("--n_samples", type=int, default=10, help='number of samples')
    parser.add_argument("--save_dir", type=str, required=True, help='path to save samples')
    parser.add_argument("--ddim_steps", type=int, default=50, help='number of DDIM steps')
    parser.add_argument("--ddim_eta", type=float, default=0.0, help='DDIM eta')
    parser.add_argument("--strength", type=float, default=1.0, help='strength of controlnet')
    parser.add_argument("--cfg", type=float, default=7.5, help='unconditional guidance scale')
    args = parser.parse_args()

    # Construct Dataset
    if args.multigen20m:
        dataset = MultiGen20M(
            path_json=os.path.join(args.dataroot, 'json_files', f'aesthetics_plus_all_group_{args.task}_all.json'),
            path_meta=args.dataroot, task=args.task, drop_rate=0.0, random_cropping=False,
        )
    elif args.coco:
        dataset = COCO(root=args.dataroot, split='val', cond=args.task)
    else:
        dataset = CustomDataset(args.dataroot)
    if args.n_samples < len(dataset):
        dataset = Subset(dataset, range(args.n_samples))
    print('Dataset size:', len(dataset))

    # Construct Model
    model = create_model(args.config).cpu()
    if isinstance(model, ControlPretrainLDM):
        model.control_model.switch_lora(args.task)
    weights = load_state_dict(args.ckpt, location='cpu')
    model.load_state_dict(weights, strict=True)
    model = model.cuda()
    print(f"Successfully load model ckpt from {args.ckpt}")

    # Construct DDIM Sampler
    ddim_sampler = DDIMSampler(model)

    # Sample
    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs(os.path.join(args.save_dir, 'sample'), exist_ok=True)
    os.makedirs(os.path.join(args.save_dir, 'control'), exist_ok=True)
    os.makedirs(os.path.join(args.save_dir, 'img'), exist_ok=True)

    with torch.no_grad():
        for idx, item in enumerate(dataset):  # type: ignore
            img = ((item['jpg'] + 1.0) / 2.0 * 255.0).astype(np.uint8)
            img = resize_image(HWC3(img), 512)                                  # img: np.uint8, [0, 255]
            prompt = item['txt']
            control = (item['hint'] * 255.0).astype(np.uint8)
            control = resize_image(HWC3(control), 512)
            control = torch.from_numpy(control).float().cuda() / 255.0
            control = einops.rearrange(control, 'h w c -> c h w')
            control = control[None, ...]                                        # control: torch.float32, [0, 1]

            H, W, C = img.shape
            shape = (4, H // 8, W // 8)

            # sample with prompts
            cond = {"c_concat": [control], "c_crossattn": [model.get_learned_conditioning([prompt])], "task": args.task}
            un_cond = {"c_concat": [control], "c_crossattn": [model.get_learned_conditioning([''])], "task": args.task}
            model.control_scales = [args.strength] * 13
            samples, _ = ddim_sampler.sample(args.ddim_steps, 1, shape, cond,
                                             verbose=False, eta=args.ddim_eta,
                                             unconditional_guidance_scale=args.cfg,
                                             unconditional_conditioning=un_cond)
            samples = model.decode_first_stage(samples)                     # samples: torch.float32, [-1, 1]

            # Save
            samples = (einops.rearrange(samples[0], 'c h w -> h w c') * 127.5 + 127.5).cpu().numpy().clip(0, 255).astype(np.uint8)
            cv2.imwrite(os.path.join(args.save_dir, 'sample', f'{idx}.png'), samples[..., ::-1])

            cv2.imwrite(os.path.join(args.save_dir, 'img', f'{idx}.png'), img[..., ::-1])

            control = (einops.rearrange(control[0], 'c h w -> h w c') * 255.0).cpu().numpy().astype(np.uint8)
            cv2.imwrite(os.path.join(args.save_dir, 'control', f'{idx}.png'), control[..., ::-1])

            with open(os.path.join(args.save_dir, 'prompt.txt'), 'a') as f:
                print(prompt.strip(), file=f)

    print('Done')
