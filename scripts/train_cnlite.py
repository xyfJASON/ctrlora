import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from share import *

import gc
import argparse
import datetime

import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader, Subset

from datasets.multigen20m import MultiGen20M
from datasets.custom_dataset import CustomDataset
from cldm.logger import ImageLogger, CheckpointEveryNSteps
from cldm.model import create_model, load_state_dict
from cldm.hack import enable_sliced_attention


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="args")
    # Dataset configs
    parser.add_argument("--dataroot", type=str, required=True, help='path to dataset')
    parser.add_argument("--drop_rate", type=float, default=0.3, help='drop rate for classifier-free guidance')
    parser.add_argument("--multigen20m", action='store_true', default=False, help='use multigen20m dataset')
    parser.add_argument("--task", type=str, choices=[
        'hed', 'canny', 'seg', 'depth', 'normal', 'openpose', 'hedsketch',
        'bbox', 'outpainting', 'inpainting', 'blur', 'grayscale',
    ], help='task name')
    parser.add_argument("--subset", type=int, default=0, help='train on a subset of the dataset')
    # Model configs
    parser.add_argument("--config", type=str, required=True, help='path to model config file')
    parser.add_argument("--sd_ckpt", type=str, required=True, help='path to pretrained stable diffusion checkpoint')
    # Training configs
    parser.add_argument("-n", "--name", type=str, help='experiment name')
    parser.add_argument("--lr", type=float, default=1e-5, help='learning rate')
    parser.add_argument("--bs", type=int, default=1, help='batchsize per device')
    parser.add_argument("--max_steps", type=int, default=100000, help='max training steps')
    parser.add_argument("--gradacc", type=int, default=1, help='gradient accumulation')
    parser.add_argument("--precision", type=int, default=32, help='precision')
    parser.add_argument("--save_memory", action='store_true', default=False, help='save memory using sliced attention')
    parser.add_argument("--img_logger_freq", type=int, default=1000, help='img logger freq')
    parser.add_argument("--ckpt_logger_freq", type=int, default=1000, help='ckpt logger freq')
    args = parser.parse_args()

    # Save memory
    if args.save_memory:
        enable_sliced_attention()

    # Construct Dataset
    if args.multigen20m:
        dataset = MultiGen20M(
            path_json=os.path.join(args.dataroot, 'json_files', f'aesthetics_plus_all_group_{args.task}_all.json'),
            path_meta=args.dataroot, task=args.task, drop_rate=args.drop_rate,
        )
    else:
        dataset = CustomDataset(args.dataroot, drop_rate=args.drop_rate)
    if args.subset > 0:
        dataset = Subset(dataset, range(args.subset))
    dataloader = DataLoader(dataset, num_workers=16, batch_size=args.bs, shuffle=True)
    print('Dataset size:', len(dataset))
    print('Number of devices:', torch.cuda.device_count())
    print('Batch size per device:', args.bs)
    print('Gradient accumulation:', args.gradacc)
    print('Total batch size:', args.bs * torch.cuda.device_count() * args.gradacc)

    # Construct Model
    model = create_model(args.config).cpu()
    model.learning_rate = args.lr

    scratch_dict = model.state_dict()

    # Copy Stable Diffusion weights to scratch_dict
    copied_keys, missing_keys = [], []
    sd_weights = load_state_dict(args.sd_ckpt, location='cpu')
    for k in sd_weights:
        if k not in scratch_dict:
            missing_keys.append(k)
        else:
            scratch_dict[k] = sd_weights[k].clone()
            copied_keys.append(k)
    os.makedirs('./tmp', exist_ok=True)
    with open('./tmp/cnlite_missing_keys_sd.txt', 'w') as f:
        f.write('\n'.join(missing_keys))
    with open('./tmp/cnlite_copied_keys_sd.txt', 'w') as f:
        f.write('\n'.join(copied_keys))

    # Load scratch_dict to model
    model.load_state_dict(scratch_dict, strict=True)
    print(f"Successfully initialize SD from {args.sd_ckpt}")
    del sd_weights, scratch_dict
    gc.collect()

    # Build Trainer
    logger_img = ImageLogger(batch_frequency=args.img_logger_freq)
    logger_checkpoint = CheckpointEveryNSteps(save_step_frequency=args.ckpt_logger_freq)
    if args.name is None:
        args.name = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    trainer = pl.Trainer(
        strategy='ddp', accelerator='gpu', devices=-1, accumulate_grad_batches=args.gradacc,
        max_steps=args.max_steps, precision=args.precision, callbacks=[logger_img, logger_checkpoint],
        default_root_dir=os.path.join('runs', args.name),
    )

    # Train!
    trainer.fit(model, dataloader)
