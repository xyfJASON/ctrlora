import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse

import torch
from cldm.model import load_state_dict


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--sd_ckpt', type=str, default='./ckpts/sd15/v1-5-pruned.ckpt', help='path to SD1.5 checkpoint')
    parser.add_argument('--base_ckpt', type=str, default='./ckpts/ctrlora-basecn/ctrlora_sd15_basecn700k.ckpt', help='path to Base ControlNet checkpoint')
    parser.add_argument('--lora_ckpt', type=str, required=True, help='path to LoRA checkpoint')
    parser.add_argument('--save_path', type=str, required=True, help='path to save combined weights')
    return parser


def main():
    args = get_parser().parse_args()

    # load SD1.5 checkpoint
    sd_ckpt = load_state_dict(args.sd_ckpt, location='cpu')
    sd_ckpt = {k: v for k, v in sd_ckpt.items() if not k.startswith('model_ema.')}

    # load Base ControlNet checkpoint
    base_ckpt = load_state_dict(args.base_ckpt, location='cpu')

    # load LoRA checkpoint
    lora_ckpt = load_state_dict(args.lora_ckpt, location='cpu')

    # combine weights
    ckpt = {}
    ckpt.update(sd_ckpt)
    ckpt.update(base_ckpt)
    ckpt.update(lora_ckpt)
    ckpt.update({'logvar': torch.zeros(1000)})

    # save combined weights
    os.makedirs(os.path.dirname(args.save_path), exist_ok=True)
    torch.save(ckpt, args.save_path)
    print(f'Saved combined weights to [{args.save_path}]')


if __name__ == '__main__':
    main()
