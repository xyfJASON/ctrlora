import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse

import torch
from cldm.model import load_state_dict


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt', type=str, required=True, help='path to trained checkpoint')
    parser.add_argument('--save_path', type=str, required=True, help='path to save extracted weights')
    args = parser.parse_args()

    ckpt = load_state_dict(args.ckpt, location='cpu')
    save_ckpt = {}
    for k in ckpt.keys():
        if 'lora_layer' in k:  # lora layers
            save_ckpt[k] = ckpt[k]
        elif 'zero_convs' in k or 'middle_block_out' in k:  # zero convs
            save_ckpt[k] = ckpt[k]
        elif 'norm' in k:  # norm layers
            save_ckpt[k] = ckpt[k]
    torch.save(save_ckpt, args.save_path)
    print(f'Saved extracted weights to {args.save_path}')
    print('Done.')
