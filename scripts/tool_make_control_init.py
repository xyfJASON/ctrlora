"""
Adapted from https://github.com/lllyasviel/ControlNet/blob/main/tool_add_control.py
"""
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from share import *

import argparse

import torch
from cldm.model import create_model, load_state_dict


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="args")
    parser.add_argument("--config", type=str, required=True, help='path to model config file')
    parser.add_argument("--sd_ckpt", type=str, required=True, help='path to pretrained stable diffusion checkpoint')
    parser.add_argument("--output_path", type=str, required=True, help='path to output file')
    parser.add_argument("--save_all", action='store_true', default=False, help='save ControlNet and SD weights')
    args = parser.parse_args()

    def get_node_name(name, parent_name):
        if len(name) <= len(parent_name):
            return False, ''
        p = name[:len(parent_name)]
        if p != parent_name:
            return False, ''
        return True, name[len(parent_name):]

    # Construct Model
    model = create_model(args.config).cpu()

    # Copy Stable Diffusion weights to ControlNet
    sd_weights = load_state_dict(args.sd_ckpt, location='cpu')
    scratch_dict = model.state_dict()
    target_dict = {}
    for k in scratch_dict.keys():
        is_control, name = get_node_name(k, 'control_')
        if is_control:
            copy_k = 'model.diffusion_' + name
        else:
            copy_k = k
        if copy_k in sd_weights:
            target_dict[k] = sd_weights[copy_k].clone()
        else:
            target_dict[k] = scratch_dict[k].clone()
            print(f'These weights are newly added: {k}')
    model.load_state_dict(target_dict, strict=True)

    # Only save ControlNet weights
    if not args.save_all:
        state_dict = {k: v for k, v in model.state_dict().items() if k.startswith('control_model')}
    else:
        state_dict = model.state_dict()
    torch.save(state_dict, args.output_path)
    print('Done.')
