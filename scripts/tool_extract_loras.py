import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse

import torch
from cldm.cldm_ctrlora_pretrain import ControlPretrainLDM
from cldm.model import create_model, load_state_dict


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt', type=str, required=True, help='path to trained checkpoint')
    parser.add_argument('--save_path', type=str, required=True, help='path to save extracted weights')
    parser.add_argument('--from_base', action='store_true', help='extract weights from the Base ControlNet')
    parser.add_argument('--from_base_config', type=str, help='path to Base ControlNet config file')
    return parser


def extract(ckpt):
    save_ckpt = {}
    for k in ckpt.keys():
        if 'control_model' in k and 'loras_dict' not in k:
            if 'lora_layer' in k:  # lora layers
                save_ckpt[k] = ckpt[k]
            elif 'zero_convs' in k or 'middle_block_out' in k:  # zero convs
                save_ckpt[k] = ckpt[k]
            elif 'norm' in k:  # norm layers
                save_ckpt[k] = ckpt[k]
    return save_ckpt


def main():
    args = get_parser().parse_args()

    ckpt = load_state_dict(args.ckpt, location='cpu')

    if not args.from_base:
        save_ckpt = extract(ckpt)
        torch.save(save_ckpt, args.save_path)
        print(f'Extracted weights saved to {args.save_path}')

    else:
        assert not os.path.isfile(args.save_path)
        os.makedirs(args.save_path, exist_ok=True)
        model = create_model(args.from_base_config).cpu()
        assert isinstance(model, ControlPretrainLDM)
        model.control_model.switch_lora('canny')
        model.load_state_dict(ckpt, strict=True)
        for task in model.control_model.tasks:
            print(f'Extracting weights for task {task}...')
            model.control_model.switch_lora(task)
            save_ckpt = extract(model.state_dict())
            save_path = os.path.join(args.save_path, f'{task}.ckpt')
            torch.save(save_ckpt, save_path)
            print(f'Extracted weights for task {task} saved to {save_path}')

    print('Done.')


if __name__ == '__main__':
    main()
