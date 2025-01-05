from share import *

import os
import einops
import numpy as np
from PIL import Image

import torch

from cldm.model import create_model, load_state_dict
from cldm.ddim_hacked import DDIMSampler
from annotator.util import HWC3


class CtrLoRA:
    def __init__(self, num_loras=1):
        self.model = None
        self.num_loras = num_loras

        if num_loras == 1:
            self.config_file = 'configs/inference/ctrlora_sd15_rank128_1lora.yaml'
        elif num_loras == 2:
            self.config_file = 'configs/inference/ctrlora_sd15_rank128_2loras.yaml'
        else:
            raise ValueError('Invalid number of LoRAs. Only 1 or 2 are supported.')

    @staticmethod
    def check_key(k):
        return 'lora_layer' in k or 'zero_convs' in k or 'middle_block_out' in k or 'norm' in k

    def create_model(
            self,
            sd_file='ckpts/sd15/v1-5-pruned.ckpt',
            basecn_file='ckpts/ctrlora-basecn/ctrlora_sd15_basecn700k.ckpt',
            lora_files=('ckpts/ctrlora-loras/novel-conditions/ctrlora_sd15_basecn700k_lineart_rank128_1kimgs_1ksteps.ckpt', ),
    ):
        # check if files exist
        assert os.path.exists(sd_file), f'File not found: {sd_file}'
        assert os.path.exists(basecn_file), f'File not found: {basecn_file}'
        if not isinstance(lora_files, (tuple, list)):
            lora_files = (lora_files, )
        for lora_file in lora_files:
            assert os.path.exists(lora_file), f'File not found: {lora_file}'
        # create model from config file
        self.model = create_model(self.config_file).cuda()
        # load sd state dict
        sd_state_dict = load_state_dict(sd_file, location='cpu')
        self.model.load_state_dict(sd_state_dict, strict=False)
        del sd_state_dict
        # load basecn state dict
        cn_state_dict = load_state_dict(basecn_file, location='cpu')
        cn_state_dict = {k: v for k, v in cn_state_dict.items() if k.startswith('control_model') and not self.check_key(k)}
        self.model.load_state_dict(cn_state_dict, strict=False)
        del cn_state_dict
        # load lora state dicts
        for i, lora_file in enumerate(lora_files):
            lora_state_dict = load_state_dict(lora_file, location='cpu')
            lora_state_dict = {k: v for k, v in lora_state_dict.items() if self.check_key(k)}
            self.model.control_model.switch_lora(i)
            self.model.load_state_dict(lora_state_dict, strict=False)
            self.model.control_model.copy_weights_to_switchable()
            del lora_state_dict

    def sample(self, cond_image_paths, prompt, n_prompt='', num_samples=1, ddim_steps=20, scale=7.5, lora_weights=(1.0, 1.0)):
        assert self.model is not None, 'Model is not loaded. Please call create_model() first.'
        if not isinstance(cond_image_paths, (tuple, list)):
            cond_image_paths = (cond_image_paths, )
        assert len(cond_image_paths) == self.num_loras, f'Expected {self.num_loras} images, got {len(cond_image_paths)}'
        # read condition images
        detected_images = []
        for cond_image_path in cond_image_paths:
            detected_image = np.array(Image.open(cond_image_path))
            detected_image = HWC3(detected_image)
            detected_images.append(detected_image)
        # sample
        if self.num_loras == 1:
            return self.sample_1lora(detected_images[0], prompt, n_prompt, num_samples, ddim_steps, scale)
        elif self.num_loras == 2:
            return self.sample_2loras(detected_images, prompt, n_prompt, num_samples, ddim_steps, scale, lora_weights)
        else:
            raise ValueError('Invalid number of LoRAs. Only 1 or 2 are supported.')

    def sample_1lora(self, detected_image, prompt, n_prompt='', num_samples=1, ddim_steps=20, scale=7.5):
        H, W, C = detected_image.shape
        ddim_sampler = DDIMSampler(self.model)
        with torch.no_grad():
            control = torch.from_numpy(detected_image.copy()).float().cuda() / 255.0
            control = torch.stack([control for _ in range(num_samples)], dim=0)
            control = einops.rearrange(control, 'b h w c -> b c h w').clone()

            cond = {"c_concat": [control], "c_crossattn": [self.model.get_learned_conditioning([prompt] * num_samples)]}
            un_cond = {"c_concat": [control], "c_crossattn": [self.model.get_learned_conditioning([n_prompt] * num_samples)]}

            self.model.control_scales = [1] * 13

            shape = (4, H // 8, W // 8)
            samples, intermediates = ddim_sampler.sample(
                ddim_steps, num_samples,
                shape, cond, verbose=False, eta=0,
                unconditional_guidance_scale=scale,
                unconditional_conditioning=un_cond,
            )

            x_samples = self.model.decode_first_stage(samples)
            x_samples = (einops.rearrange(x_samples, 'b c h w -> b h w c') * 127.5 + 127.5).cpu().numpy().clip(0, 255).astype(np.uint8)
            results = [Image.fromarray(x_samples[i]) for i in range(num_samples)]
        return results

    def sample_2loras(self, detected_images, prompt, n_prompt='', num_samples=1, ddim_steps=20, scale=7.5, lora_weights=(1.0, 1.0)):
        detected_image, detected_image2 = detected_images
        # center crop to smaller image
        H, W, C = detected_image.shape
        H2, W2, C2 = detected_image2.shape
        if H2 > H:
            detected_image2 = detected_image2[(H2-H)//2:(H2+H)//2]
        else:
            detected_image = detected_image[(H-H2)//2:(H+H2)//2]
        if W2 > W:
            detected_image2 = detected_image2[:, (W2-W)//2:(W2+W)//2]
        else:
            detected_image = detected_image[:, (W-W2)//2:(W+W2)//2]
        H, W, C = detected_image.shape
        H2, W2, C2 = detected_image2.shape
        assert H == H2 and W == W2

        ddim_sampler = DDIMSampler(self.model)
        with torch.no_grad():
            control = torch.from_numpy(detected_image.copy()).float().cuda() / 255.0
            control = torch.stack([control for _ in range(num_samples)], dim=0)
            control = einops.rearrange(control, 'b h w c -> b c h w').clone()

            control2 = torch.from_numpy(detected_image2.copy()).float().cuda() / 255.0
            control2 = torch.stack([control2 for _ in range(num_samples)], dim=0)
            control2 = einops.rearrange(control2, 'b h w c -> b c h w').clone()

            cond = {"c_concat": [control], "c_crossattn": [self.model.get_learned_conditioning([prompt] * num_samples)]}
            un_cond = {"c_concat": [control], "c_crossattn": [self.model.get_learned_conditioning([n_prompt] * num_samples)]}
            cond2 = {"c_concat": [control2], "c_crossattn": [self.model.get_learned_conditioning([prompt] * num_samples)]}
            un_cond2 = {"c_concat": [control2], "c_crossattn": [self.model.get_learned_conditioning([n_prompt] * num_samples)]}

            self.model.control_scales = [1] * 13
            self.model.lora_weights = [lora_weights[0], lora_weights[1]]

            shape = (4, H // 8, W // 8)
            samples, intermediates = ddim_sampler.sample(
                ddim_steps, num_samples,
                shape, [cond, cond2], verbose=False, eta=0,
                unconditional_guidance_scale=scale,
                unconditional_conditioning=[un_cond, un_cond2],
            )

            x_samples = self.model.decode_first_stage(samples)
            x_samples = (einops.rearrange(x_samples, 'b c h w -> b h w c') * 127.5 + 127.5).cpu().numpy().clip(0, 255).astype(np.uint8)
            results = [Image.fromarray(x_samples[i]) for i in range(num_samples)]
        return results
