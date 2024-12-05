import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ['GRADIO_TEMP_DIR'] = './tmp'

from share import *
import config

import cv2
import einops
import gradio as gr
import numpy as np
import torch
import random
import re
from typing import Any

from pytorch_lightning import seed_everything
from annotator.util import resize_image, HWC3
from cldm.model import create_model, load_state_dict
from cldm.ddim_hacked import DDIMSampler


CKPT_DIR = './ckpts'
CKPT_SD15_DIR = os.path.join(CKPT_DIR, 'sd15')
CKPT_CN_DIR = os.path.join(CKPT_DIR, 'controlnet')
CONFIG_DIR = './configs'

model: Any = None
ddim_sampler: Any = None
preprocessor: Any = None
last_config = None
last_ckpts = (None, None)

det_choices = [
    'none', 'canny', 'hed', 'seg', 'depth', 'normal', 'openpose', 'hedsketch', 'grayscale', 'blur', 'pad', 'bbox',  # from unicontrol
    'lineart', 'lineart_coarse', 'lineart_anime', 'shuffle', 'mlsd',                                                # from controlnet v1.1
    'palette', 'pixel', 'illusion', 'densepose', 'lineart_anime_with_color_prompt',                                 # proposed new conditions
]

add_prompts = {
    'General-short': 'masterpiece, best quality',
    'General-long': 'masterpiece, best quality, high quality, award winning, award-winning',
    'Realistic': 'RAW photo, 8K UHD, DSLR, film grain, highres, high resolution, high detail, extremely detailed, soft lighting, award winning photography',
}

neg_prompts = {
    'General-short': 'worst quality, low quality, NSFW',
    'General-long': 'worst quality, low quality, bad quality, normal quality, lowres, low resolution, JPEG artifacts, blurry, bad composition, cropped, mutilated, out of frame, duplicate, multiple views, multiple_views, tiling, ugly, morbid, distorted, disgusting, watermark, signature, NSFW',
    'General-human': 'bad anatomy, wrong anatomy, bad proportions, gross proportions, deformed, deformed iris, deformed pupils, inaccurate eyes, cross-eye, cloned face, bad hands, mutation, mutated hands, mutation hands, mutated fingers, mutation fingers, fused fingers, too many fingers, extra fingers, extra digit, missing fingers, fewer digits, malformed limbs, inaccurate limb, extra limbs, missing limbs, floating limbs, disconnected limbs, extra arms, extra legs, missing arms, missing legs, error, bad legs, error legs, bad feet, long neck, disfigured, amputation, dehydrated, nude, thighs, cleavage',
    'Realistic': 'semi-realistic, CGI, 3D, render, sketch, drawing, comic, cartoon, anime, vector art',
    '2.5D': 'sketch, drawing, comic, cartoon, anime, vector art',
    'Painting': 'photorealistic, CGI, 3D, render',
}


def load_state_dict_sd(sd_ckpt):
    global model
    state_dict = load_state_dict(os.path.join(CKPT_SD15_DIR, sd_ckpt), location='cpu')
    model.load_state_dict(state_dict, strict=False)  # noqa
    del state_dict


def load_state_dict_cn(cn_ckpt):
    global model
    state_dict = load_state_dict(os.path.join(CKPT_CN_DIR, cn_ckpt), location='cpu')
    state_dict = {k: v for k, v in state_dict.items() if k.startswith('control_model')}
    model.load_state_dict(state_dict, strict=False)  # noqa
    del state_dict


def get_config(cn_ckpt):
    if 'ctrlora' in cn_ckpt:
        return os.path.join(CONFIG_DIR, 'ctrlora_finetune_sd15_full.yaml')
    elif 'cnlite' in cn_ckpt:
        return os.path.join(CONFIG_DIR, 'cnlite_sd15.yaml')
    elif 'cnxs' in cn_ckpt:
        return os.path.join(CONFIG_DIR, 'cnxs_sd15.yaml')
    elif 'cn' in cn_ckpt:
        return os.path.join(CONFIG_DIR, 'cldm_v15.yaml')
    else:
        raise ValueError(f'Unknown cn_ckpt: {cn_ckpt}')


def build_model(sd_ckpt, cn_ckpt):
    global model, ddim_sampler, last_ckpts, last_config
    assert sd_ckpt is not None
    assert cn_ckpt is not None

    current_config = get_config(cn_ckpt)

    if current_config != last_config:
        print(f'Loading config...')
        last_config = current_config
        model = create_model(current_config).cuda()
        ddim_sampler = DDIMSampler(model)
        print(f'Config loaded')

    if last_ckpts != (sd_ckpt, cn_ckpt):
        print(f'Loading checkpoints')
        load_state_dict_sd(sd_ckpt)
        load_state_dict_cn(cn_ckpt)
        last_ckpts = (sd_ckpt, cn_ckpt)
        print(f'Checkpoints loaded')


def detect(det, input_image, detect_resolution, image_resolution):
    global preprocessor
    if det == 'none':
        preprocessor = None
        params = dict()
    elif det == 'canny':
        from annotator.canny import CannyDetector
        if not isinstance(preprocessor, CannyDetector):
            preprocessor = CannyDetector()
        params = dict(low_threshold=100, high_threshold=200)
    elif det == 'hed':
        from annotator.hed import HEDdetector
        if not isinstance(preprocessor, HEDdetector):
            preprocessor = HEDdetector()
        params = dict()
    elif det == 'seg':
        from annotator.uniformer import UniformerDetector
        if not isinstance(preprocessor, UniformerDetector):
            preprocessor = UniformerDetector()
        params = dict()
    elif det in ['depth', 'normal']:
        from annotator.midas import MidasDetector
        if not isinstance(preprocessor, MidasDetector):
            preprocessor = MidasDetector()
        params = dict()
    elif det == 'openpose':
        from annotator.openpose import OpenposeDetector
        if not isinstance(preprocessor, OpenposeDetector):
            preprocessor = OpenposeDetector()
        params = dict()
    elif det == 'hedsketch':
        from annotator.hedsketch import HEDSketchDetector
        if not isinstance(preprocessor, HEDSketchDetector):
            preprocessor = HEDSketchDetector()
        params = dict()
    elif det == 'grayscale':
        from annotator.grayscale import GrayscaleConverter
        if not isinstance(preprocessor, GrayscaleConverter):
            preprocessor = GrayscaleConverter()
        params = dict()
    elif det == 'blur':
        from annotator.blur import Blurrer
        if not isinstance(preprocessor, Blurrer):
            preprocessor = Blurrer()
        ksize = np.random.randn() * 0.5 + 0.5
        ksize = int(ksize * (50 - 5)) + 5
        ksize = ksize * 2 + 1
        params = dict(ksize=ksize)
    elif det == 'pad':
        from annotator.pad import Padder
        if not isinstance(preprocessor, Padder):
            preprocessor = Padder()
        params = dict(top_ratio=0.50, bottom_ratio=0.50, left_ratio=0.50, right_ratio=0.50)
    elif det == 'bbox':
        from annotator.bbox import BBoxDetector
        if not isinstance(preprocessor, BBoxDetector):
            preprocessor = BBoxDetector()
        params = dict()
    elif det in ['lineart', 'lineart_coarse']:
        from annotator.lineart import LineartDetector
        if not isinstance(preprocessor, LineartDetector):
            preprocessor = LineartDetector()
        params = dict(coarse=(det == 'lineart_coarse'))
    elif det in ['lineart_anime', 'lineart_anime_with_color_prompt']:
        from annotator.lineart_anime import LineartAnimeDetector
        if not isinstance(preprocessor, LineartAnimeDetector):
            preprocessor = LineartAnimeDetector()
        params = dict()
    elif det == 'shuffle':
        from annotator.shuffle import ContentShuffleDetector
        if not isinstance(preprocessor, ContentShuffleDetector):
            preprocessor = ContentShuffleDetector()
        params = dict()
    elif det == 'mlsd':
        from annotator.mlsd import MLSDdetector
        if not isinstance(preprocessor, MLSDdetector):
            preprocessor = MLSDdetector()
        thr_v = np.random.rand() * 1.9 + 0.1  # [0.1, 2.0]
        thr_d = np.random.rand() * 19.9 + 0.1  # [0.1, 20.0]
        params = dict(thr_v=thr_v, thr_d=thr_d)
    elif det == 'palette':
        from annotator.palette import PaletteDetector
        if not isinstance(preprocessor, PaletteDetector):
            preprocessor = PaletteDetector()
        params = dict()
    elif det == 'pixel':
        from annotator.pixel import Pixelater
        if not isinstance(preprocessor, Pixelater):
            preprocessor = Pixelater()
        n_colors = np.random.randint(8, 17)  # [8,16] -> 3-4 bits
        scale = np.random.randint(4, 9)  # [4,8]
        params = dict(n_colors=n_colors, scale=scale, down_interpolation=cv2.INTER_LANCZOS4)
    elif det == 'illusion':
        from annotator.illusion import IllusionConverter
        if not isinstance(preprocessor, IllusionConverter):
            preprocessor = IllusionConverter()
        params = dict()
    elif det == 'densepose':
        from annotator.densepose import DenseposeDetector
        if not isinstance(preprocessor, DenseposeDetector):
            preprocessor = DenseposeDetector()
        params = dict()
    else:
        raise ValueError('Unknown preprocessor')

    if isinstance(input_image, dict):
        input_image = input_image['composite']

    with torch.no_grad():
        input_image = HWC3(input_image)
        if preprocessor is not None:
            resized_image = resize_image(input_image, detect_resolution)
            detected_map = preprocessor(resized_image, **params)
            if det == 'depth':
                detected_map = detected_map[0]
            elif det == 'normal':
                detected_map = detected_map[1]
        else:
            detected_map = input_image
        detected_map = HWC3(detected_map)
        H, W, C = resize_image(input_image, image_resolution).shape
        detected_map = cv2.resize(detected_map, (W, H), interpolation=cv2.INTER_LINEAR)
    return detected_map


def process_detected_image(det, detected_image):
    if isinstance(detected_image, dict):
        detected_image = detected_image['composite']
    detected_image = HWC3(detected_image)
    return detected_image


def reformat_prompt(prompt):
    prompt = re.sub(r'\[\[', ',', prompt)
    prompt = re.sub(r']]', ',', prompt)
    prompt = re.sub(r'\n', ',', prompt)
    prompt = re.sub(r'\s+', ' ', prompt)
    prompt = re.sub(r',\s+', ',', prompt)
    prompt = re.sub(r'\s+,', ',', prompt)
    prompt = re.sub(r',+', ',', prompt)
    prompt = prompt.strip(',').strip()
    prompt = re.sub(r',', ', ', prompt)
    return prompt


def process(det, detected_image, prompt, n_prompt, num_samples, ddim_steps, guess_mode, strength, scale, seed, eta, sd_ckpt, cn_ckpt):
    global model, ddim_sampler, last_ckpts, last_config

    build_model(sd_ckpt, cn_ckpt)

    prompt = reformat_prompt(prompt)
    n_prompt = reformat_prompt(n_prompt)
    print(f'Prompt is: {prompt}')
    print(f'Negative Prompt is: {n_prompt}')

    with torch.no_grad():
        detected_image = process_detected_image(det, detected_image)
        H, W, C = detected_image.shape

        control = torch.from_numpy(detected_image.copy()).float().cuda() / 255.0
        control = torch.stack([control for _ in range(num_samples)], dim=0)
        control = einops.rearrange(control, 'b h w c -> b c h w').clone()

        if seed == -1:
            seed = random.randint(0, 65535)
        seed_everything(seed)

        if config.save_memory:
            model.low_vram_shift(is_diffusing=False)

        cond = {"c_concat": [control], "c_crossattn": [model.get_learned_conditioning([prompt] * num_samples)]}
        un_cond = {"c_concat": None if guess_mode else [control], "c_crossattn": [model.get_learned_conditioning([n_prompt] * num_samples)]}

        if config.save_memory:
            model.low_vram_shift(is_diffusing=True)

        model.control_scales = [strength * (0.825 ** float(12 - i)) for i in range(13)] if guess_mode else ([strength] * 13)
        # Magic number. IDK why. Perhaps because 0.825**12<0.01 but 0.826**12>0.01

        shape = (4, H // 8, W // 8)
        samples, intermediates = ddim_sampler.sample(ddim_steps, num_samples,
                                                     shape, cond, verbose=False, eta=eta,
                                                     unconditional_guidance_scale=scale,
                                                     unconditional_conditioning=un_cond)

        if config.save_memory:
            model.low_vram_shift(is_diffusing=False)

        x_samples = model.decode_first_stage(samples)
        x_samples = (einops.rearrange(x_samples, 'b c h w -> b h w c') * 127.5 + 127.5).cpu().numpy().clip(0, 255).astype(np.uint8)

        results = [x_samples[i] for i in range(num_samples)]
    return [detected_image] + results


def listdir_r(path):
    path = os.path.expanduser(path)
    files = [os.path.join(dp, f) for dp, dn, fn in os.walk(path, followlinks=True) for f in fn]
    files = [f[len(path) + 1:] for f in files]
    return files


def update_ckpts():
    sd_ckpt = gr.Dropdown(label='Select stable diffusion checkpoint', choices=sorted(listdir_r(CKPT_SD15_DIR)))
    cn_ckpt = gr.Dropdown(label='Select controlnet checkpoint', choices=sorted(listdir_r(CKPT_CN_DIR)))
    return sd_ckpt, cn_ckpt


def update_prompt(prompt, evt: gr.SelectData):
    if evt.selected:
        prompt = prompt.strip() + '\n' + f'[[ {add_prompts[evt.value]} ]]'
    else:
        prompt = prompt.replace(f'[[ {add_prompts[evt.value]} ]]', '').replace('\n\n', '\n')
    prompt = prompt.strip()
    if prompt.endswith(']]'):
        prompt = prompt + '\n'
    return prompt


def update_n_prompt(n_prompt, evt: gr.SelectData):
    if evt.selected:
        n_prompt = n_prompt.strip() + '\n' + f'[[ {neg_prompts[evt.value]} ]]'
    else:
        n_prompt = n_prompt.replace(f'[[ {neg_prompts[evt.value]} ]]', '').replace('\n\n', '\n')
    n_prompt = n_prompt.strip()
    if n_prompt.endswith(']]'):
        n_prompt = n_prompt + '\n'
    return n_prompt


def tab1():
    with gr.Row():
        sd_ckpt = gr.Dropdown(label='Select stable diffusion checkpoint', choices=sorted(listdir_r(CKPT_SD15_DIR)), scale=3)
        cn_ckpt = gr.Dropdown(label='Select controlnet checkpoint', choices=sorted(listdir_r(CKPT_CN_DIR)), scale=3)
        refresh_button = gr.Button(value="Refresh", scale=1)
        run_button = gr.Button(value="Run", scale=1, variant='primary')

    with gr.Row():
        with gr.Column(scale=2):
            with gr.Group():
                prompt = gr.Textbox(label="Prompt", lines=3)
                a_prompt_choices = gr.CheckboxGroup(choices=list(add_prompts.keys()), type="value", label="Examples")

            with gr.Group():
                n_prompt = gr.Textbox(label="Negative Prompt", lines=2)
                n_prompt_choices = gr.CheckboxGroup(choices=list(neg_prompts.keys()), type="value", label="Examples")

            with gr.Accordion("Basic options", open=True):
                with gr.Group():
                    with gr.Row():
                        seed = gr.Slider(label="Seed", minimum=-1, maximum=2147483647, step=1, value=12345)
                        num_samples = gr.Slider(label="Images", minimum=1, maximum=12, value=1, step=1)
                        image_resolution = gr.Slider(label="Image Resolution", minimum=256, maximum=768, value=512, step=64)
                        guess_mode = gr.Checkbox(label='Guess Mode', value=False, visible=False)
                    with gr.Row():
                        ddim_steps = gr.Slider(label="DDIM Steps", minimum=1, maximum=100, value=20, step=1)
                        eta = gr.Slider(label="DDIM ETA", minimum=0.0, maximum=1.0, value=0.0, step=0.01)
                        strength = gr.Slider(label="Control Strength", minimum=0.0, maximum=2.0, value=1.0, step=0.01)
                        scale = gr.Slider(label="Guidance Scale", minimum=0.1, maximum=30.0, value=7.5, step=0.1)

            with gr.Accordion("Condition", open=True):
                with gr.Row():
                    input_image = gr.ImageEditor(sources=['upload', 'clipboard'], type="numpy", layers=False)
                    detected_image = gr.ImageEditor(sources=['upload', 'clipboard'], type="numpy", layers=False)
                det = gr.Radio(choices=det_choices, type="value", value="none", label="Preprocessor")
                detect_resolution = gr.Slider(label="Preprocessor Resolution", minimum=128, maximum=1024, value=512, step=1)
                detect_button = gr.Button(value="Detect")

        with gr.Column(scale=1):
            result_gallery = gr.Gallery(label='Output', show_label=False, elem_id="gallery", object_fit='scale-down', height=650)

    refresh_button.click(fn=update_ckpts, inputs=[], outputs=[sd_ckpt, cn_ckpt])
    a_prompt_choices.select(fn=update_prompt, inputs=[prompt], outputs=[prompt])
    n_prompt_choices.select(fn=update_n_prompt, inputs=[n_prompt], outputs=[n_prompt])
    detect_button.click(fn=detect, inputs=[det, input_image, detect_resolution, image_resolution], outputs=[detected_image])
    run_button.click(fn=process, inputs=[det, detected_image, prompt, n_prompt, num_samples, ddim_steps, guess_mode, strength, scale, seed, eta, sd_ckpt, cn_ckpt], outputs=[result_gallery])


def main():
    blocks = gr.Blocks().queue()
    with blocks:
        with gr.Row():
            gr.Markdown("## ControlNet / ControlNet-LITE / ControlNet-XS / CtrLoRA-BASE")
        tab1()
    blocks.launch(server_name='0.0.0.0')


if __name__ == '__main__':
    main()
