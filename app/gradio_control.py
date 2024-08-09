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

from pytorch_lightning import seed_everything
from annotator.util import resize_image, HWC3
from cldm.model import create_model, load_state_dict
from cldm.ddim_hacked import DDIMSampler


CKPT_DIR = './ckpts'
CKPT_SD15_DIR = os.path.join(CKPT_DIR, 'sd15')
CKPT_BASECN_DIR = os.path.join(CKPT_DIR, 'ctrlora-basecn')
CKPT_LORAS_DIR = os.path.join(CKPT_DIR, 'ctrlora-loras')
CONFIG_DIR = './configs'

model = None
ddim_sampler = None
preprocessor = None
last_config = None
last_ckpts = (None, None, None)


def check_key(k):
    return 'lora_layer' in k or 'zero_convs' in k or 'middle_block_out' in k or 'norm' in k


def load_state_dict_sd(sd_ckpt):
    global model
    state_dict = load_state_dict(os.path.join(CKPT_SD15_DIR, sd_ckpt), location='cpu')
    model.load_state_dict(state_dict, strict=False)  # noqa
    del state_dict


def load_state_dict_cn(cn_ckpt):
    global model
    state_dict = load_state_dict(os.path.join(CKPT_BASECN_DIR, cn_ckpt), location='cpu')
    state_dict = {k: v for k, v in state_dict.items() if k.startswith('control_model') and not check_key(k)}
    model.load_state_dict(state_dict, strict=False)  # noqa
    del state_dict


def load_state_dict_lora(lora_ckpt):
    global model
    state_dict = load_state_dict(os.path.join(CKPT_LORAS_DIR, lora_ckpt), location='cpu')
    state_dict = {k: v for k, v in state_dict.items() if check_key(k)}
    model.load_state_dict(state_dict, strict=False)  # noqa
    del state_dict


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
    elif det in ['grayscale', 'grayscale_with_color_prompt', 'grayscale_with_color_brush']:
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
    elif det in ['lineart', 'lineart(coarse)']:
        from annotator.lineart import LineartDetector
        if not isinstance(preprocessor, LineartDetector):
            preprocessor = LineartDetector()
        params = dict(coarse=(det == 'lineart(coarse)'))
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
        params = dict(n_colors=n_colors, scale=scale)
    elif det == 'pixel2':
        from annotator.pixel import Pixelater
        if not isinstance(preprocessor, Pixelater):
            preprocessor = Pixelater()
        n_colors = np.random.randint(8, 17)  # [8,16] -> 3-4 bits
        scale = np.random.randint(4, 9)  # [4,8]
        params = dict(n_colors=n_colors, scale=scale, down_interpolation=cv2.INTER_LANCZOS4)
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


def process(det, detected_image, prompt, a_prompt, n_prompt, num_samples, ddim_steps, guess_mode, strength, scale, seed, eta, sd_ckpt, cn_ckpt, lora_ckpt):
    global model, ddim_sampler, last_ckpts, last_config

    if isinstance(detected_image, dict):
        if det in ['grayscale_with_color_prompt', 'grayscale_with_color_brush']:
            yuv_bg = cv2.cvtColor(HWC3(detected_image['background']), cv2.COLOR_RGB2YUV)
            yuv_cp = cv2.cvtColor(HWC3(detected_image['composite']), cv2.COLOR_RGB2YUV)
            yuv_cp[:, :, 0] = yuv_bg[:, :, 0]
            detected_image = cv2.cvtColor(yuv_cp, cv2.COLOR_YUV2RGB)
        else:
            detected_image = detected_image['composite']

    assert sd_ckpt is not None
    assert cn_ckpt is not None
    assert lora_ckpt is not None
    if 'rank128' in lora_ckpt:
        current_config = os.path.join(CONFIG_DIR, 'ctrlora_finetune_sd15_rank128.yaml')
    elif 'rank256' in lora_ckpt:
        current_config = os.path.join(CONFIG_DIR, 'ctrlora_finetune_sd15_rank256.yaml')
    elif 'rank512' in lora_ckpt:
        current_config = os.path.join(CONFIG_DIR, 'ctrlora_finetune_sd15_rank512.yaml')
    else:
        raise ValueError('Unknown config')

    if current_config != last_config:
        print(f'Loading config...')
        last_config = current_config
        model = create_model(current_config).cuda()
        ddim_sampler = DDIMSampler(model)
        print(f'Config loaded')

    if last_ckpts != (sd_ckpt, cn_ckpt, lora_ckpt):
        print(f'Loading checkpoints')
        load_state_dict_sd(sd_ckpt)
        load_state_dict_cn(cn_ckpt)
        load_state_dict_lora(lora_ckpt)
        last_ckpts = (sd_ckpt, cn_ckpt, lora_ckpt)
        print(f'Checkpoints loaded')

    with torch.no_grad():
        detected_image = HWC3(detected_image)
        H, W, C = detected_image.shape

        control = torch.from_numpy(detected_image.copy()).float().cuda() / 255.0
        control = torch.stack([control for _ in range(num_samples)], dim=0)
        control = einops.rearrange(control, 'b h w c -> b c h w').clone()

        if seed == -1:
            seed = random.randint(0, 65535)
        seed_everything(seed)

        if config.save_memory:
            model.low_vram_shift(is_diffusing=False)

        cond = {"c_concat": [control], "c_crossattn": [model.get_learned_conditioning([prompt + ', ' + a_prompt] * num_samples)]}
        un_cond = {"c_concat": None if guess_mode else [control], "c_crossattn": [model.get_learned_conditioning([n_prompt] * num_samples)]}
        shape = (4, H // 8, W // 8)

        if config.save_memory:
            model.low_vram_shift(is_diffusing=True)

        model.control_scales = [strength * (0.825 ** float(12 - i)) for i in range(13)] if guess_mode else ([strength] * 13)
        # Magic number. IDK why. Perhaps because 0.825**12<0.01 but 0.826**12>0.01

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
    files = [os.path.join(dp, f) for dp, dn, fn in os.walk(path) for f in fn]
    files = [f[len(path) + 1:] for f in files]
    return files


def update_ckpts():
    sd_ckpt = gr.Dropdown(label='Select stable diffusion checkpoint', choices=sorted(listdir_r(CKPT_SD15_DIR)))
    cn_ckpt = gr.Dropdown(label='Select base controlnet checkpoint', choices=sorted(listdir_r(CKPT_BASECN_DIR)))
    lora_ckpt = gr.Dropdown(label='Select lora checkpoint', choices=sorted(listdir_r(CKPT_LORAS_DIR)))
    return sd_ckpt, cn_ckpt, lora_ckpt


def main():
    block = gr.Blocks().queue()
    with block:
        with gr.Row():
            gr.Markdown("## CtrLoRA")
        with gr.Row():
            sd_ckpt = gr.Dropdown(label='Select stable diffusion checkpoint', choices=sorted(listdir_r(CKPT_SD15_DIR)), value='v1-5-pruned.ckpt', scale=2)
            cn_ckpt = gr.Dropdown(label='Select base controlnet checkpoint', choices=sorted(listdir_r(CKPT_BASECN_DIR)), scale=2)
            lora_ckpt = gr.Dropdown(label='Select lora checkpoint', choices=sorted(listdir_r(CKPT_LORAS_DIR)), scale=2)
            refresh_button = gr.Button(value="Refresh", scale=1)
        with gr.Row():
            with gr.Column(scale=2):
                with gr.Row():
                    input_image = gr.ImageEditor(sources=['upload', 'clipboard'], type="numpy")
                    detected_image = gr.ImageEditor(sources=['upload', 'clipboard'], type="numpy")
                prompt = gr.Textbox(label="Prompt")
                num_samples = gr.Slider(label="Images", minimum=1, maximum=12, value=1, step=1)
                seed = gr.Slider(label="Seed", minimum=-1, maximum=2147483647, step=1, value=12345)
                det = gr.Radio(choices=[
                    'none', 'canny', 'hed', 'seg', 'depth', 'normal', 'openpose', 'hedsketch', 'grayscale', 'blur',
                    'lineart', 'lineart(coarse)', 'lineart_anime', 'shuffle', 'mlsd',
                    'palette', 'pixel', 'pixel2', 'grayscale_with_color_prompt', 'grayscale_with_color_brush',
                    'lineart_anime_with_color_prompt',
                ], type="value", value="none", label="Preprocessor")
                with gr.Row():
                    detect_button = gr.Button(value="Detect")
                    run_button = gr.Button(value="Run")
                with gr.Accordion("Advanced options", open=False):
                    image_resolution = gr.Slider(label="Image Resolution", minimum=256, maximum=768, value=512, step=64)
                    strength = gr.Slider(label="Control Strength", minimum=0.0, maximum=2.0, value=1.0, step=0.01)
                    guess_mode = gr.Checkbox(label='Guess Mode', value=False)
                    detect_resolution = gr.Slider(label="Preprocessor Resolution", minimum=128, maximum=1024, value=512, step=1)
                    ddim_steps = gr.Slider(label="Steps", minimum=1, maximum=100, value=20, step=1)
                    scale = gr.Slider(label="Guidance Scale", minimum=0.1, maximum=30.0, value=7.5, step=0.1)
                    eta = gr.Slider(label="DDIM ETA", minimum=0.0, maximum=1.0, value=0.0, step=0.01)
                    a_prompt = gr.Textbox(label="Added Prompt", value='best quality')
                    n_prompt = gr.Textbox(label="Negative Prompt", value='lowres, bad anatomy, bad hands, cropped, worst quality')
            with gr.Column(scale=1):
                result_gallery = gr.Gallery(label='Output', show_label=False, elem_id="gallery")
        refresh_button.click(fn=update_ckpts, inputs=[], outputs=[sd_ckpt, cn_ckpt, lora_ckpt])
        detect_button.click(fn=detect, inputs=[det, input_image, detect_resolution, image_resolution], outputs=[detected_image])
        ips = [det, detected_image, prompt, a_prompt, n_prompt, num_samples, ddim_steps, guess_mode, strength, scale, seed, eta, sd_ckpt, cn_ckpt, lora_ckpt]
        run_button.click(fn=process, inputs=ips, outputs=[result_gallery])

    block.launch(server_name='0.0.0.0')


if __name__ == '__main__':
    main()
