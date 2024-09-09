import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2
import tqdm
import random
import hashlib
import argparse
import numpy as np
from PIL import Image
import multiprocessing as mp

from annotator.util import HWC3, resize_image


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--detector", type=str, choices=[
        'canny', 'hed', 'seg', 'depth', 'normal', 'openpose', 'hedsketch',      # from unicontrol
        'bbox', 'outpainting', 'blur', 'grayscale', 'inpainting',               # from unicontrol
        'lineart', 'lineart_anime', 'shuffle', 'mlsd',                          # from controlnet v1.1
        'jpeg', 'palette', 'pixel', 'pixel2', 'illusion', 'densepose',          # proposed new conditions
        'grayscale_with_color_prompt', 'grayscale_with_color_brush',
        'lineart_anime_with_color_prompt', 'inpainting_brush',
    ], required=True)
    parser.add_argument('--n_processes', type=int, default=1)
    return parser


def discrete_normal(a, b):
    x = np.random.randn() * 0.5 + 0.5  # [0, 1]
    x = int(x * (b - a)) + a
    return x


def set_seed_by_hash(obj_id):
    obj_hash = hashlib.sha256(str(obj_id).encode())
    seed = int(obj_hash.hexdigest(), 16)
    np.random.seed(seed % 2**32)
    random.seed(seed)


def func(file):
    set_seed_by_hash(file)

    img = Image.open(os.path.join(args.input_dir, file))
    img = np.array(img)
    img = resize_image(HWC3(img), 512)

    params = dict()
    if args.detector == 'canny':
        low_threshold = 100
        high_threshold = 200
        params = dict(low_threshold=low_threshold, high_threshold=high_threshold)
    elif args.detector == 'outpainting':
        rand_h = np.random.randint(20, 80)
        rand_w = np.random.randint(20, 80)
        params = dict(rand_h=rand_h, rand_w=rand_w)
    elif args.detector == 'blur':
        ksize = discrete_normal(5, 50)
        ksize = ksize * 2 + 1
        params = dict(ksize=ksize)
    elif args.detector == 'inpainting':
        rand_h = discrete_normal(20, 80)
        rand_h_1 = discrete_normal(20, 80)
        rand_w = discrete_normal(20, 80)
        rand_w_1 = discrete_normal(20, 80)
        if rand_h > rand_h_1:
            rand_h, rand_h_1 = rand_h_1, rand_h
        if rand_w > rand_w_1:
            rand_w, rand_w_1 = rand_w_1, rand_w
        params = dict(rand_h=rand_h, rand_h_1=rand_h_1, rand_w=rand_w, rand_w_1=rand_w_1)
    elif args.detector == 'jpeg':
        jpeg_quality = discrete_normal(10, 30)
        params = dict(jpeg_quality=jpeg_quality)
    elif args.detector == 'pixel':
        n_colors = np.random.randint(8, 17)  # [8,16] -> 3-4 bits
        scale = np.random.randint(4, 9)  # [4,8]
        params = dict(n_colors=n_colors, scale=scale)
    elif args.detector == 'pixel2':
        n_colors = np.random.randint(8, 17)  # [8,16] -> 3-4 bits
        scale = np.random.randint(4, 9)  # [4,8]
        params = dict(n_colors=n_colors, scale=scale, down_interpolation=cv2.INTER_LANCZOS4)
    elif args.detector == 'lineart':
        coarse = np.random.rand() > 0.5
        params = dict(coarse=coarse)
    elif args.detector == 'mlsd':
        thr_v = np.random.rand() * 1.9 + 0.1  # [0.1, 2.0]
        thr_d = np.random.rand() * 19.9 + 0.1  # [0.1, 20.0]
        params = dict(thr_v=thr_v, thr_d=thr_d)

    img = detector(img, **params)
    if args.detector == 'openpose' and img.sum() == 0:
        return
    if img is None:
        return
    if args.detector == 'depth':
        img = img[0]
    elif args.detector == 'normal':
        img = img[1]
    img = HWC3(img)
    img = Image.fromarray(img)
    img.save(os.path.join(args.output_dir, file), quality=95, icc_profile=None)


if __name__ == '__main__':
    # Arguments
    args = get_parser().parse_args()
    args.n_processes = args.n_processes or mp.cpu_count()
    print(f'Using {args.n_processes} processes')

    # Build detector
    if args.detector == 'canny':
        from annotator.canny import CannyDetector
        detector = CannyDetector()
    elif args.detector == 'hed':
        from annotator.hed import HEDdetector
        detector = HEDdetector()
    elif args.detector == 'seg':
        from annotator.uniformer import UniformerDetector
        detector = UniformerDetector()
    elif args.detector in ['depth', 'normal']:
        from annotator.midas import MidasDetector
        detector = MidasDetector()
    elif args.detector == 'openpose':
        from annotator.openpose import OpenposeDetector
        detector = OpenposeDetector()
    elif args.detector == 'hedsketch':
        from annotator.hedsketch import HEDSketchDetector
        detector = HEDSketchDetector()
    elif args.detector == 'outpainting':
        from annotator.outpainting import Outpainter
        detector = Outpainter()
    elif args.detector == 'blur':
        from annotator.blur import Blurrer
        detector = Blurrer()
    elif args.detector == 'grayscale':
        from annotator.grayscale import GrayscaleConverter
        detector = GrayscaleConverter()
    elif args.detector == 'inpainting':
        from annotator.inpainting import Inpainter
        detector = Inpainter()
    elif args.detector == 'lineart':
        from annotator.lineart import LineartDetector
        detector = LineartDetector()
    elif args.detector == 'lineart_anime':
        from annotator.lineart_anime import LineartAnimeDetector
        detector = LineartAnimeDetector()
    elif args.detector == 'shuffle':
        from annotator.shuffle import ContentShuffleDetector
        detector = ContentShuffleDetector()
    elif args.detector == 'mlsd':
        from annotator.mlsd import MLSDdetector
        detector = MLSDdetector()
    elif args.detector == 'jpeg':
        from annotator.jpeg import JpegCompressor
        detector = JpegCompressor()
    elif args.detector == 'palette':
        from annotator.palette import PaletteDetector
        detector = PaletteDetector()
    elif args.detector in ['pixel', 'pixel2']:
        from annotator.pixel import Pixelater
        detector = Pixelater()
    elif args.detector == 'illusion':
        from annotator.illusion import IllusionConverter
        detector = IllusionConverter()
    elif args.detector == 'densepose':
        from annotator.densepose import DenseposeDetector
        detector = DenseposeDetector()
    elif args.detector == 'grayscale_with_color_prompt':
        from annotator.grayscale_with_color_prompt import GrayscaleWithColorPromptConverter
        detector = GrayscaleWithColorPromptConverter()
    elif args.detector == 'grayscale_with_color_brush':
        from annotator.grayscale_with_color_brush import GrayscaleWithColorBrushConverter
        detector = GrayscaleWithColorBrushConverter()
    elif args.detector == 'lineart_anime_with_color_prompt':
        from annotator.lineart_anime_with_color_prompt import LineartAnimeWithColorPromptConverter
        detector = LineartAnimeWithColorPromptConverter()
    elif args.detector == 'inpainting_brush':
        from annotator.inpainting_brush import BrushInpainter
        detector = BrushInpainter()
    else:
        raise NotImplementedError

    os.makedirs(args.output_dir, exist_ok=True)

    if args.n_processes == 1:
        # Single process
        files = os.listdir(args.input_dir)
        for f in tqdm.tqdm(files):
            func(f)

    else:
        if args.detector in [
            'hed', 'seg', 'depth', 'normal', 'openpose', 'hedsketch', 'bbox',
            'lineart', 'lineart_anime', 'lineart_anime_with_color_prompt', 'densepose',
        ]:
            raise ValueError(f'{args.detector} detector is not compatible with multiprocessing, please pass --n_processes=1')
        # Multiprocessing
        mp.set_start_method('fork')
        pool = mp.Pool(processes=args.n_processes)
        files = os.listdir(args.input_dir)
        for _ in tqdm.tqdm(pool.imap(func, files), total=len(files)):
            pass
        pool.close()
        pool.join()

    print('Done')
