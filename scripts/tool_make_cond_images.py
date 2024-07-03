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
    parser.add_argument("--source_dir", type=str, required=True)
    parser.add_argument("--detector", type=str, choices=[
        'jpeg', 'palette', 'pixel', 'pixel2', 'blur', 'grayscale', 'inpainting',
    ], required=True)
    parser.add_argument('--n_processes', type=int, default=4)
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

    img = Image.open(os.path.join(args.source_dir, file))
    img = np.array(img)
    img = resize_image(HWC3(img), 512)

    params = dict()
    if args.detector == 'blur':
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

    img = detector(img, **params)
    img = HWC3(img)
    img = Image.fromarray(img)
    img.save(os.path.join(target_dir, file))


if __name__ == '__main__':
    # Arguments
    args = get_parser().parse_args()
    args.n_processes = args.n_processes or mp.cpu_count()
    print(f'Using {args.n_processes} processes')

    # Build detector
    if args.detector == 'jpeg':
        from annotator.jpeg import JpegCompressor
        detector = JpegCompressor()
    elif args.detector == 'palette':
        from annotator.palette import PaletteDetector
        detector = PaletteDetector()
    elif args.detector in ['pixel', 'pixel2']:
        from annotator.pixel import Pixelater
        detector = Pixelater()
    elif args.detector == 'blur':
        from annotator.blur import Blurrer
        detector = Blurrer()
    elif args.detector == 'grayscale':
        from annotator.grayscale import GrayscaleConverter
        detector = GrayscaleConverter()
    elif args.detector == 'inpainting':
        from annotator.inpainting import Inpainter
        detector = Inpainter()
    else:
        raise NotImplementedError

    # Create target directory
    target_dir = f"{args.source_dir}-{args.detector}"
    os.makedirs(target_dir, exist_ok=True)

    # Multiprocessing
    mp.set_start_method('fork')
    pool = mp.Pool(processes=args.n_processes)
    files = os.listdir(args.source_dir)
    for _ in tqdm.tqdm(pool.imap(func, files), total=len(files)):
        pass
    pool.close()
    pool.join()
    print('Done')
