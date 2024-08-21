import argparse
import os

from PIL import Image
import pylib as py


# ==============================================================================
# =                                   param                                    =
# ==============================================================================

parser = argparse.ArgumentParser()
parser.add_argument('--ori_image_dir', required=True)
parser.add_argument('--min_short_size', type=int, required=True)
parser.add_argument('--min_image_ratio', type=float, default=0)
parser.add_argument('--save_image_format', default='jpg')
args = parser.parse_args()

save_dir = "./data/laion_aesthetics_6.5p636k"


# ==============================================================================
# =                                    run                                     =
# ==============================================================================

save_img_dir = os.path.join(save_dir, f'images_{args.min_short_size}_{args.min_image_ratio:.3f}_{args.save_image_format}')
os.makedirs(save_img_dir, exist_ok=True)
img_paths = [os.path.join(args.ori_image_dir, f) for f in os.listdir(args.ori_image_dir)]


def img_hq(img):
    w, h = img.size
    return (min(w, h) >= args.min_short_size) and (min(w / h, h / w) >= args.min_image_ratio)


def work_fn(i):
    try:
        with Image.open(img_paths[i]) as img:
            if img_hq(img):
                save_path = os.path.join(save_img_dir, f'{i:012d}.{args.save_image_format}')
                img.save(save_path, quality=95)
                return 1
    except IOError:
        ...


results = py.run_parallels(work_fn, range(len(img_paths)), max_workers=48)
print(len([i for i in results if i is not None]))
