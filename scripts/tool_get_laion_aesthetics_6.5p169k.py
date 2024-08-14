import argparse
import io
import json
import os

import datasets
from PIL import Image
import pylib as py


# ==============================================================================
# =                                   param                                    =
# ==============================================================================

parser = argparse.ArgumentParser()
parser.add_argument('--min_short_size', type=int, required=True)
parser.add_argument('--min_image_ratio', type=float, default=0)
parser.add_argument('--save_image_format', default='jpg')
args = parser.parse_args()

save_dir = "./data/laion_aesthetics_6.5p169k"


# ==============================================================================
# =                                    run                                     =
# ==============================================================================

ds = datasets.load_dataset('bhargavsdesai/laion_improved_aesthetics_6.5plus_with_images')['train']
img_dir = os.path.join(save_dir, f'images_{args.min_short_size}_{args.min_image_ratio:.2f}_{args.save_image_format}')
prompt_path = os.path.join(save_dir, f'prompt_{args.min_short_size}_{args.min_image_ratio:.2f}_{args.save_image_format}.json')
os.makedirs(img_dir, exist_ok=True)


def img_hq(img):
    w, h = img.size
    return (min(w, h) >= args.min_short_size) and (min(w / h, h / w) >= args.min_image_ratio)


def work_fn(i):
    try:
        img_bytes = ds[i]['image']['bytes']
        prompt = ds[i]['text']
        with Image.open(io.BytesIO(img_bytes)) as img:
            if img_hq(img):
                # save_name = f'{i:012d}.{img.format.lower()}'
                save_name = f'{i:012d}.{args.save_image_format}'
                save_path = os.path.join(img_dir, save_name)
                # with open(save_path, 'wb') as img_f:
                #     f.write(img_bytes)
                img.save(save_path, quality=95)
                prompt_dict = {'source': f'source/{save_name}', 'target': f'target/{save_name}', 'prompt': prompt}
                return json.dumps(prompt_dict)
    except IOError:
        ...


prompt_dicts = py.run_parallels(work_fn, range(len(ds)), max_workers=16)
prompt_dicts = [pd + '\n' for pd in prompt_dicts if pd is not None]
with open(prompt_path, 'w') as prompt_f:
    prompt_f.writelines(prompt_dicts)
