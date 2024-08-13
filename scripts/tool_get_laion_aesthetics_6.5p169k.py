import argparse
import io
import json
import os

import datasets
from PIL import Image
import tqdm


# ==============================================================================
# =                                   param                                    =
# ==============================================================================

parser = argparse.ArgumentParser()
parser.add_argument('--min_short_size', type=int, required=True)
parser.add_argument('--min_image_ratio', type=float, default=0)
args = parser.parse_args()

save_dir = "./data/laion_aesthetics_6.5p169k"
min_short_size = args.min_short_size
min_image_ratio = args.min_image_ratio


# ==============================================================================
# =                                    run                                     =
# ==============================================================================

def img_hq(img):
    w, h = img.size
    return (min(w, h) >= min_short_size) and (min(w / h, h / w) >= min_image_ratio)


ds = datasets.load_dataset('bhargavsdesai/laion_improved_aesthetics_6.5plus_with_images')['train']

img_dir = os.path.join(save_dir, f'images_{min_short_size}_{min_image_ratio:.2f}')
prompt_path = os.path.join(save_dir, f'prompt_{min_short_size}_{min_image_ratio:.2f}.json')
os.makedirs(img_dir, exist_ok=True)
with open(prompt_path, 'w') as prompt_f:
    for i in tqdm.tqdm(range(len(ds))):
        try:
            img_bytes = ds[i]['image']['bytes']
            prompt = ds[i]['text']
            with Image.open(io.BytesIO(img_bytes)) as img:
                if img_hq(img):
                    # save_name = f'{i:012d}.{img.format.lower()}'
                    save_name = f'{i:012d}.jpg'
                    save_path = os.path.join(img_dir, save_name)
                    # with open(save_path, 'wb') as img_f:
                    #     f.write(img_bytes)
                    img.save(save_path, format='JPEG', quality=95)
                    prompt_dict = {'source': f'source/{save_name}', 'target': f'target/{save_name}', 'prompt': prompt}
                    prompt_f.write(json.dumps(prompt_dict) + '\n')
        except IOError:
            ...
