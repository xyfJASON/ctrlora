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
parser.add_argument('--save_image_format', default='jpg')
args = parser.parse_args()

save_dir = "./data/densepose25k"


# ==============================================================================
# =                                    run                                     =
# ==============================================================================

ds = datasets.load_dataset('jschoormans/densepose_1024')['train']
img_dir = os.path.join(save_dir, f'images_{args.save_image_format}')
seg_dir = os.path.join(save_dir, f'segmentaions_{args.save_image_format}')
prompt_path = os.path.join(save_dir, f'prompt_{args.save_image_format}.json')
os.makedirs(img_dir, exist_ok=True)
os.makedirs(seg_dir, exist_ok=True)

def work_fn(i):
    try:
        img = ds[i]['file_name']
        seg = ds[i]['conditioning_image']
        prompt = ds[i]['caption']
        # save_name = f'{i:012d}.{img.format.lower()}'
        save_name = f'{i:012d}.{args.save_image_format}'
        img.save(os.path.join(img_dir, save_name), quality=95)
        seg.save(os.path.join(seg_dir, save_name), quality=95)
        prompt_dict = {'source': f'source/{save_name}', 'target': f'target/{save_name}', 'prompt': prompt}
        return json.dumps(prompt_dict)
    except:
        ...


prompt_dicts = py.run_parallels(work_fn, range(len(ds)), max_workers=16)

prompt_dicts = [pd + '\n' for pd in prompt_dicts if pd is not None]
with open(prompt_path, 'w') as prompt_f:
    prompt_f.writelines(prompt_dicts)
