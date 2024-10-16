import os
import argparse
from tqdm import tqdm
from PIL import Image
import multiprocessing as mp


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=str, required=True)
    parser.add_argument('--target', type=str, required=True)
    parser.add_argument('--save_format', type=str, default=None)
    parser.add_argument('--n_processes', type=int, default=4)
    return parser


def func(f):
    try:
        img = Image.open(os.path.join(args.source, f))
        assert img.mode in ['L', 'RGB', 'RGBA']
        img_name, img_format = os.path.splitext(f)
        save_name = img_name + (img_format if args.save_format is None else f'.{args.save_format}')
        portion = 512 / min(img.size[0], img.size[1])
        sz = (int(img.size[0] * portion), int(img.size[1] * portion))
        img = img.resize(sz, Image.LANCZOS)
        img = img.crop((img.size[0] // 2 - 256, img.size[1] // 2 - 256,
                        img.size[0] // 2 + 256, img.size[1] // 2 + 256))
        img.save(os.path.join(args.target, save_name), quality=95, icc_profile=None)
    except:
        return


if __name__ == '__main__':
    # Arguments
    args = get_parser().parse_args()
    args.n_processes = args.n_processes or mp.cpu_count()
    print(f'Using {args.n_processes} processes')

    os.makedirs(args.target, exist_ok=True)

    # Multiprocessing
    mp.set_start_method('fork')
    pool = mp.Pool(processes=args.n_processes)
    files = os.listdir(args.source)
    for _ in tqdm(pool.imap(func, files), total=len(files)):
        pass
    pool.close()
    pool.join()
    print('Done')
