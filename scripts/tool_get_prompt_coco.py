import json
import argparse


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--ann_file', type=str, required=True)
    parser.add_argument('--save_path', type=str, required=True)
    args = parser.parse_args()

    with open(args.ann_file, 'r') as f:
        data = json.load(f)

    captions = dict()
    for i in range(len(data['annotations'])):
        filename = str(data['annotations'][i]['image_id']).zfill(12) + '.jpg'
        if filename not in captions:
            captions[filename] = data['annotations'][i]['caption']
    captions = {k: v for k, v in sorted(captions.items())}

    with open(args.save_path, 'w') as f:
        for filename, prompt in captions.items():
            line = dict(
                source=f'source/{filename}',
                target=f'target/{filename}',
                prompt=prompt,
            )
            f.write(json.dumps(line) + '\n')
