import json
import argparse


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--ann_file', type=str, required=True)
    parser.add_argument('--save_path', type=str, required=True)
    parser.add_argument('--ann_person_file', type=str, default=None, help='filter person images')
    args = parser.parse_args()

    with open(args.ann_file, 'r') as f:
        data = json.load(f)

    image_ids = set(data['annotations'][i]['image_id'] for i in range(len(data['annotations'])))
    if args.ann_person_file is not None:
        with open(args.ann_person_file, 'r') as f:
            data_person = json.load(f)
        image_ids_person = set(data_person['annotations'][i]['image_id'] for i in range(len(data_person['annotations'])))
        image_ids = image_ids & image_ids_person

    captions = dict()
    for i in range(len(data['annotations'])):
        filename = str(data['annotations'][i]['image_id']).zfill(12) + '.jpg'
        if filename not in captions and data['annotations'][i]['image_id'] in image_ids:
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
