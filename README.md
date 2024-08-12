# ctrlora

Add LoRA to ControlNet.



## Installation

Clone this repo:

```shell
git clone https://github.com/xyfJASON/ctrlora.git
cd ctrlora
```

Create and activate a new conda environment:

```shell
conda create -n ctrlora python=3.10
conda activate ctrlora
```

Install pytorch and other dependencies:

```shell
pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu117
pip install -r requirements.txt
```

<br/>



## Datasets Preparation

### Custom dataset

Please put your custom data under `./data/custom_data_name` and organize it in the following structure:

```
data
└── custom_data_name
    ├── prompt.json
    ├── source
    │   ├── 0000.jpg
    │   ├── 0001.jpg
    │   └── ...
    └── target
        ├── 0000.jpg
        ├── 0001.jpg
        └── ...
```

- `source` contains condition images, such as canny edges, segmentation maps, depth images, etc.
- `target` contains ground-truth images corresponding to the condition images.
- Each line in `prompt.json` should follow the following format: `{"source": "source/0000.jpg", "target": "target/0000.jpg", "prompt": "The quick brown fox jumps over the lazy dog."}`.



### COCO 2017

Please download the training set (`train2017.zip`), validation set (`val2017.zip`) and annotations file (`annotations_trainval2017.zip`) from [here](https://cocodataset.org/#download). 
Unzip and organize the files as follows:

```
data
└── coco
    ├── annotations
    │   ├── captions_train2017.json
    │   └── captions_val2017.json
    ├── train2017
    │   ├── 000000000009.jpg
    │   ├── 000000000025.jpg
    │   └── ...
    └── val2017
        ├── 000000000139.jpg
        ├── 000000000285.jpg
        └── ...
```

Then, run the following commands to process the data:

```shell
python scripts/tool_resize_images.py --source ./data/coco/train2017 --target ./data/coco/train2017-resized
python scripts/tool_resize_images.py --source ./data/coco/val2017 --target ./data/coco/val2017-resized
python scripts/tool_get_prompt_coco.py --ann_file ./data/coco/annotations/captions_train2017.json --save_path ./data/coco/prompt-train.json
python scripts/tool_get_prompt_coco.py --ann_file ./data/coco/annotations/captions_val2017.json --save_path ./data/coco/prompt-val.json
```

After processing, the files should look like this:

```
data
└── coco
    ├── prompt-train.json
    ├── prompt-val.json
    ├── train2017-resized  (contains 118287 images)
    ├── val2017-resized    (contains 5000 images)
    └── ...
```

To use the coco dataset for training / evaluation, we need to organize it into the structure of a custom dataset. 
It is recommended to create symbolic links so that you don't need to copy the images.

Take `lineart` as an example:

```shell
mkdir ./data/coco-lineart-train
ln -s $(pwd)/data/coco/prompt-train.json ./data/coco-lineart-train/prompt.json
ln -s $(pwd)/data/coco/train2017-resized ./data/coco-lineart-train/target
python scripts/tool_make_cond_images.py --input_dir ./data/coco-lineart-train/target --output_dir ./data/coco-lineart-train/source --detector lineart
```

After running the above commands, the files should look like this:

```
data
└── coco-lineart-train
    ├── prompt.json (symbolic link)
    ├── source
    │   ├── 000000000009.jpg
    │   ├── 000000000025.jpg
    │   └── ...
    └── target (symbolic link)
        ├── 000000000009.jpg
        ├── 000000000025.jpg
        └── ...
```

So now the dataset can be used just like a custom dataset.



### MultiGen-20M

MultiGen-20M is a large image-prompt-condition dataset proposed by [UniControl](https://github.com/salesforce/UniControl). Please download the dataset from [here](https://console.cloud.google.com/storage/browser/sfr-unicontrol-data-research/dataset) and unzip it to `./data/MultiGen-20M`. The files should be organized as follows:

```
data
└── MultiGen-20M
    ├── conditions
    │   ├── aesthetics_6_25_plus_group_0_bbox
    │   ├── aesthetics_6_25_plus_group_0_blur
    │   ├── ...
    │   └── aesthetics_6_25_plus_group_9_segbase
    ├── images
    │   ├── aesthetics_6_25_plus_3m
    │   ├── aesthetics_6_plus_0
    │   ├── ...
    │   └── aesthetics_6_plus_3
    └── json_files
        ├── aesthetics_plus_all_group_bbox_all.json
        ├── aesthetics_plus_all_group_blur_all.json
        ├── ...
        └── aesthetics_plus_all_group_seg_all.json
```

<br/>



## Checkpoints Preparation

First, download the [Stable Diffusion v1.5 checkpoint](https://huggingface.co/runwayml/stable-diffusion-v1-5/tree/main) and put it in `./ckpts`. You only need to download `v1-5-pruned.ckpt`.

Then, make a ControlNet checkpoint initialized with SD UNet encoder by:

```shell
python scripts/tool_make_control_init.py --config ./configs/cldm_v15.yaml --sd_ckpt ./ckpts/v1-5-pruned.ckpt --output_path ./ckpts/control_sd15_init.pth
```

<br/>



## Pretrain the Base ControlNet

```shell
python scripts/train_ctrlora_pretrain.py \
    --dataroot DATAROOT \
    --config CONFIG \
    --sd_ckpt SD_CKPT \
    --cn_ckpt CN_CKPT \
    [--name NAME] \
    [--lr LR] \
    [--bs BS] \
    [--max_steps MAX_STEPS] \
    [--gradacc GRADACC] \
    [--precision PRECISION] \
    [--save_memory] \
    [--img_logger_freq IMG_LOGGER_FREQ] \
    [--ckpt_logger_freq CKPT_LOGGER_FREQ]
```

Arguments related to dataset:

- `--dataroot`: Path to the MultiGen-20M dataset, e.g, `./data/MultiGen-20M`.

Arguments related to model:

- `--config`: Path to the config file, e.g., `./configs/ctrlora_pretrain_sd15_9tasks_rank128.yaml`.
- `--sd_ckpt`: Path to the Stable Diffusion checkpoint, e.g., `./ckpts/v1-5-pruned.ckpt`.
- `--cn_ckpt`: Path to the ControlNet checkpoint, e.g., `./ckpts/control_sd15_init.pth`.

Arguments related to training:

- `--name`: Optional. Name of the experiment. The logging directory will be `./runs/name`. Default: current time.
- `--lr`: Optional. Learning rate. Default: `1e-5`.
- `--bs`: Optional. Batch size on each process. Default: `4`.
- `--max_steps`: Optional. Maximum number of training steps. Default: `700000`.
- `--gradacc`: Optional. Gradient accumulation. Default: `1`.
- `--precision`: Optional. Precision. Default: `32`.
- `--save_memory`: Optional. Save memory by using sliced attention. Default: `False`.
- `--img_logger_freq`: Optional. Frequency of logging images. Default: `10000`.
- `--ckpt_logger_freq`: Optional. Frequency of saving checkpoints. Default: `10000`.

The training logs and checkpoints will be saved to `./runs/name`.

For example, to train BaseControlNet-9tasks-700ksteps with 8 RTX 4090 GPUs and a total batch size of 32:

```shell
python scripts/train_ctrlora_pretrain.py --dataroot ./data/MultiGen-20M --config ./configs/ctrlora_pretrain_sd15_9tasks_rank128.yaml --sd_ckpt ./ckpts/v1-5-pruned.ckpt --cn_ckpt ./ckpts/control_sd15_init.pth --bs 1 --gradacc 4 --save_memory --max_steps 700000
```

<br/>



## Finetune the Base ControlNet (with lora or full-params)

```shell
python scripts/train_ctrlora_finetune.py \
    --dataroot DATAROOT \
    [--drop_rate DROP_RATE] \
    [--multigen20m] \
    [--task TASK] \
    --config CONFIG \
    --sd_ckpt SD_CKPT \
    --cn_ckpt CN_CKPT \
    [--name NAME] \
    [--lr LR] \
    [--bs BS] \
    [--max_steps MAX_STEPS] \
    [--gradacc GRADACC] \
    [--precision PRECISION] \
    [--save_memory] \
    [--img_logger_freq IMG_LOGGER_FREQ] \
    [--ckpt_logger_freq CKPT_LOGGER_FREQ]
```

Arguments related to custom dataset:

- `--dataroot`: Path to the dataset.
- `--drop_rate`: Optional. Drop rate for classifier-free guidance. Default: 0.3.

Arguments related to MultiGen-20M dataset:

- `--multigen20m`: Set this flag to use MultiGen-20M.
- `--dataroot`: Path to the MultiGen-20M dataset, e.g., `./data/MultiGen-20M`.
- `--drop_rate`: Optional. Drop rate for classifier-free guidance. Default: 0.3.
- `--task`: Task to train on. Choices: `{'hed', 'canny', 'seg', 'depth', 'normal', 'openpose', 'hedsketch', 'bbox', 'outpainting', 'inpainting', 'blur', 'grayscale'}`.

Arguments related to model:

- `--config`: Path to the config file, e.g., `./configs/ctrlora_finetune_sd15_rank128.yaml`.
- `--sd_ckpt`: Path to the Stable Diffusion checkpoint, e.g., `./ckpts/v1-5-pruned.ckpt`.
- `--cn_ckpt`: Path to the ControlNet checkpoint, e.g., `./ckpts/control_sd15_init.pth`.

Arguments related to training:

- `--name`: Optional. Name of the experiment. The logging directory will be `./runs/name`. Default: current time.
- `--lr`: Optional. Learning rate. Default: `1e-5`.
- `--bs`: Optional. Batch size. Default: `1`.
- `--max_steps`: Optional. Maximum number of training steps. Default: `100000`.
- `--gradacc`: Optional. Gradient accumulation. Default: `1`.
- `--precision`: Optional. Precision. Default: `32`.
- `--save_memory`: Optional. Save memory by using sliced attention. Default: `False`.
- `--img_logger_freq`: Optional. Frequency of logging images. Default: `1000`.
- `--ckpt_logger_freq`: Optional. Frequency of saving checkpoints. Default: `1000`.

The training logs and checkpoints will be saved to `./runs/name`.

<br/>



## Sample images

```shell
python sample.py --dataroot DATAROOT \
                 [--multigen20m] \
                 [--task TASK] \
                 --config CONFIG \
                 --ckpt CKPT \
                 --n_samples N_SAMPLES \
                 --save_dir SAVE_DIR \
                 [--ddim_steps DDIM_STEPS] \
                 [--ddim_eta DDIM_ETA] \
                 [--strength STRENGTH] \
                 [--cfg CFG]
```

Arguments related to custom dataset:

- `--dataroot`: Path to the dataset.

Arguments related to MultiGen-20M dataset:

- `--multigen20m`: Set this flag to use MultiGen-20M.
- `--dataroot`: Path to the MultiGen-20M dataset, e.g., `./data/MultiGen-20M`.
- `--task`: Task to test on. Choices: `{'hed', 'canny', 'seg', 'depth', 'normal', 'openpose', 'hedsketch', 'bbox', 'outpainting', 'inpainting', 'blur', 'grayscale'}`.

Arguments related to model:

- `--config`: Path to the config file. e.g., `./configs/ctrlora_finetune_sd15_rank128.yaml`.
- `--ckpt`: Path to the checkpoint, e.g., `./runs/xxx/lightning_logs/version_xxx/checkpoints/xxx.ckpt`.
- `--n_samples`: Number of samples to generate.
- `--save_dir`: Directory to save the generated images.
- `--ddim_steps`: Optional. Number of DDIM steps. Default: `50`.
- `--ddim_eta`: Optional. DDIM eta. Default: `0.0`.
- `--strength`: Optional. Strength of the ControlNet. Default: `1.0`.
- `--cfg`: Optional. Strength of classifier-free guidance. Default: `7.5`.
