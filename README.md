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

### MultiGen-20M

Please download the dataset from [here](https://console.cloud.google.com/storage/browser/sfr-unicontrol-data-research/dataset) and unzip it to `./data/MultiGen-20M`. The files should be organized as follows:

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

- `--config`: Path to the config file, e.g., `./configs/ctrlora_pretrain_sd15_9tasks_lora128.yaml`.
- `--sd_ckpt`: Path to the Stable Diffusion checkpoint, e.g., `./ckpts/v1-5-pruned.ckpt`.
- `--cn_ckpt`: Path to the ControlNet checkpoint, e.g., `./ckpts/control_sd15_init.pth`.

Arguments related to training:

- `--lr`: Optional. Learning rate. Default: `1e-5`.
- `--bs`: Optional. Batch size on each process. Default: `4`.
- `--max_steps`: Optional. Maximum number of training steps. Default: `800000`.
- `--gradacc`: Optional. Gradient accumulation. Default: `1`.
- `--precision`: Optional. Precision. Default: `32`.
- `--save_memory`: Optional. Save memory by using sliced attention. Default: `False`.
- `--img_logger_freq`: Optional. Frequency of logging images. Default: `1000`.
- `--ckpt_logger_freq`: Optional. Frequency of saving checkpoints. Default: `10000`.

The training logs and checkpoints will be saved to `./lightning_logs/version_xxx/`.

For example, to train BaseControlNet-9tasks-800ksteps with 8 RTX 4090 GPUs and a total batch size of 32:

```shell
python scripts/train_ctrlora_pretrain.py --dataroot ./data/MultiGen-20M --config ./configs/ctrlora_pretrain_sd15_9tasks_rank128.yaml --sd_ckpt ./ckpts/v1-5-pruned.ckpt --cn_ckpt ./ckpts/control_sd15_init.pth --bs 2 --gradacc 2 --save_memory --max_steps 800000
```

## Finetune the Base ControlNet (full-params or with lora)

```shell
python scripts/train_ctrlora_finetune.py \
    --dataroot DATAROOT \
    [--drop_rate DROP_RATE] \
    [--multigen20m] \
    [--coco] \
    [--task TASK] \
    --config CONFIG \
    --sd_ckpt SD_CKPT \
    --cn_ckpt CN_CKPT \
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

Arguments related to COCO dataset:

- `--coco`: Set this flag to use COCO.
- `--dataroot`: Path to the COCO dataset, e.g., `./data/coco`.
- `--drop_rate`: Optional. Drop rate for classifier-free guidance. Default: 0.3.
- `--task`: Task to train on. Choices: `{'jpeg', 'palette', 'depth', 'inpainting', 'blur', 'grayscale'}`.

Arguments related to model:

- `--config`: Path to the config file, e.g., `./configs/finetune_sd15_lora128.yaml`.
- `--sd_ckpt`: Path to the Stable Diffusion checkpoint, e.g., `./ckpts/v1-5-pruned.ckpt`.
- `--cn_ckpt`: Path to the ControlNet checkpoint, e.g., `./ckpts/control_sd15_init.pth`.

Arguments related to training:

- `--lr`: Optional. Learning rate. Default: `1e-5`.
- `--bs`: Optional. Batch size. Default: `1`.
- `--max_steps`: Optional. Maximum number of training steps. Default: `100000`.
- `--gradacc`: Optional. Gradient accumulation. Default: `1`.
- `--precision`: Optional. Precision. Default: `32`.
- `--save_memory`: Optional. Save memory by using sliced attention. Default: `False`.
- `--img_logger_freq`: Optional. Frequency of logging images. Default: `1000`.
- `--ckpt_logger_freq`: Optional. Frequency of saving checkpoints. Default: `1000`.

The training logs and checkpoints will be saved to `./lightning_logs/version_xxx/`.
