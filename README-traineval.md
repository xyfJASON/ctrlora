# Instructions for Training and Evaluation

This document provides detailed instructions on how to train and evaluate the proposed Base ControlNet and LoRAs.



## Datasets Preparation

### COCO 2017

We use the COCO 2017 training set for most of the new conditions presented in the paper, such as Lineart, Palette, etc.
We use the COCO 2017 validation set for all quantitative evaluations in the paper. You don't need to prepare this dataset if you just want to train the model on your custom dataset.

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

Take Lineart as an example:

```shell
COND=lineart
mkdir ./data/coco-$COND-train
ln -s $(pwd)/data/coco/prompt-train.json ./data/coco-$COND-train/prompt.json
ln -s $(pwd)/data/coco/train2017-resized ./data/coco-$COND-train/target
python scripts/tool_make_cond_images.py --input_dir ./data/coco-$COND-train/target --output_dir ./data/coco-$COND-train/source --detector $COND
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

MultiGen-20M is a large image-prompt-condition dataset proposed by [UniControl](https://github.com/salesforce/UniControl). We use this dataset for training our Base ControlNet.

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

For example, to train BaseControlNet on 9 tasks for 700k steps with 8 RTX 4090 GPUs and a total batch size of 32:

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

For example, to finetune the Base ControlNet on Lineart with a lora rank of 128 for 1000 steps:

```shell
python scripts/train_ctrlora_finetune.py --dataroot ./data/coco-lineart-train --config ./configs/ctrlora_finetune_sd15_rank128.yaml --sd_ckpt ./ckpts/sd15/v1-5-pruned.ckpt --cn_ckpt ./ckpts/ctrlora-basecn/ctrlora_sd15_basecn700k.ckpt --max_steps 1000
```

**Extract LoRAs**: During training, the saved checkpoints contain all the components of the model including Stable Diffusion, Base ControlNet and LoRAs. To extract LoRAs from a checkpoint, you can run the following command:

```shell
python scripts/tool_extract_weights.py -t lora --ckpt CHECKPOINT --save_path SAVE_PATH
```

- `--ckpt`: Path to the checkpoint.
- `--save_path`: Path to save the extracted LoRAs.

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

<br/>



## Quantitative Evaluation

For control-type conditions including Canny, HED, Sketch, Depth, Normal, Segmentation, Skeleton, Lineart, Palette and Densepose:

```shell
python scripts/evaluate_control.py --sample_dir SAMPLE_DIR --detector DETECTOR 
```

- `--sample_dir`: Path to the directory containing the generated images.
- `--detector`: Detector type. Choices: `{'canny', 'hed', 'seg', 'depth', 'normal', 'openpose', 'hedsketch', 'lineart', 'lineart_anime', 'palette', 'densepose'}`.

For restoration-type conditions including Outpainting, Inpainting, and Dehazing:

```shell
python scripts/evaluate_restore.py --sample_dir SAMPLE_DIR
```

- `--sample_dir`: Path to the directory containing the sampled images.

To evaluate the image quality, we use `torch-fidelity` to compute FID and Inception Score:

```shell
fidelity --gpu 0 --fid --isc --input1 INPUT1 --input2 INPUT2
```

- `--input1`: Path to the directory containing the generated images.
- `--input2`: Path to the directory containing the ground-truth images.
