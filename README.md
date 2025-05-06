# StereoRBHM# Get Started

## Installation

1. clone this repo.
    ```
    https://github.com/XiandaGuo/OpenStereo
    ```
2. Install dependenices:
    - pytorch >= 1.13.1
    - torchvision
    - timm == 0.5.4
    - pyyaml
    - tensorboard
    - opencv-python
    - tqdm
    - scikit-image

   Create a conda environment by:
   ```
   conda create -n openstereo python=3.8 
   ```
   
   Install pytorch by [Anaconda](https://pytorch.org/get-started/locally/):
   ```
   conda install pytorch==1.13.1 torchvision==0.14.1 torchaudio==0.13.1 pytorch-cuda=11.7 -c pytorch -c nvidia
   ```
   Install other dependencies by pip:
   ```
   pip install -r requirements.txt
   ```

## Prepare dataset


## Get trained model

Go to the [model zoom](https://pan.baidu.com/s/1iHdBTdyuTUcr4vX9N0exqg?pwd=1e2k), download the model file and uncompress it to output.

## Train
python tools/train.py --cfg_file cfgs/rbhm/igevrbhm.py
Train a model with a Single GPU
```
python tools/train.py --cfg_file cfgs/rbhm/rbhm.py
```
Multi-GPU Training on Single Node
```
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
torchrun --nnodes=1 --nproc_per_node=8 --rdzv_backend=c10d --rdzv_endpoint=localhost:23456 tools/train.py --dist_mode --cfg_file cfgs/rbhm/rbhm.py
```
- `--config` The path to the config file.
- `--dist_mode` If specified, the program will use DDP to train.
- your exp will saved in '/save_root_dir/DATASET_NAME/MODEL_NAME/config_file_perfix/extra_tag', save_root_dir and extra_tag can specified in train argparse

Train HeightPredict Moudle
```
python tools/train.py --cfg_file cfgs/rbhm/rbhm.py
```

Train through HeightPredict Moudle
```
python tools/train.py --cfg_file cfgs/rbhm/igevrbhm.py
```
you should change the HeightPredict Moudle path according to your path in the  cfgs/rbhm/igevrbhm.py


## Val

Evaluate the trained model by
```
python tools/eval.py --cfg_file cfgs/rbhm/rbhm.py --pretrained_model "/file_system/vepfs/algorithm/ruilin.wang/code/LightStereoX/output/SpeedBumpDataset/RBHM/rbhm_v4/ckpt/epoch_34/pytorch_model.bin"
```
Generalization Evaluation:
```
python tools/eval.py --cfg_file cfgs/rbhm/igevrbhm.py --pretrained_model "/file_system/vepfs/algorithm/ruilin.wang/code/LightStereoX/output/SpeedBumpDataset/COEX/coex_rbhm_v4_new/ckpt/epoch_60/pytorch_model.bin"

python tools/eval.py --cfg_file cfgs/rbhm/coexrbhm.py --pretrained_model "/file_system/vepfs/algorithm/ruilin.wang/code/LightStereoX/output/SpeedBumpDataset/COEX/coex_rbhm_v4_new/ckpt/epoch_60/pytorch_model.bin"
```

- `--cfg_file` The path to the config file.
- `--eval_data_cfg_file` The dataset config you want to eval.
- `--pretrained_model` your pre-trained checkpoint
- HeightPredict Moudle is the same as the one in train

**Tip**: Other arguments are the same as the train phase.

## Infer

Infer the trained model by
```
python tools/infer.py --cfg_file cfgs/rbhm/igevrbhm.py --pretrained_model "/file_system/vepfs/algorithm/ruilin.wang/code/LightStereoX/output/SpeedBumpDataset/COEX/coex_rbhm_v4_new/ckpt/epoch_60/pytorch_model.bin"
```
**Tip**: the pretrained_model should be written in cfg_file, and if you want to process multiple image pairs at once, please organize the file structure and write a simple loop.


## Customize

## Other
1. You can set the default pre-trained model path, by `export TORCH_HOME="/path/to/pretrained_models"`
