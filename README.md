# UprightRL: Upright orientation estimation of 3D shapes via reinforcement learning

![overview](./pretrain/upright.jpg)

## Introduction

In this paper, we study the problem of 3D shape upright orientation estimation from the perspective of reinforcement learning,
i.e. we teach a machine (agent) to orientate 3D shapes step by step to upright given its current observation. Unlike previous
methods, we take this problem as a sequential decision-making process instead of a strong supervised learning problem. To
achieve this, we propose UprightRL, a deep network architecture designed for upright orientation estimation.

For more details and materials, please refer to our [project page](https://vcc.tech/research/2021/UprightRL).


## Dependencies
Check [`requirements.txt`](./requirements.txt) for more details.


## Usage

All codes are tested on Ubuntu 18.04 and Windows 10. 

Before you run the following scripts, please prepare the dataset you required (introduced below)
### Training
```
./scripts/train.sh
```
### Testing
```
./scripts/test.sh
```
### Render ( check [render folder](./render/README.md) for detial )
```
cd render/
blender -b render_demo.blend -P render_single_pcd.py
```

## Network structure:

Layer | Input dim | Output dim
:-: | :-: | :-:
PointNet Encoder | (3, 2048) | 1024 |
FC | 1024 | 512 |
Batch Normalize | - | - |
FC | 512 | 256 |
Batch Normalize | - | - |
FC | 256 | 128 |
Batch Normalize | - | - |
LSTM | 128 | 128 |
Actor FC | 128 | 5 |
Critic FC | 128 | 1 |

## File structure:

* **code/**: Main code folder.
* **render/**: Render code.
* **dataset/**: Dataset folder, you can download our trianing/testing dataset from [google drive](https://drive.google.com/drive/folders/1QbMgPpWqxfrqKvdx8-i_ZRFbElab7rCe?usp=sharing).
* **pretrain/**: Model we trained before.
* **scripts/**: Some scripts to train/test network.

### Parameters of main.py
    ========= Task setting =========
    --task: string, the task to do
        "test": test the network without training
        "train": train the network

    --n_epochs: int, epoch num
    --batch_size: int, batch size, default as 128

    --num_steps: int, number of forward steps in uprightRL model
    --rot_angle: int, angle step to use for each action update
    --num_point: int, point cloud num of input

    ========= Data setting =========
    --data_type: string, the dataset type
        'complete': complete dataset
        'partial': partial dataset
        'single_scan': single scan dataset
        'uprl': dataset from [LZL16]

    --data_dir: string, the path of dataset

    ========= Result folder setting =========

    --result_dir: string, the folder will save the results

    --note: string, any words you want to say
        When the main.py is running, it will create a folder under to store the result, and the note string will be a part of the folder's name

    --pretrain_dir: string, the location of pre-trained network model
        If the path of a network model is "./model/best.pth", then the pretrain_dir should be "./model/"
    

Example:

    python ./code/main.py \
        --cuda='0' \
        --task='train' \
        --num_step=20 \
        --n_epochs=400 \
        --batch_size=4 \
        --rot_angle=4 \
        --data_type='single_scan' \
        --data_dir='./dataset' \
        --result_dir='./outputs' \
        --note='example'

This command means you want to train the network in **400** epochs, batch size is **4**, using the **single_scan** dataset in **./dataset** folder. The max steps num is **20** steps, each rotation angle step is **4** degree.

In our code, the GPU memory is mainly affected by the `batch_size` and the `num_step`, we test on an NVIDIA TITAN Xp, it takes about 7 GB to execute the `train.sh`.

Then it will generate a folder unber **./outputs** folder to store the training or testing result.
The folder structure of `./outputs` may looks like this:

    outputs/
    ????????? train/   <--- your task
        ????????? xxxx/  <--- the folder create by our code
            ????????? checkpoints/     <--- store models of each epoch
            ????????? logs/            <--- some log info
            ????????? plots/           <--- some reward / loss curve
            ????????? xxx.h5           <--- the test/val result

## Dataset
The `dataset/` folder contain the data, the folder structure of `dataset` looks like this:

    dataset/
    ????????? shapenet_complete/
    |   ????????? xxxxxx.h5
    ????????? shapenet_partial/
    ????????? shapenet_single_scan/
    ????????? upright16_data/
        ????????? testTTA/

`shapenet_complete` is the complete dataset, download from [google drive](https://drive.google.com/file/d/1loWWDxAyAKUZ6Q8KWys9SYYxwD1GR2Ld/view?usp=sharing)

`shapenet_partial` is the partial dataset, download for [google drive](https://drive.google.com/file/d/1FKsL9gS32wgvHROav7jR0QXchKZudKsw/view?usp=sharing)

`shapenet_single_scan` is the single scan dataset, download from [google drive](https://drive.google.com/file/d/1E06efZAo4y0vUFJvpZbTWRisyvHke8Ny/view?usp=sharing)

`upright16_data` is the dataset from [LZL16](https://github.com/zishun/UprightOrientation).

## Citation
Please cite the paper in your publications if it helps your research:
```
@article{UPRL21,
title = {Learning Elastic Constitutive Material and Damping Models},
author = {Luanmin Chen and Juzhan Xu and Chuan Wang and Haibin Huang and Hui Huang and Ruizhen Hu},
journal = {Computer Graphics Forum (Proceedings of Pacific Graphics 2021)},
volume = {40},
number = {7},
year = {2021},
}
```

## License
Our code is released under MIT License. See LICENSE file for details.
