# YOLO-IOD: Towards Real Time Incremental Object Detection

This repository contains the official implementation of YOLO-IOD, a real-time incremental object detection framework.

## Installation

Please follow the installation instructions from [YOLO-World](https://github.com/AILab-CVC/YOLO-World) to set up the environment. 

## Data Preparation

**Organize the dataset structure:**

```
data/
└── coco/
    │── train2017/
    │── val2017/
    └── annotations/
        ├── instances_train2017.json
        └── instances_val2017.json
```

**setting split**

```
python utils/coco2017_split.py
```

## Pre-trained Models

YOLO-IOD is constructed upon the pre-trained YOLO-World model. You will need to download the appropriate pre-trained YOLO-World weights. Please refer to the [YOLO-World official repository](https://github.com/xxxxx/YOLO-World) .

Place the pre-trained weights `x_stage1-62b674ad.pth` in a designated `weights/` directory:

## Unknown Pseudo label

```
python utils/pseudo_unknown_label.py
```

## Train

```
bash run_40+40.sh
```

