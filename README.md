# YOLO-IOD: Towards Real Time Incremental Object Detection

Official Pytorch implementation for "*YOLO-IOD: Towards Real Time Incremental Object Detection*", AAAI 2026 Poster. [[Paper](https://arxiv.org/abs/2512.22973)]

![image-20250818142604338](./assets/framework.png)

## 🚀 Contributions

- We introduce YOLO-IOD, an integrated and real-time IOD framework, and pinpoint three causes of forgetting: foreground-background confusion, parameter interference, and misaligned knowledge distillation conflict.
- YOLO-IOD incorporates three modules (CPR, IKS, and CAKD) to mitigate forgetting, with particular emphasis on the dual-teacher CAKD module that alleviates distillation misalignment using both former and current teacher detection heads.
- We introduce LoCo COCO, a practical benchmark designed to remove image overlap between stages and consider category co-occurrence, allowing for a more equitable assessment of incremental object detection.

## Get Started

This repo is based on [YOLO-World](https://github.com/AILab-CVC/YOLO-World). Please follow the installation of YOLO-World and make sure you can run it successfully.

```bash
conda create -n yolo-iod python=3.10.16 -y
conda activate yolo-iod
# Please follow the official YOLO-World Installation Guide to set up this environment (PyTorch, MMCV, MMEngine, etc.).
cd our project
pip install -v -e .
cd third_party/mmyolo/
pip install -v -e .
```

## Dataset

**Organize the dataset structure:**

```bash
data/
└── coco/
    │── train2017/
    │── val2017/
    └── annotations/
        ├── instances_train2017.json
        └── instances_val2017.json
```

**setting split**

```bash
python script/coco2017_split.py --pattern 40+40
```

## Pre-trained Models

YOLO-IOD is built upon the pre-trained YOLO-World model. Please download the required YOLO-World checkpoint from the following link: [x_stage1-62b674ad.pth](https://huggingface.co/wondervictor/YOLO-World-V2.1/resolve/main/x_stage1-62b674ad.pth). After downloading, place the checkpoint file in the weights/ directory:

## Training & Evaluation

- Generate unknown pseudo labels using the CPR module before incremental training:

```bash
python script/cpr_unknown_pseudo_label.py --setting COCO --task 40+40 --stage 0
```

- **Base Stage Training：**Train the detector on the base categories

```bash
bash tools/dist_train_gps.sh configs/40_40/yolo_iod_coco_40_40_task0.py 4 --amp
```

- **Incremental Training**

```bash
# Generate CPR Pseudo Labels
python script/pseudo_label_sc.py --task 40+40 --stage 1
# Train Current-Stage Teacher (for CAKD)
bash tools/dist_train_gps.sh configs/40_40/yolo_iod_coco_40_40_stage1.py 4 --amp
# Incremental Learning
bash tools/dist_train_gps.sh configs/40_40/yolo_iod_coco_40_40_task1.py 4 --amp
```

## Acknowledgement

Our code is based on the project [[YOLO-World](https://github.com/AILab-CVC/YOLO-World)]

## Citation

Please cite our paper if this repo helps your research:

```
@InProceedings{Zhang_2025_CVPR,
    author    = {Zhang, Shizhou and Lv, Xueqiang and Xing, Yinghui and Wu, Qirui and Xu, Di and Zhang, Yanning},
    title     = {Revisiting Generative Replay for Class Incremental Object Detection},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    month     = {June},
    year      = {2025},
    pages     = {20340-20349}
}
```
