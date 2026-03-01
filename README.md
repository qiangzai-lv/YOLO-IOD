# YOLO-IOD: Towards Real Time Incremental Object Detection

Official Pytorch implementation for "*YOLO-IOD: Towards Real Time Incremental Object Detection*", AAAI 2026 Poster. [[Paper](https://arxiv.org/abs/2512.22973)]

![image-20250818142604338](./assets/framework.png)

## ✨ Contributions

- We introduce YOLO-IOD, an integrated and real-time IOD framework, and pinpoint three causes of forgetting: foreground-background confusion, parameter interference, and misaligned knowledge distillation conflict.
- YOLO-IOD incorporates three modules (CPR, IKS, and CAKD) to mitigate forgetting, with particular emphasis on the dual-teacher CAKD module that alleviates distillation misalignment using both former and current teacher detection heads.
- We introduce LoCo COCO, a practical benchmark designed to remove image overlap between stages and consider category co-occurrence, allowing for a more equitable assessment of incremental object detection.

## 🚀 Get Started

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

#### Pre-trained Models

YOLO-IOD is built upon the pre-trained YOLO-World model. Please download the required YOLO-World checkpoint from the following link: [x_stage1-62b674ad.pth](https://huggingface.co/wondervictor/YOLO-World-V2.1/resolve/main/x_stage1-62b674ad.pth). After downloading, place the checkpoint file in the weights/ directory:

## 📦 Dataset

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

**The traditional COCO setting split**

```bash
python script/coco2017_split.py --pattern 40+40
```

## 🧪 LOCO COCO 

**Data Split:** Generate the LOCO COCO incremental split from the original COCO dataset:

```bash
python script/loco_coco_split.py --pattern 40+40
```

After running the script, annotation files will be created as:

```
data/
└── coco/
    ├── train2017/
    ├── val2017/
    └── loco_annotations/
        └── 40+40/
            └── instances_train2017_part0.json
```

For convenience, we also provide pre-generated LOCO COCO annotations: **[Download: LOCO COCO](https://drive.google.com/drive/folders/1uz7UA66hEabZp7MVVScnja11u7t71FCr?usp=sharing)** 

Place the downloaded files under: `data/coco/loco_annotations/`

## ⚙️ Training & Evaluation

#### Traditional COCO Setting

- **Base Stage Training**: 

```bash
# Generate unknown pseudo labels using the CPR module
python script/cpr_unknown_pseudo_label.py --setting COCO --task 40+40 --stage 0 --num_clusters 30
# Train the detector on the base categories
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

#### LOCO COCO Training

- **Base Stage Training**: 

```bash
python script/cpr_unknown_pseudo_label.py --setting LOCO_COCO --task 40+40 --stage 0 --num_clusters 30
bash tools/dist_train_gps.sh configs/loco_40_40/yolo_iod_loco_coco_40_40_task0.py 4 --amp
```

- **Incremental Training**

```bash
python script/pseudo_label_sc.py --setting LOCO_COCO --task 40+40 --stage 1
bash tools/dist_train_gps.sh configs/loco_40_40/yolo_iod_loco_coco_40_40_stage1.py 4 --amp
bash tools/dist_train_gps.sh configs/loco_40_40/yolo_iod_loco_coco_40_40_task1.py 4 --amp
```

## **📈 LOCO COCO Results**

We report Incremental Object Detection performance on **LOCO COCO** under different incremental settings. **CoGap** denotes the AP drop compared with the original COCO partition (lower is better).

### **🔹 40 + 40 Setting**

| **Method**          | **Baseline**    | **AP**   | **AP50** | **AP75** | **CoGap ↓** |
| ------------------- | --------------- | -------- | -------- | -------- | ----------- |
| RGR                 | Faster R-CNN    | 35.0     | 55.7     | 37.2     | **0.6%**    |
| CL-DETR             | Deformable DETR | 40.9     | 58.8     | 43.8     | 1.1%        |
| GCD                 | Grounding DINO  | 44.7     | 61.4     | 48.7     | 1.0%        |
| **YOLO-IOD (Ours)** | YOLO-World (X)  | **52.2** | **68.7** | **57.3** | 0.8%        |

### **🔹 70 + 10 Setting**

| **Method**          | **Baseline**    | **AP**   | **AP50** | **AP75** | **CoGap ↓** |
| ------------------- | --------------- | -------- | -------- | -------- | ----------- |
| RGR                 | Faster R-CNN    | 34.6     | 54.7     | 37.4     | 2.0%        |
| CL-DETR             | Deformable DETR | 39.6     | 56.0     | 41.2     | 1.8%        |
| GCD                 | Grounding DINO  | 44.8     | 61.6     | 48.7     | 1.9%        |
| **YOLO-IOD (Ours)** | YOLO-World (X)  | **50.7** | **67.0** | **55.6** | **1.7%**    |

### **🔹 40 − 20 Setting**

| **Method**          | **Baseline**    | **AP**   | **AP50** | **AP75** | **CoGap ↓** |
| ------------------- | --------------- | -------- | -------- | -------- | ----------- |
| RGR                 | Faster R-CNN    | 32.5     | 52.3     | 35.0     | 1.8%        |
| CL-DETR             | Deformable DETR | 33.6     | 50.1     | 36.4     | 1.7%        |
| GCD                 | Grounding DINO  | 42.4     | 58.0     | 46.2     | 1.6%        |
| **YOLO-IOD (Ours)** | YOLO-World (X)  | **50.9** | **66.9** | **55.7** | **1.0%**    |

## Acknowledgement

Our code is based on the project [[YOLO-World](https://github.com/AILab-CVC/YOLO-World)]

## Citation

Please cite our paper if this repo helps your research:

```
@inproceedings{zhang2026yoloiod,
  title     = {YOLO-IOD: Towards Real-Time Incremental Object Detection},
  author    = {Zhang, Shizhou and Lv, Xueqiang and Xing, Yinghui and Wu, Qirui and Xu, Di and Zhao, Chen and Zhang, Yanning},
  booktitle = {Proceedings of the AAAI Conference on Artificial Intelligence},
  year      = {2026},
}
```
