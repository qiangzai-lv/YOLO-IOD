import argparse
import json
import os
import warnings

from mmdet.apis import init_detector
from pycocotools.coco import COCO
from inference import inference_detector
from tqdm import tqdm

warnings.filterwarnings('ignore')

cls_coco = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush']

def calculate_iou(box1, box2):
    # 计算两个边界框的交集部分
    x_left = max(box1[0], box2[0])
    y_top = max(box1[1], box2[1])
    x_right = min(box1[2], box2[2])
    y_bottom = min(box1[3], box2[3])

    if x_right < x_left or y_bottom < y_top:
        return 0.0  # 如果没有交集则返回0

    # 计算交集面积
    intersection_area = (x_right - x_left) * (y_bottom - y_top)

    # 计算并集面积
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union_area = box1_area + box2_area - intersection_area

    # 计算IoU
    iou = intersection_area / union_area
    return iou


def calculate_max_iou(bbox, bboxs):
    max_iou = 0
    for b in bboxs:
        x1, y1, w, h = b
        x2, y2 = x1 + w, y1 + h
        gt = [x1, y1, x2, y2]
        iou = calculate_iou(bbox, gt)
        if iou > max_iou:
            max_iou = iou
    return max_iou


class Output:
    def __init__(self, xyxy: list, scores: list, cls: list, path):
        self.xyxy = xyxy
        self.scores = scores
        self.cls = cls
        self.path = path


class MmdetModel:
    def __init__(self, cfg_path, pt_path, class_json, skip_scores=0.5) -> None:
        self.cfg_path = cfg_path
        self.pt_path = pt_path
        self.skip_scores = skip_scores
        # 原来映射
        self.class_names = cls_coco
        self.model = init_detector(self.cfg_path, self.pt_path)
        self.class_texts = json.load(open(class_json, 'r'))

    def predict(self, img_path):
        result = inference_detector(self.model, img_path, self.class_texts)
        labels = result.pred_instances.labels
        bboxes = result.pred_instances.bboxes
        scores = result.pred_instances.scores
        ins = scores > self.skip_scores
        bboxes = bboxes[ins, :]
        labels = labels[ins]
        scores = scores[ins]
        return Output(bboxes, scores.tolist(), [self.class_names[cls_idx] for cls_idx in labels], img_path)


def main(mmdet_cfg, mmdet_pt, class_json, ann_file, ann_save_file, img_path, skip_scores, iou_thr):

    model = MmdetModel(mmdet_cfg, mmdet_pt, skip_scores=skip_scores, class_json = class_json)
    # 加载 COCO 数据集
    coco = COCO(ann_file)

    # 加载类别信息
    cats = coco.loadCats(coco.getCatIds())
    categories_map = {}

    for cat in cats:
        categories_map[cat['name']] = cat['id']

    # 加载 COCO 数据集
    coco = COCO(ann_file)
    ann_save = json.load(open(ann_file))

    # score 置为1
    for ann in ann_save['annotations']:
        ann['score'] = 1.0

    max_id = 0
    for ann in ann_save["annotations"]:
        max_id = max(max_id, ann["id"])

    # 获取所有图片
    image_ids = coco.getImgIds()
    print(f"数据集中共有 {len(image_ids)} 张图片")
    for idx, image_id in tqdm(enumerate(image_ids)):
        image_info = coco.loadImgs([image_id])[0]
        # 加载图片
        image_path = os.path.join(img_path, image_info['file_name'])
        # 获取标注信息
        ann_ids = coco.getAnnIds(imgIds=[image_info['id']])
        annotations = coco.loadAnns(ann_ids)
        bboxs_gt = [ann['bbox'] for ann in annotations]
        result = model.predict(image_path)

        boxes, types, scores = result.xyxy, result.cls, result.scores,
        for box, cls, score in zip(boxes, types, scores):
            box = box.tolist()

            if cls not in categories_map:
                continue

            max_iou = calculate_max_iou(box, bboxs_gt)
            cls = categories_map[cls]
            # print(image_path + f' max_iou:{max_iou}')
            if max_iou < iou_thr:
                # 创建object元素
                max_id += 1
                x1, y1, x2, y2 = box
                w = x2 - x1
                h = y2 - y1
                ann = {'image_id': image_id, 'bbox': [x1, y1, w, h], 'category_id': cls, 'id': max_id,
                       'iscrowd': 0,
                       'area': w * h,
                       'segmentation': [[
                           x1, y1,  # 左上角
                           x2, y1,  # 右上角
                           x2, y2,  # 右下角
                           x1, y2  # 左下角
                       ]],
                       'score': score
                       }
                ann_save["annotations"].append(ann)

    # 将数据写入名为 'data.json' 的文件中
    with open(ann_save_file, 'w') as f:
        json.dump(ann_save, f)

def parse_args():
    parser = argparse.ArgumentParser(description='Train a detector')
    parser.add_argument('--ann_file', help='train config file path')
    parser.add_argument('--mmdet_cfg', help='the dir to save logs and models')
    parser.add_argument('--mmdet_pt', help='the dir to save logs and models')
    parser.add_argument('--class_json', help='the dir to save logs and models')
    args = parser.parse_args()
    return args

if __name__ == '__main__':

    args = parse_args()
    ann_file = args.ann_file
    img_path = 'data/coco/train2017'
    ann_save_file = ann_file.replace('.json', '_ps.json')
    mmdet_cfg = args.mmdet_cfg
    mmdet_pt = args.mmdet_pt
    class_json = args.class_json
    main(mmdet_cfg, mmdet_pt, class_json, ann_file, ann_save_file, img_path, 0.1, 0.5)
