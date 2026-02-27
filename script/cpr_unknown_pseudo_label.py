import argparse
import json
import os
import warnings
from collections import Counter, defaultdict

import numpy as np
import torch
import tqdm
from mmdet.apis import init_detector
from pycocotools.coco import COCO
from sklearn.cluster import KMeans
from transformers import (AutoTokenizer, CLIPTextModelWithProjection)

from inference import inference_detector

warnings.filterwarnings('ignore')

classes = [
    "cherry",
    "soap",
    "toothbrush",
    "hair dryer",
    "keyboard",
    "strawberry",
    "pumpkin",
    "mop",
    "trash bin",
    "sushi",
    "shrimp",
    "remote",
    "pencil case",
    "skateboard",
    "belt",
    "flowers",
    "cat",
    "ruler",
    "pizza",
    "tricycle",
    "gun",
    "sheep",
    "shovel",
    "chainsaw",
    "donut",
    "truck",
    "animal",
    "desk",
    "violin",
    "dolphin",
    "flower",
    "sailboat",
    "golf ball",
    "spoon",
    "dessert",
    "candy",
    "ambulance",
    "Electronics",
    "marker",
    "bow tie",
    "tv",
    "lobster",
    "equipment",
    "kettle",
    "tennis",
    "apple",
    "helicopter",
    "wild animal",
    "train",
    "storage box",
    "air conditioner",
    "radish",
    "refrigerator",
    "barbell",
    "lighter",
    "lantern",
    "mango",
    "electric drill",
    "trolley",
    "lion",
    "swan",
    "toilet",
    "boat",
    "towel",
    "parrot",
    "balloon",
    "tennis racket",
    "butterfly",
    "paddle",
    "skis",
    "bathtub",
    "handbag",
    "mouse",
    "vegetables",
    "pen",
    "asparagus",
    "picture",
    "bicycle",
    "Vehicles",
    "camel",
    "noddles",
    "street lights",
    "teddy bear",
    "lamp",
    "helmet",
    "traffic light",
    "fire hydrant",
    "stool",
    "egg",
    "vase",
    "toiletry",
    "bowl",
    "Trees",
    "broom",
    "Plants",
    "hot-air balloon",
    "broccoli",
    "telephone",
    "pineapple",
    "stick",
    "surfboard",
    "baseball glove",
    "pear",
    "earphone",
    "bracelet",
    "carrot",
    "hamburger",
    "drum",
    "hammer",
    "baseball bat",
    "sports ball",
    "couch",
    "garlic",
    "van",
    "Appliances",
    "oven",
    "sandwich",
    "bread",
    "nightstand",
    "suitcase",
    "coconut",
    "soccer",
    "yak",
    "luggage",
    "egg tart",
    "brush",
    "candle",
    "parking meter",
    "cell phone",
    "basketball",
    "cup",
    "clock",
    "guitar",
    "steak",
    "giraffe",
    "tea pot",
    "mask",
    "person",
    "eggplant",
    "chicken",
    "durian",
    "motorcycle",
    "deer",
    "tomato",
    "bird",
    "eraser",
    "fire extinguisher",
    "Furniture",
    "bear",
    "mushroom",
    "skiboard",
    "onion",
    "pie",
    "table tennis",
    "mirror",
    "ice cream",
    "horse",
    "wine glass",
    "fork",
    "bed",
    "orange",
    "cake",
    "jellyfish",
    "corn",
    "comb",
    "volleyball",
    "microphone",
    "traffic sign",
    "crab",
    "stroller",
    "key",
    "table tennis paddle",
    "elephant",
    "sausage",
    "vegetable",
    "scissors",
    "scooter",
    "speaker",
    "flag",
    "car",
    "potted plant",
    "cd",
    "wallet",
    "cabbage",
    "stop sign",
    "umbrella",
    "kite",
    "tablet",
    "backpack",
    "potato",
    "tape",
    "domestic animal",
    "pot",
    "basket",
    "binoculars",
    "donkey",
    "dog",
    "book",
    "Food",
    "fire truck",
    "penguin",
    "dumpling",
    "Clothing",
    "airplane",
    "chopsticks",
    "pig",
    "carpet",
    "tent",
    "piano",
    "ship",
    "jug",
    "printer",
    "treadmill",
    "laptop",
    "baseball",
    "goose",
    "rabbit",
    "ring",
    "ladder",
    "rice",
    "peach",
    "chair",
    "knife",
    "vehicle",
    "necklace",
    "Sports",
    "washing machine",
    "cheese",
    "cow",
    "cucumber",
    "banana",
    "Stationery",
    "fruit",
    "fish",
    "scallop",
    "camera",
    "lipstick",
    "frisbee",
    "bench",
    "cookies",
    "fan",
    "monkey",
    "lemon",
    "wheelchair",
    "toaster",
    "pet",
    "bucket",
    "pillow",
    "chips",
    "dumbbell",
    "radiator",
    "pliers",
    "bottle",
    "Drinks",
    "grape",
    "zebra",
    "lettuce",
    "bus",
    "sink",
    "building",
    "tie",
    "watermelon",
    "hair drier",
    "faucet",
    "microwave",
    "duck",
    "snowboard",
    "hot dog",
    "dining table",
    "pomegranate"
]


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
        self.class_names = classes
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


def pseudo_unknown_label(ann_file, ann_save_file, img_path, skip_scores):
    mmdet_cfg = "configs/baseline/yolo_world_v2_x.py"
    mmdet_pt = './pretrain/x_stage1-62b674ad.pth'

    model = MmdetModel(mmdet_cfg, mmdet_pt, skip_scores=skip_scores, class_json='./data/texts/unknown_class_texts.json')
    # 加载 COCO 数据集
    coco = COCO(ann_file)
    ann_save = json.load(open(ann_file))

    # 转换为更结构化的JSON格式
    categories = [{"id": idx, "name": name} for idx, name in enumerate(classes, start=1)]

    # 加载类别信息
    categories_map = {}
    for cat in categories:
        categories_map[cat['name']] = cat['id']

    max_id = 0
    ann_save["categories"] = categories
    ann_save["annotations"] = []

    # 获取所有图片
    image_ids = coco.getImgIds()
    print(f"数据集中共有 {len(image_ids)} 张图片")
    for idx, image_id in tqdm(enumerate(image_ids)):
        image_info = coco.loadImgs([image_id])[0]
        # 加载图片
        image_path = os.path.join(img_path, image_info['file_name'])
        # 获取标注信息
        result = model.predict(image_path)

        boxes, types, scores = result.xyxy, result.cls, result.scores

        for box, cls, score in zip(boxes, types, scores):
            box = box.tolist()
            cls = categories_map[cls]
            # 创建object元素
            max_id += 1
            x1, y1, x2, y2 = box
            w = x2 - x1
            h = y2 - y1
            ann = {'image_id': image_id, 'bbox': [x1, y1, w, h], 'category_id': cls, 'id': max_id,
                   'iscrowd': 0,
                   'score': score,
                   'segmentation': [], 'area': w * h}
            ann_save["annotations"].append(ann)

    # 将数据写入名为 'data.json' 的文件中
    with open(ann_save_file, 'w') as f:
        json.dump(ann_save, f)


def compute_iou(box1, box2):
    """计算两个边界框的 IoU"""
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2

    xa = max(x1, x2)
    ya = max(y1, y2)
    xb = min(x1 + w1, x2 + w2)
    yb = min(y1 + h1, y2 + h2)

    inter_area = max(0, xb - xa) * max(0, yb - ya)
    box1_area = w1 * h1
    box2_area = w2 * h2
    union_area = box1_area + box2_area - inter_area

    return inter_area / union_area if union_area > 0 else 0


def merge_coco_annotations(original_coco_path, pseudo_coco_path, output_coco_path, iou_threshold=0.5,
                           score_threshold=0.5):
    # 加载原始和伪标签 COCO 格式数据
    with open(original_coco_path, 'r') as f:
        original_data = json.load(f)

    with open(pseudo_coco_path, 'r') as f:
        pseudo_data = json.load(f)

    # 类别映射
    category_name_to_id = {cat['name']: cat['id'] for cat in original_data['categories']}
    category_name_set = [cat['name'] for cat in original_data['categories']]
    pseudo_id_to_name = {cat['id']: cat['name'] for cat in pseudo_data['categories']}

    # 添加 unknown 类别
    max_category_id = max(category_name_to_id.values(), default=0)
    unknown_cat_id = max_category_id + 1
    original_data['categories'].append({'id': unknown_cat_id, 'name': 'unknown'})

    # 创建 image_id -> annotations 映射
    original_ann_map = {}
    for ann in original_data['annotations']:
        original_ann_map.setdefault(ann['image_id'], []).append(ann)

    original_img_ids = set(img['id'] for img in original_data['images'])
    max_annotation_id = max((ann['id'] for ann in original_data['annotations']), default=0)

    unknown_ann_count = 0
    for pseudo_ann in tqdm.tqdm(pseudo_data['annotations']):
        img_id = pseudo_ann['image_id']
        if img_id not in original_img_ids:
            continue

        pseudo_bbox = pseudo_ann['bbox']
        pseudo_ann_id = pseudo_ann['id']
        pseudo_ann_cat_name = pseudo_id_to_name.get(pseudo_ann['category_id'], "unknown")

        # 如果和已知类别重复去除
        if pseudo_ann_cat_name in category_name_set:
            continue

        # score filter
        score = pseudo_ann['score']
        if score < score_threshold:
            continue

        gt_bboxes = [ann['bbox'] for ann in original_ann_map.get(img_id, [])]
        max_iou = max((compute_iou(pseudo_bbox, gt_bbox) for gt_bbox in gt_bboxes), default=0)

        if max_iou < iou_threshold:
            max_annotation_id += 1
            new_ann = pseudo_ann.copy()
            new_ann['id'] = max_annotation_id
            new_ann['category_id'] = unknown_cat_id
            x, y, w, h = new_ann['bbox']
            segmentation = [[x, y, x + w, y, x + w, y + h, x, y + h]]
            new_ann['segmentation'] = segmentation
            new_ann['iscrowd'] = 0
            new_ann['ori_id'] = pseudo_ann_id
            new_ann['score'] = score
            new_ann['ori_cat_name'] = pseudo_ann_cat_name
            original_data['annotations'].append(new_ann)
            unknown_ann_count += 1

    # 保存新的标注文件
    with open(output_coco_path, 'w') as f:
        json.dump(original_data, f)
    print("Total unknown annotations: {}".format(unknown_ann_count))
    print(f"Merged annotations saved to {output_coco_path}")


def text_embedding(texts):
    model = './pretrain/clip-vit-base-patch32'
    tokenizer = AutoTokenizer.from_pretrained(model)
    model = CLIPTextModelWithProjection.from_pretrained(model)
    device = 'cuda:0'
    model.to(device)
    texts = tokenizer(text=texts, return_tensors='pt', padding=True)
    texts = texts.to(device)
    text_outputs = model(**texts)
    txt_feats = text_outputs.text_embeds
    txt_feats = txt_feats / txt_feats.norm(p=2, dim=-1, keepdim=True)
    return txt_feats.reshape(-1, txt_feats.shape[-1]).to('cpu')


# ===== 2. 聚类操作 =====
def cluster_ori_categories(annotations, num_clusters=10):
    # 统计每个ori_cat_name的数量
    ori_cat_counter = Counter()
    for ann in annotations:
        if 'ori_cat_name' in ann:
            ori_cat_counter[ann['ori_cat_name']] += 1

    print('ori_cat_counter', ori_cat_counter)

    unique_names = list(ori_cat_counter.keys())
    weights = np.array([np.log1p(ori_cat_counter[name]) for name in unique_names])

    # 提取CLIP文本特征
    features = text_embedding(unique_names)
    # 使用sample_weight进行加权KMeans聚类
    kmeans = KMeans(n_clusters=num_clusters, random_state=42, n_init='auto')
    kmeans.fit(features.detach().numpy(), sample_weight=weights)
    # kmeans.fit(features.detach().numpy())

    cluster_centers = kmeans.cluster_centers_

    # 生成name -> cluster 映射
    name_to_cluster = {
        name: int(cluster_id) for name, cluster_id in zip(unique_names, kmeans.labels_)
    }

    cluster_groups = defaultdict(list)
    for cls, label in zip(unique_names, kmeans.labels_):
        cluster_groups[label].append(cls)

    print("\n📊 每个聚类簇包含的类别如下：")
    for cluster_id in sorted(cluster_groups.keys()):
        print(f"\nCluster {cluster_id} ({len(cluster_groups[cluster_id])} 类):")
        for cls in cluster_groups[cluster_id]:
            print(f"  - {cls}")

    return name_to_cluster, cluster_centers


def rewrite_coco_with_clusters(coco, name_to_cluster, cluster_centers, output_path):
    annotations = coco['annotations']

    # 获取当前最大类别 ID，用于新 unknown 类别
    max_cat_id = max([cat['id'] for cat in coco['categories']], default=0)

    # 构建 cluster_label -> new_cat_id 映射
    cluster_id_to_cid = {}
    new_categories = []

    for i, center in enumerate(cluster_centers):
        cid = max_cat_id + 1 + i
        cluster_id_to_cid[i] = cid
        new_categories.append({'id': cid, 'name': f'unknown_{i}'})

    # 更新 annotation 中的 category_id
    for ann in annotations:
        if 'ori_cat_name' in ann:
            cluster_id = name_to_cluster[ann['ori_cat_name']]
            ann['category_id'] = cluster_id_to_cid[cluster_id]

    # 添加新类别
    coco['categories'].extend(new_categories)

    # remove unknown
    cat_without_unknown = []
    for cat in coco['categories']:
        if cat['name'] != 'unknown':
            cat_without_unknown.append(cat)
    coco['categories'] = cat_without_unknown

    with open(output_path, 'w') as f:
        json.dump(coco, f)

    print(f"[✓] 已保存修改后的 COCO 文件到 {output_path}")

    # 保存聚类中心
    cluster_tensor = {f'unknown_{i}': torch.tensor(center) for i, center in enumerate(cluster_centers)}
    cluster_centers_save_pth = output_path.replace('.json', '_cluster_centers.pt')
    torch.save(cluster_tensor, cluster_centers_save_pth)
    print(f"[✓] 聚类中心已保存为 {cluster_centers_save_pth}")

    coco_meta_info = {'cls': [], 'text': []}
    for cat in coco['categories']:
        coco_meta_info['cls'].append(cat['name'])
        coco_meta_info['text'].append([cat['name']])
    coco_json_relabel_meta_info_path = output_path.replace('.json', '_cls_txt.json')
    with open(coco_json_relabel_meta_info_path, 'w') as f:
        json.dump(coco_meta_info['text'], f)
    coco_json_relabel_meta_info_path = output_path.replace('.json', '_cls.json')
    with open(coco_json_relabel_meta_info_path, 'w') as f:
        json.dump(coco_meta_info['cls'], f)


def parse_args():
    parser = argparse.ArgumentParser(
        description='Pseudo label generation and clustering for COCO.'
    )

    parser.add_argument('--setting', default='COCO', type=str,
                        help='Dataset setting (e.g., COCO or LOCO COCO)')
    parser.add_argument('--task', default='40+40', type=str,
                        help='Incremental task split')
    parser.add_argument('--stage', default=0, type=int,
                        help='Training stage index')

    parser.add_argument('--img_path', default='./data/coco/train2017', type=str)

    parser.add_argument('--skip_scores', default=0.5, type=float)
    parser.add_argument('--score_threshold', default=0.5, type=float)
    parser.add_argument('--num_clusters', default=50, type=int)

    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()

    ann_folder = './data/coco/annotations'
    if args.setting == 'LOCO COCO':
        ann_folder = './data/coco/loco_annotations'

    ann_file = f'{ann_folder}/{args.task}(order)/instances_train2017_part{args.stage}.json'

    ann_save_file = ann_file.replace('.json', '_unknown_pseudo.json')
    ann_merge_save_file = ann_file.replace('.json', '_merged_pseudo.json')
    cls_out_save_file = ann_file.replace('.json', '_cls_out.json')

    # -------------------------
    # 1. pseudo label
    # -------------------------
    pseudo_unknown_label(
        ann_file,
        ann_save_file,
        args.img_path,
        args.skip_scores
    )

    # -------------------------
    # 2. merge with GT
    # -------------------------
    merge_coco_annotations(
        ann_file,
        ann_save_file,
        ann_merge_save_file,
        iou_threshold=args.skip_scores,
        score_threshold=args.score_threshold
    )

    # -------------------------
    # 3. clustering
    # -------------------------
    with open(ann_merge_save_file, 'r') as f:
        coco = json.load(f)

    name_to_cluster, cluster_centers = cluster_ori_categories(
        annotations=coco['annotations'],
        num_clusters=args.num_clusters
    )

    rewrite_coco_with_clusters(
        coco,
        name_to_cluster,
        cluster_centers,
        cls_out_save_file
    )
