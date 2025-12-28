import json
import os
import warnings

import torch
from pycocotools.coco import COCO
from mmdet.apis import init_detector
from inference import inference_detector
from tqdm import tqdm

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


def main(ann_file, ann_save_file, img_path, skip_scores):
    mmdet_cfg = "configs/baseline/yolo_world_v2_x.py"
    mmdet_pt = './pretrain/x_stage1-62b674ad.pth'

    model = MmdetModel(mmdet_cfg, mmdet_pt, skip_scores=skip_scores, class_json = './data/texts/coco_class_unknown.json')
    # 加载 COCO 数据集
    coco = COCO(ann_file)
    ann_save = json.load(open(ann_file))

    # 转换为更结构化的JSON格式
    categories = [{"id": idx, "name": name} for idx, name in enumerate(classes, start=1)]

    # 加载类别信息
    categories_map = {}
    for cat in categories:
        categories_map[cat['name']] = cat['id']

    # images embeddings

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

if __name__ == '__main__':
    ann_file = './data/coco/annotations/instances_train2017.json'  # 标注文件路径
    img_path = './data/coco/train2017'
    ann_save_file = './data/coco/annotations/instances_train2017_un.json'
    main(ann_file, ann_save_file, img_path, 0.5)
