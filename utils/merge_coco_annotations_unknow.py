import json

import tqdm


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


def merge_coco_annotations(original_coco_path, pseudo_coco_path, output_coco_path, iou_threshold=0.5, score_threshold=0.5):
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


# 使用示例
merge_coco_annotations(
    'data/coco/annotations/40+40(order)/instances_train2017_part0.json',
    'data/coco/annotations/instances_train2017_un.json',
    'data/coco/annotations/40+40(order)/instances_train2017_part0_un.json', iou_threshold=0.5, score_threshold = 0.5)
