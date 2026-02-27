import argparse
import json
import os
import random
from collections import defaultdict
from copy import deepcopy

from tqdm import tqdm

random.seed(42)

g_10_10 = [
    ['person', 'bench', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'skateboard', 'surfboard',
     'cell phone'],
    ['bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'traffic light', 'fire hydrant', 'stop sign'],
    ['chair', 'couch', 'potted plant', 'tv', 'laptop', 'remote', 'keyboard', 'book', 'vase', 'mouse'],
    ['zebra', 'giraffe', 'sheep', 'cow', 'elephant', 'bird', 'horse', 'kite', 'boat', 'bear'],
    ['sports ball', 'baseball bat', 'baseball glove', 'tennis racket', 'snowboard', 'frisbee', 'dog', 'skis', 'donut',
     'parking meter'],
    ['wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'pizza', 'dining table', 'bottle', 'cake'],
    ['banana', 'apple', 'orange', 'broccoli', 'carrot', 'cat', 'teddy bear', 'bed', 'sandwich', 'hot dog'],
    ['toilet', 'sink', 'hair drier', 'toothbrush', 'toaster', 'scissors', 'microwave', 'oven', 'refrigerator', 'clock']
]

g_20_20 = [
    ['person', 'bicycle', 'car', 'motorcycle', 'bus', 'train', 'truck', 'boat', 'traffic light', 'bench', 'backpack',
     'umbrella', 'handbag', 'tie', 'suitcase', 'skis', 'skateboard', 'surfboard', 'cell phone', 'kite']
    , ['cat', 'couch', 'potted plant', 'bed', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'book', 'clock', 'vase',
       'scissors', 'teddy bear', 'banana', 'dog', 'frisbee', 'toilet', 'toothbrush', 'donut']
    , ['sports ball', 'baseball bat', 'baseball glove', 'tennis racket', 'giraffe', 'sheep', 'hot dog', 'parking meter',
       'cow', 'elephant', 'bird', 'snowboard', 'airplane', 'stop sign', 'fire hydrant', 'zebra', 'bear', 'hair drier',
       'horse', 'toaster']
    , ['bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'apple', 'sandwich', 'orange', 'broccoli',
       'carrot', 'pizza', 'cake', 'dining table', 'microwave', 'oven', 'sink', 'refrigerator', 'chair']]

g_40_40 = [['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
            'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'dog', 'horse', 'sheep', 'cow', 'elephant',
            'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis',
            'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard',
            'tennis racket', 'hair drier', 'toothbrush'],
           ['cat', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich',
            'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 'potted plant',
            'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave',
            'oven', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'toaster']]

g_40_10 = [['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
            'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'dog', 'horse', 'sheep', 'cow', 'elephant',
            'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis',
            'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard',
            'tennis racket', 'hair drier', 'toothbrush']
    , ['chair', 'couch', 'potted plant', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'book', 'cell phone']
    , ['bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'pizza', 'cake', 'dining table']
    , ['banana', 'apple', 'orange', 'broccoli', 'cat', 'carrot', 'bed', 'sandwich', 'teddy bear', 'hot dog']
    , ['toilet', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'scissors', 'donut', 'clock', 'vase']]

g_40_20 = [
    ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
     'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear',
     'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard',
     'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'hair drier',
     'toothbrush'],
    ['cat', 'couch', 'potted plant', 'bed', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'book',
     'clock', 'vase', 'scissors', 'teddy bear', 'toaster', 'hot dog', 'donut', 'toilet', 'banana']
    , ['bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'apple', 'sandwich', 'orange', 'broccoli',
       'carrot', 'pizza', 'cake', 'dining table', 'microwave', 'oven', 'sink', 'refrigerator', 'chair']
]

g_70_10 = [['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
            'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'dog', 'horse', 'sheep', 'cow', 'elephant',
            'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis',
            'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard',
            'tennis racket', 'hair drier', 'toothbrush', 'chair', 'couch', 'potted plant', 'tv', 'laptop', 'mouse',
            'remote', 'keyboard', 'book', 'cell phone', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
            'pizza', 'cake', 'dining table', 'banana', 'apple', 'orange', 'broccoli', 'cat', 'carrot', 'bed',
            'sandwich', 'teddy bear', 'hot dog']
    , ['toilet', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'scissors', 'donut', 'clock', 'vase']]


def remap_category_ids(coco, start_index=1):
    # 1. 按照 categories 出现顺序创建旧id -> 新id 映射
    old_to_new_id = {}
    new_categories = []
    for idx, cat in enumerate(coco['categories']):
        new_id = idx + start_index
        old_to_new_id[cat['id']] = new_id
        new_cat = deepcopy(cat)
        new_cat['id'] = new_id
        new_categories.append(new_cat)

    # 2. 替换 annotations 中的 category_id
    new_annotations = []
    for ann in coco['annotations']:
        new_ann = deepcopy(ann)
        old_id = ann['category_id']
        if old_id not in old_to_new_id:
            raise ValueError(f"Old category_id {old_id} not found in categories.")
        new_ann['category_id'] = old_to_new_id[old_id]
        new_annotations.append(new_ann)

    # 3. 构造新的 coco 字典
    coco['categories'] = new_categories
    coco['annotations'] = new_annotations

    # 4. 保存新文件
    return coco


def coco_2017_train_split(input_json_path, output_dir, category_name_groups):
    # ==== 加载 COCO 标注 ====
    with open(input_json_path, 'r') as f:
        coco = json.load(f)

    images = coco['images']
    annotations = coco['annotations']
    categories = coco['categories']

    total_images_len = len(images)
    total_annotations_len = len(annotations)

    print(f"total_images:{total_images_len}")
    print(f"total_annotations:{total_annotations_len}")

    # ==== 构建类别名 ↔ ID 映射 ====
    cat_name_to_id = {cat['name']: cat['id'] for cat in categories}

    # ==== 检查类别名是否存在 ====
    for group in category_name_groups:
        for name in group:
            if name not in cat_name_to_id:
                raise ValueError(f"类别名 '{name}' 不存在于 COCO 标注中，请检查拼写。")

    # ==== 构建类别 ID 到分组编号映射 ====
    cat_id_to_group = {}
    for group_idx, group in enumerate(category_name_groups):
        print(f'Group {group_idx} ', len(group))
        for name in group:
            cat_id = cat_name_to_id[name]
            cat_id_to_group[cat_id] = group_idx

    group_id_to_cat_ids = {}
    for group_idx, group in enumerate(category_name_groups):
        group_id_to_cat_ids[group_idx] = [cat_name_to_id[name] for name in group]

    # ==== 统计每张图片涉及的分组 ====
    image_id_to_groups = defaultdict(set)
    image_id_to_cat_ids = defaultdict(set)
    image_id_to_annotations = defaultdict(list)

    for ann in annotations:
        img_id = ann['image_id']
        cat_id = ann['category_id']
        image_id_to_cat_ids[img_id].add(cat_id)
        if cat_id in cat_id_to_group:
            group_id = cat_id_to_group[cat_id]
            image_id_to_groups[img_id].add(group_id)
            image_id_to_annotations[img_id].append(ann)

    # 统计跨组图片
    coco_single_group_st = defaultdict(int)
    coco_cross_group_st = defaultdict(int)
    for img in tqdm(images):
        img_id = img['id']
        groups = list(image_id_to_groups.get(img_id, []))
        if not groups:
            continue  # 不属于任何组，跳过
        if len(groups) > 1:
            coco_cross_group_st[len(groups)] += 1
        else:
            coco_single_group_st[groups[0]] += 1

    print('单组统计')
    for g_id, st in coco_single_group_st.items():
        print(f'group_{g_id}: {st}')

    print('跨组统计')
    for g_id, st in coco_cross_group_st.items():
        print(f'group_len_{g_id}: {st}')

    # ==== 随机将图片分配到某一个组 ====
    group_to_image_ids = defaultdict(list)

    print("开始图片划分")
    for img in tqdm(images):
        img_id = img['id']
        groups = list(image_id_to_groups.get(img_id, []))
        if not groups:
            continue
        assigned_group = random.choice(groups)
        group_to_image_ids[assigned_group].append(img_id)

    # ==== 保存每个组的子集 ====
    print('保存每个组的子集')
    os.makedirs(output_dir, exist_ok=True)
    all_categories = []
    for group_id in sorted(group_to_image_ids.keys()):
        img_ids = group_to_image_ids[group_id]
        print(f"Group {group_id} {len(img_ids)} 开始划分")
        sub_images = [img for img in images if img['id'] in img_ids]
        sub_annotations = []
        cat_ids_in_group = group_id_to_cat_ids[group_id]

        for img_id in tqdm(img_ids):
            anns = image_id_to_annotations[img_id]
            anns_s = [ann for ann in anns if ann['category_id'] in cat_ids_in_group]
            sub_annotations.extend(anns_s)

        sub_categories = [cat for cat in categories if cat['id'] in cat_ids_in_group]
        all_categories.extend(sub_categories)

        output_data = {
            'images': sub_images,
            'annotations': sub_annotations,
            'categories': all_categories
        }

        total_images_len -= len(sub_images)
        total_annotations_len -= len(sub_annotations)

        output_path = os.path.join(output_dir, f'instances_train2017_part{group_id}.json')
        print('对ann进行re_id')
        output_data = remap_category_ids(output_data)
        print(output_data['categories'])
        with open(output_path, 'w') as f:
            json.dump(output_data, f)

        print(f"[Group {group_id}] 写入 {len(sub_images)} 张图片，{len(sub_annotations)} 条标注 → {output_path}")
        all_cat_names = [c['name'] for c in all_categories]
        sub_cat_names = [c['name'] for c in sub_categories]
        with open(os.path.join(output_dir, f'loco_class_cats_stage{group_id}.json'), 'w') as f:
            json.dump(all_cat_names, f)
        with open(os.path.join(output_dir, f'loco_class_texts_stage{group_id}.json'), 'w') as f:
            json.dump([[ca] for ca in all_cat_names], f)
        with open(os.path.join(output_dir, f'loco_class_texts_curs{group_id}.json'), 'w') as f:
            json.dump([[ca] for ca in sub_cat_names], f)
        print(f"[Group {group_id}] {all_cat_names}")

    print(f"剩余图片:{total_images_len}")
    print(f"剩余标注:{total_annotations_len}")


def split_val(input_json_path, output_dir, category_name_groups):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # ==== 加载 COCO 标注 ====
    with open(input_json_path, 'r') as f:
        coco = json.load(f)

    images = coco['images']
    annotations = coco['annotations']
    categories = coco['categories']

    print(f"总标注数目：{len(annotations)}")
    print(f"总图片{len(images)}")

    # ==== 构建类别名 ↔ ID 映射 ====
    cat_name_to_id = {cat['name']: cat['id'] for cat in categories}

    # ==== 检查类别名是否存在 ====
    for group in category_name_groups:
        for name in group:
            if name not in cat_name_to_id:
                raise ValueError(f"类别名 '{name}' 不存在于 COCO 标注中，请检查拼写。")

    # ==== 构建类别 ID 到分组编号映射 ====
    cat_id_to_group = {}
    for group_idx, group in enumerate(category_name_groups):
        print(f'Group {group_idx} ', len(group))
        for name in group:
            cat_id = cat_name_to_id[name]
            cat_id_to_group[cat_id] = group_idx

    group_id_to_cat_ids = {}
    for group_idx, group in enumerate(category_name_groups):
        group_id_to_cat_ids[group_idx] = [cat_name_to_id[name] for name in group]

    # ==== 统计每张图片涉及的分组 ====
    image_id_to_groups = defaultdict(set)
    image_id_to_annotations = defaultdict(list)

    for ann in annotations:
        img_id = ann['image_id']
        cat_id = ann['category_id']
        if cat_id in cat_id_to_group:
            group_id = cat_id_to_group[cat_id]
            image_id_to_groups[img_id].add(group_id)
            image_id_to_annotations[img_id].append(ann)

    # ==== 随机将图片分配到某一个组 ====
    group_to_image_ids = defaultdict(list)

    empty_images_ids = []

    print("开始图片划分")
    for img in tqdm(images):
        img_id = img['id']
        groups = list(image_id_to_groups.get(img_id, []))
        if not groups:
            empty_images_ids.append(img_id)
        for group in groups:
            group_to_image_ids[group].append(img_id)

    # ==== 保存每个组的子集 ====
    os.makedirs(output_dir, exist_ok=True)

    all_categories = []
    all_categories_id = []
    all_images_id = empty_images_ids

    for group_id in sorted(group_to_image_ids.keys()):
        print(f"Group {group_id} 开始划分")
        img_ids = group_to_image_ids[group_id]
        all_images_id.extend(img_ids)
        all_images_id = list(set(all_images_id))
        sub_images = [img for img in images if img['id'] in all_images_id]
        sub_annotations = []
        cat_ids_in_group = group_id_to_cat_ids[group_id]
        all_categories_id.extend(cat_ids_in_group)
        print(len(all_categories_id))

        for img_id in tqdm(all_images_id):
            anns = image_id_to_annotations[img_id]
            anns_s = [ann for ann in anns if ann['category_id'] in all_categories_id]
            sub_annotations.extend(anns_s)

        sub_categories = [cat for cat in categories if cat['id'] in cat_ids_in_group]
        all_categories.extend(sub_categories)

        output_data = {
            'images': sub_images,
            'annotations': sub_annotations,
            'categories': all_categories
        }

        output_data = remap_category_ids(output_data)
        print(output_data['categories'])

        output_path = os.path.join(output_dir, f'instances_val2017_part{group_id}.json')
        with open(output_path, 'w') as f:
            json.dump(output_data, f)

        print(f"[Group {group_id}] 写入 {len(sub_images)} 张图片，{len(sub_annotations)} 条标注 → {output_path}")
        all_cat_names = [c['name'] for c in all_categories]
        print(f"[Group {group_id}] {all_cat_names}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Split COCO dataset for incremental learning"
    )

    parser.add_argument(
        "--train_json",
        default="data/coco/annotations/instances_train2017.json",
        type=str,
        help="Path to COCO train annotation"
    )

    parser.add_argument(
        "--val_json",
        default="data/coco/annotations/instances_val2017.json",
        type=str,
        help="Path to COCO val annotation"
    )

    parser.add_argument(
        "--output_dir",
        default="data/coco/loco_annotations",
        type=str,
        help="Output directory for split annotations"
    )

    parser.add_argument(
        "--pattern",
        required=True,
        type=str,
        help="Split pattern (e.g., 40+40, 20+20+20+20, 40+20+20)"
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    input_train_json_path = args.train_json
    input_val_json_path = args.val_json
    output_dir = os.path.join(args.output_dir, args.pattern + '(order)')

    pattern = args.pattern

    # choose category split
    if pattern == "40+40":
        category_name_groups = g_40_40
    elif pattern == "70+10":
        category_name_groups = g_70_10
    elif pattern == "40+10+10+10+10":
        category_name_groups = g_40_10
    elif pattern == "40+20+20":
        category_name_groups = g_40_20
    elif pattern == "20+20+20+20":
        category_name_groups = g_20_20
    elif pattern == "10+10+10+10+10+10+10+10":
        category_name_groups = g_10_10
    else:
        raise ValueError(f"Unsupported pattern: {pattern}")

    os.makedirs(output_dir, exist_ok=True)

    # train split
    coco_2017_train_split(
        input_train_json_path,
        output_dir,
        category_name_groups
    )

    # val split
    split_val(
        input_val_json_path,
        output_dir,
        category_name_groups
    )
