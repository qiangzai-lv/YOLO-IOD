import json
import os
import random
from collections import defaultdict
from copy import deepcopy

from tqdm import tqdm


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

        output_path = os.path.join(output_dir, f'group_{group_id}_val.json')
        with open(output_path, 'w') as f:
            json.dump(output_data, f)

        print(f"[Group {group_id}] 写入 {len(sub_images)} 张图片，{len(sub_annotations)} 条标注 → {output_path}")
        all_cat_names = [c['name'] for c in all_categories]
        print(f"[Group {group_id}] {all_cat_names}")


if __name__ == '__main__':
    # ==== 配置 ====
    input_json_path = 'data/coco/annotations/instances_val2017.json'
    # split_val(input_json_path, 'data/coco/new_spilt/20_20_20_20', g_20_20)
    # split_val(input_json_path, 'data/coco/new_spilt/40_40', g_40_40)
    # split_val(input_json_path, 'data/coco/new_spilt/40_20_20', g_40_20)
    # split_val(input_json_path, 'data/coco/new_spilt/40_10', g_40_10)
    split_val(input_json_path, 'data/coco/new_spilt/70_10', g_70_10)