import json
import os

from tqdm import tqdm

split_point = 70

def split_cat(json_file_path, pattern):
    with open(json_file_path, 'r') as f:
        coco_data = json.load(f)

    # 获取所有类别
    categories = coco_data['categories']

    # 打乱类别列表
    # random.shuffle(categories)

    # 分割类别列表
    assert len(categories) == 80
    categories_parts = []
    categories_nums = pattern.split('+')
    base = 0
    for num in categories_nums:
        num = int(num)
        categories_parts.append(categories[base: base + num])
        base += num
    assert base == 80
    return categories_parts

def replace_order(origin_path, new_path, output_dir):
    with open(origin_path, 'r') as f:
        origin_data = json.load(f)
    with open(new_path, 'r') as f:
        new_data = json.load(f)

    origin_data['categories'] = new_data['categories']
    output_file_part = os.path.join(output_dir, 'instances_val2017_replace_order.json')
    with open(output_file_part, 'w') as f:
        json.dump(origin_data, f)

def split_coco_categories(json_file_path, output_dir, categories_parts, train_val):
    # 加载COCO数据集
    with open(json_file_path, 'r') as f:
        coco_data = json.load(f)
    # 分割类别列表

    # print("categories_part1:", categories_part1)
    # print("categories_part2:", categories_part2)

    # 创建两个新的COCO数据集
    coco_data_parts = []
    image_id_to_annotations = []
    categories_parts_accumulate = []
    print(len(categories_parts))
    for categories_part in categories_parts:
        categories_parts_accumulate.extend(categories_part)
        coco_data_parts.append(
            {
                'info': coco_data['info'],
                'licenses': coco_data['licenses'],
                'images': [],
                'annotations': [],
                'categories': [cat for cat in categories_parts_accumulate]
            })
        image_id_to_annotations.append({})
        print(coco_data_parts[0]['categories'])
        print("=====================")

    for annotation in tqdm(coco_data['annotations']):
        image_id = annotation['image_id']
        category_id = annotation['category_id']

        for index in range(len(categories_parts)):
            if train_val == 'val':
                if category_id in [cat['id'] for cat in coco_data_parts[index]['categories']]:
                    if image_id not in image_id_to_annotations[index]:
                        image_id_to_annotations[index][image_id] = []
                    image_id_to_annotations[index][image_id].append(annotation)
            else:
                if category_id in [cat['id'] for cat in categories_parts[index]]:
                    if image_id not in image_id_to_annotations[index]:
                        image_id_to_annotations[index][image_id] = []
                    image_id_to_annotations[index][image_id].append(annotation)

    # 重建图像和注释列表
    if train_val == 'val':
        for image in tqdm(coco_data['images']):
            image_id = image['id']
            for image_id_to_annotation, coco_data_part in zip(image_id_to_annotations, coco_data_parts):
                if image_id in image_id_to_annotation:
                    annotations_for_image = image_id_to_annotation[image_id]
                    if any(cat['id'] == annotation['category_id'] for annotation in annotations_for_image for cat in
                           coco_data_part['categories']):
                        coco_data_part['images'].append(image)
                        coco_data_part['annotations'].extend(annotations_for_image)
    else:
        for image in tqdm(coco_data['images']):
            image_id = image['id']
            for image_id_to_annotation, coco_data_part, categories_part in zip(image_id_to_annotations, coco_data_parts,
                                                                               categories_parts):
                if image_id in image_id_to_annotation:
                    annotations_for_image = image_id_to_annotation[image_id]
                    if any(cat['id'] == annotation['category_id'] for annotation in annotations_for_image for cat in
                           categories_part):
                        coco_data_part['images'].append(image)
                        coco_data_part['annotations'].extend(annotations_for_image)

    # 保存新的COCO数据集
    print('all image num: {}'.format(len(coco_data['images'])))
    for index, coco_data_part in enumerate(coco_data_parts):
        print('part{} image num: {}'.format(index, len(coco_data_part['images'])))
        output_file_part = os.path.join(output_dir, 'instances_{}2017_part{}.json'.format(train_val, index))
        with open(output_file_part, 'w') as f:
            json.dump(coco_data_part, f)
        cats = [cat['name'] for cat in coco_data_parts[index]['categories']]
        output_file_part = os.path.join(output_dir, 'cat_part{}.txt'.format(index))
        with open(output_file_part, 'w') as file:
            file.write(', '.join('\'{}\''.format(str(item)) for item in cats))


    print("Split completed. New JSON files saved.")

def merge_cate(stage1, stage2, train_val):
    output_dir = '/home/Newdisk/luowenlong/Datasets/COCO/2017/increase/70+10(shuffle_v1)/new/'
    with open(stage1, 'r') as f:
        coco_data = json.load(f)

    # 获取所有类别
    categories1 = coco_data['categories']

    with open(stage2, 'r') as f:
        coco_data = json.load(f)
    categories2 = coco_data['categories']
    categories1.extend(categories2)
    coco_data['categories'] = categories1

    output_file_part = os.path.join(output_dir, 'instances_{}2017_part{}.json'.format(train_val, 1))
    with open(output_file_part, 'w') as f:
        json.dump(coco_data, f)
    cats = [cat['name'] for cat in categories1]
    output_file_part = os.path.join(output_dir, 'cat_part{}.txt'.format(1))
    with open(output_file_part, 'w') as file:
        file.write(', '.join('\'{}\''.format(str(item)) for item in cats))

if __name__ == '__main__':
    input_json_path_train = r'data/coco/annotations/instances_train2017.json'  # 输入COCO JSON文件路径
    input_json_path_val = r'data/coco/annotations/instances_val2017.json'  # 输入COCO JSON文件路径
    pattern = '40+40'
    categories_parts = split_cat(input_json_path_train, pattern)
    output_dir = r'data/coco/annotations/{}(order)'.format(pattern)  # 输出目录路径

    # 确保输出目录存在
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    split_coco_categories(input_json_path_train, output_dir, categories_parts, 'train')
    split_coco_categories(input_json_path_val, output_dir, categories_parts, 'val')

