import json
from collections import Counter, defaultdict

import numpy as np
import torch
from sklearn.cluster import KMeans
from transformers import (AutoTokenizer, CLIPTextModelWithProjection)


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

    coco_meta_info = {'cls':[], 'text' : []}
    for cat in coco['categories']:
        coco_meta_info['cls'].append(cat['name'])
        coco_meta_info['text'].append([cat['name']])
    coco_json_relabel_meta_info_path = output_path.replace('.json', '_cls_txt.json')
    with open(coco_json_relabel_meta_info_path, 'w') as f:
        json.dump(coco_meta_info['text'], f)
    coco_json_relabel_meta_info_path = output_path.replace('.json', '_cls.json')
    with open(coco_json_relabel_meta_info_path, 'w') as f:
        json.dump(coco_meta_info['cls'], f)


# ===== 主程序入口 =====
if __name__ == "__main__":
    coco_ann_file = 'data/coco/annotations/40+40(order)/instances_train2017_part0_un.json'
    output_path = 'data/coco/annotations/40+40(order)/instances_train2017_part0_un_cls.json'

    # 先聚类
    with open(coco_ann_file, 'r') as f:
        coco = json.load(f)

    name_to_cluster, cluster_centers = cluster_ori_categories(annotations=coco['annotations'], num_clusters=60)

    # 然后重写COCO
    rewrite_coco_with_clusters(coco, name_to_cluster, cluster_centers, output_path)
