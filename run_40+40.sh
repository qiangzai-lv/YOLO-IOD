## unknown cluster
#python utils/merge_coco_annotations_unknow.py
#python utils/cls_cluste.py
#
## base
#bash tools/dist_train_gps.sh configs/40_40/yolo_world_v2_x_finetune_coco_40_40_t0_un_sep.py 4 --amp
#
## incr
#bash tools/dist_train_gps.sh configs/40_40/yolo_world_v2_x_finetune_coco_40_40_t1_gps.py 4 --amp
#
#python utils/pseudo_label_sc.py \
# --ann_file=data/coco/annotations/40+40\(order\)/instances_train2017_part1.json \
# --mmdet_cfg=work_dirs/yolo_world_v2_x_finetune_coco_40_40_t0_un_sep/yolo_world_v2_x_finetune_coco_40_40_t0_un_sep.py \
# --mmdet_pt=work_dirs/yolo_world_v2_x_finetune_coco_40_40_t0_un_sep/epoch_20.pth \
# --class_json=data/texts/coco_class_texts_40.json

bash tools/dist_train_gps.sh configs/40_40/yolo_world_v2_x_finetune_coco_40_40_t1_iod.py 4 --amp
