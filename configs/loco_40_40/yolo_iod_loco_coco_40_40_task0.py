_base_ = (
    '../../third_party/mmyolo/configs/yolov8/'
    'yolov8_x_mask-refine_syncbn_fast_8xb16-500e_coco.py')
custom_imports = dict(
    imports=['yolo_world'],
    allow_failed_imports=False)

# hyper-parameters
num_classes = 70
num_training_classes = 70
num_known_cls = 40
max_epochs = 20  # Maximum training epochs
close_mosaic_epochs = 10
save_epoch_intervals = 1
text_channels = 512
neck_embed_channels = [128, 256, _base_.last_stage_out_channels // 2]
neck_num_heads = [4, 8, _base_.last_stage_out_channels // 2 // 32]
base_lr = 2e-4
weight_decay = 0.05
train_batch_size_per_gpu = 16
load_from = './weights/x_stage1-62b674ad.pth'
text_model_name = 'openai/clip-vit-base-patch32'
persistent_workers = False

classes = ["person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket", "hair drier", "toothbrush", "unknown_0", "unknown_1", "unknown_2", "unknown_3", "unknown_4", "unknown_5", "unknown_6", "unknown_7", "unknown_8", "unknown_9", "unknown_10", "unknown_11", "unknown_12", "unknown_13", "unknown_14", "unknown_15", "unknown_16", "unknown_17", "unknown_18", "unknown_19", "unknown_20", "unknown_21", "unknown_22", "unknown_23", "unknown_24", "unknown_25", "unknown_26", "unknown_27", "unknown_28", "unknown_29"]
classes_val = ["person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket", "hair drier", "toothbrush"]

# model settings
model = dict(
    type='YOLOWorldDetector',
    mm_neck=True,
    num_train_classes=num_training_classes,
    num_test_classes=num_classes,
    data_preprocessor=dict(type='YOLOWDetDataPreprocessor'),
    backbone=dict(
        _delete_=True,
        type='MultiModalYOLOBackbone',
        image_model={{_base_.model.backbone}},
        text_model=dict(
            type='HuggingCacheCLIPCLSPSLanguageBackbone',
            cls_unknown_weight_path='data/coco/loco_annotations/40+40(order)/instances_train2017_part0_cls_out_cluster_centers.pt',
            model_name=text_model_name,
            frozen_modules=['all'])),
    neck=dict(type='YOLOWorldPAFPN',
              guide_channels=text_channels,
              embed_channels=neck_embed_channels,
              num_heads=neck_num_heads,
              block_cfg=dict(type='MaxSigmoidCSPLayerWithTwoConv')),
    bbox_head=dict(type='YOLOWorldUnHead',
                   known_cls_num=num_known_cls,
                   head_module=dict(type='YOLOWorldUnHeadModule',
                                    use_bn_head=True,
                                    known_cls_num=num_known_cls,
                                    embed_dims=text_channels,
                                    num_classes=num_training_classes)),
    train_cfg=dict(assigner=dict(num_classes=num_training_classes)))

# dataset settings
text_transform = [
    dict(type='RandomLoadTextUnknown',
         num_neg_samples=(num_classes, num_classes),
         max_num_samples=num_training_classes,
         padding_to_max=True,
         padding_value=''),
    dict(type='mmdet.PackDetInputs',
         meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape', 'flip',
                    'flip_direction', 'texts'))
]
mosaic_affine_transform = [
    dict(
        type='MultiModalMosaic',
        img_scale=_base_.img_scale,
        pad_val=114.0,
        pre_transform=_base_.pre_transform),
    dict(type='YOLOv5CopyPaste', prob=_base_.copypaste_prob),
    dict(
        type='YOLOv5RandomAffine',
        max_rotate_degree=0.0,
        max_shear_degree=0.0,
        max_aspect_ratio=100.,
        scaling_ratio_range=(1 - _base_.affine_scale,
                             1 + _base_.affine_scale),
        # img_scale is (width, height)
        border=(-_base_.img_scale[0] // 2, -_base_.img_scale[1] // 2),
        border_val=(114, 114, 114),
        min_area_ratio=_base_.min_area_ratio,
        use_mask_refine=_base_.use_mask2refine)
]
train_pipeline = [
    *_base_.pre_transform,
    *mosaic_affine_transform,
    dict(
        type='YOLOv5MultiModalMixUp',
        prob=_base_.mixup_prob,
        pre_transform=[*_base_.pre_transform,
                       *mosaic_affine_transform]),
    *_base_.last_transform[:-1],
    *text_transform
]
train_pipeline_stage2 = [
    *_base_.train_pipeline_stage2[:-1],
    *text_transform
]
coco_train_dataset = dict(
    _delete_=True,
    type='MultiModalDataset',
    dataset=dict(
        type='YOLOv5CocoDataset',
        data_root='data/coco',
        metainfo=dict(classes=classes),
        ann_file='loco_annotations/40+40(order)/instances_train2017_part0_cls_out.json',
        data_prefix=dict(img='train2017/'),
        filter_cfg=dict(filter_empty_gt=False, min_size=32)),
    class_text_path='data/coco/loco_annotations/40+40(order)/instances_train2017_part0_cls_out_cls_txt.json',
    pipeline=train_pipeline)

train_dataloader = dict(
    persistent_workers=persistent_workers,
    batch_size=train_batch_size_per_gpu,
    collate_fn=dict(type='yolow_collate'),
    dataset=coco_train_dataset)

test_pipeline = [
    *_base_.test_pipeline[:-1],
    dict(type='LoadText'),
    dict(
        type='mmdet.PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor', 'pad_param', 'texts'))
]

coco_val_dataset = dict(
    _delete_=True,
    type='MultiModalDataset',
    dataset=dict(
        type='YOLOv5CocoDataset',
        data_root='data/coco',
        ann_file='loco_annotations/40+40(order)/instances_val2017_part0.json',
        data_prefix=dict(img='val2017/'),
        metainfo=dict(classes=classes_val),
        filter_cfg=dict(filter_empty_gt=False, min_size=32)),
    class_text_path='data/coco/loco_annotations/40+40(order)/loco_class_texts_stage0.json',
    pipeline=test_pipeline)
val_dataloader = dict(dataset=coco_val_dataset)
test_dataloader = val_dataloader
# training settings
default_hooks = dict(
    param_scheduler=dict(
        scheduler_type='linear',
        lr_factor=0.01,
        max_epochs=max_epochs),
    checkpoint=dict(
        max_keep_ckpts=4,
        save_best=None,
        interval=save_epoch_intervals))
custom_hooks = [
    dict(
        type='EMAHook',
        ema_type='ExpMomentumEMA',
        momentum=0.0001,
        update_buffers=True,
        strict_load=False,
        priority=49),
    dict(
        type='mmdet.PipelineSwitchHook',
        switch_epoch=max_epochs - close_mosaic_epochs,
        switch_pipeline=train_pipeline_stage2)
]

train_cfg = dict(
    type='EpochBasedTrainGPSLoop',
    gps_ratio=0.8,
    up_ratio=0.8,
    importance_save_path='data/coco/loco_annotations/40+40(order)/t0_importance_iks.pt',
    max_epochs=max_epochs,
    val_interval=1,
    dynamic_intervals=[((max_epochs - close_mosaic_epochs),
                        _base_.val_interval_stage2)])

optim_wrapper = dict(
    optimizer=dict(
        _delete_=True,
        type='AdamW',
        lr=base_lr,
        weight_decay=weight_decay,
        batch_size_per_gpu=train_batch_size_per_gpu),
    paramwise_cfg=dict(
        custom_keys={
            'backbone': dict(lr_mult=0.1, decay_mult=1.0),
            'backbone.text_model': dict(lr_mult=0.01),
            'logit_scale': dict(weight_decay=0.0)}),
    constructor='YOLOWv5OptimizerConstructor')

# evaluation settings
val_evaluator = dict(
    _delete_=True,
    type='mmdet.CocoMetric',
    proposal_nums=(100, 1, 10),
    classwise=True,
    metric_items=['mAP', 'mAP_50', 'mAP_75', 'mAP_s', 'mAP_m', 'mAP_l', 'AR@100', 'AR@300', 'AR@1000', 'AR_s@1000',
                  'AR_m@1000', 'AR_l@1000'],
    ann_file='data/coco/loco_annotations/40+40(order)/instances_val2017_part0.json',
    metric='bbox')
test_evaluator = val_evaluator
find_unused_parameters=True