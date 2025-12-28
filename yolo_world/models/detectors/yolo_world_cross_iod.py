# Copyright (c) Tencent Inc. All rights reserved.
import os
import time
from pathlib import Path
from typing import List, Union
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmdet.models.utils import filter_scores_and_topk
from mmdet.registry import MODELS
from mmdet.structures import OptSampleList
from mmdet.structures import SampleList
from mmdet.utils import (ConfigType, OptConfigType)
from mmengine.config import Config
from mmyolo.models.detectors import YOLODetector
from mmyolo.registry import MODELS
from torch import Tensor


@MODELS.register_module()
class YOLOWorldCrossIodDetector(YOLODetector):
    """Implementation of YOLO World Series"""

    def __init__(self,
                 *args,
                 mm_neck: bool = False,
                 num_train_classes=80,
                 num_test_classes=80,
                 load_from_weight='/home/Newdisk1/luyue/lv_workdir/YOLO-Drone/pretrain/vis_5+5_weight.pth',
                 ori_setting: ConfigType = None,
                 cur_setting: ConfigType = None,
                 kd_cfg: OptConfigType = None,
                 prompt_dim=512,
                 num_prompts=80,
                 embedding_path='',
                 reparameterized=False,
                 freeze_prompt=False,
                 use_mlp_adapter=False,
                 **kwargs) -> None:
        self.mm_neck = mm_neck
        self.num_training_classes = num_train_classes
        self.num_test_classes = num_test_classes
        self.prompt_dim = prompt_dim
        self.num_prompts = num_prompts
        self.reparameterized = reparameterized
        self.freeze_prompt = freeze_prompt
        self.use_mlp_adapter = use_mlp_adapter
        super().__init__(*args, **kwargs)

        if not self.reparameterized:
            if len(embedding_path) > 0:
                import numpy as np
                self.embeddings = torch.nn.Parameter(
                    torch.from_numpy(np.load(embedding_path)).float())
            else:
                # random init
                embeddings = nn.functional.normalize(torch.randn(
                    (num_prompts, prompt_dim)),
                    dim=-1)
                self.embeddings = nn.Parameter(embeddings)

            if self.freeze_prompt:
                self.embeddings.requires_grad = False
            else:
                self.embeddings.requires_grad = True

            if use_mlp_adapter:
                self.adapter = nn.Sequential(
                    nn.Linear(prompt_dim, prompt_dim * 2), nn.ReLU(True),
                    nn.Linear(prompt_dim * 2, prompt_dim))
            else:
                self.adapter = None
        # Build old model
        assert isinstance(ori_setting.config, (str, Path)), 'ori_setting config must be str or Path'
        assert ori_setting.ckpt is not None, 'ori_setting ckpt must not be None'
        ori_config = Config.fromfile(ori_setting.config)
        self.ori_model = MODELS.build(ori_config['model'])
        # Build cur model
        assert isinstance(cur_setting.config, (str, Path)), 'cur_setting config must be str or Path'
        assert cur_setting.ckpt is not None, 'cur_setting ckpt must not be None'
        cur_config = Config.fromfile(cur_setting.config)
        self.cur_model = MODELS.build(cur_config['model'])
        # freeze
        self.freeze(self.cur_model)
        self.freeze(self.ori_model)
        # kd loss
        self.loss_cls_kd_weight_old = kd_cfg['loss_cls_kd_weight_old']
        self.loss_reg_kd_weight_old = kd_cfg['loss_reg_kd_weight_old']
        self.loss_cls_kd_weight_new = kd_cfg['loss_cls_kd_weight_new']
        self.loss_reg_kd_weight_new = kd_cfg['loss_reg_kd_weight_new']
        self.loss_kd_iou_threshold = kd_cfg.get('iou_threshold', 0.5)
        self.loss_kd_score_threshold = kd_cfg.get('score_threshold', 0.2)
        self.ori_num_classes = kd_cfg.get('ori_num_classes', 5)

        self.reused_teacher_head_idx = kd_cfg['reused_teacher_head_idx']
        # 参数重组
        self.parameter_reorganization(ori_setting, cur_setting, load_from_weight)

    def loss(self, batch_inputs: Tensor,
             batch_data_samples: SampleList) -> Union[dict, list]:
        """Calculate losses from a batch of inputs and data samples."""
        self.bbox_head.num_classes = self.num_training_classes

        batch_gt_instances = batch_data_samples['bboxes_labels']
        batch_img_metas = batch_data_samples['img_metas']

        img_feats, txt_feats = self.extract_feat(batch_inputs,
                                                 batch_data_samples)

        # target forward
        losses_target, cls_feat_hold, reg_feat_hold = self.bbox_head.loss_and_hold(img_feats, txt_feats,
                                                                                   batch_data_samples,
                                                                                   self.reused_teacher_head_idx)

        self.ori_model.bbox_head.num_classes = self.ori_model.num_training_classes
        # # 老模型蒸馏
        # # old_model forward
        img_feats_old, txt_feats_old = self.ori_model.extract_feat(batch_inputs,
                                                                   batch_data_samples)
        cls_logit_old, bbox_preds_old, _, cls_feat_hold_old, reg_feat_hold_old = self.ori_model.bbox_head.forward_and_hold(
            img_feats_old, txt_feats_old, self.reused_teacher_head_idx)

        # align
        cross_old_cls_align = []
        cross_old_reg_align = []

        for cls_feat_hold_it, cls_feat_hold_old_it in zip(cls_feat_hold, cls_feat_hold_old):
            cross_old_cls_align.append(self.align_scale(cls_feat_hold_it, cls_feat_hold_old_it))
        for reg_feat_hold_it, reg_feat_hold_old_it in zip(reg_feat_hold, reg_feat_hold_old):
            cross_old_reg_align.append(self.align_scale(reg_feat_hold_it, reg_feat_hold_old_it))

        cls_logit_old_cross, bbox_preds_old_cross, _ = self.ori_model.bbox_head.forward_with_hold(
            img_feats, cross_old_cls_align, cross_old_reg_align, txt_feats_old, self.reused_teacher_head_idx)

        flatten_cls_logit_old, flatten_bbox_preds_old = self.ori_model.bbox_head.loss_result_process(
            cls_logit_old, bbox_preds_old, batch_img_metas)
        flatten_cls_logit_cross_old, flatten_bbox_preds_cross_old = self.ori_model.bbox_head.loss_result_process(
            cls_logit_old_cross, bbox_preds_old_cross, batch_img_metas)

        # cal kd loss
        loss_cls_kd_old, loss_reg_kd_old = self.distill_loss(flatten_cls_logit_cross_old, flatten_cls_logit_old,
                                                             flatten_bbox_preds_cross_old,
                                                             flatten_bbox_preds_old,
                                                             batch_gt_instances,
                                                             conf_thresh=self.loss_kd_score_threshold,
                                                             iou_thresh=self.loss_kd_iou_threshold,
                                                             ori_class_num = self.ori_num_classes)

        # new_model forward
        img_feats_new, txt_feats_new = self.cur_model.extract_feat(batch_inputs,
                                                                   batch_data_samples)
        cls_logit_new, bbox_preds_new, _, cls_feat_hold_new, reg_feat_hold_new = self.cur_model.bbox_head.forward_and_hold(
            img_feats_new, txt_feats_new, self.reused_teacher_head_idx)

        # align
        cross_new_cls_align = []

        for cls_feat_hold_it, cls_feat_hold_new_it in zip(cls_feat_hold, cls_feat_hold_new):
            cross_new_cls_align.append(self.align_scale(cls_feat_hold_it, cls_feat_hold_new_it))

        cross_new_reg_align = []
        for reg_feat_hold_it, reg_feat_hold_new_it in zip(reg_feat_hold, reg_feat_hold_new):
            cross_new_reg_align.append(self.align_scale(reg_feat_hold_it, reg_feat_hold_new_it))

        # 新模型蒸馏
        cls_logit_new_cross, bbox_preds_new_cross, _ = self.cur_model.bbox_head.forward_with_hold(
            img_feats, cross_new_cls_align, cross_new_reg_align, txt_feats_new, self.reused_teacher_head_idx)
        flatten_cls_logit_new, flatten_bbox_preds_new = self.cur_model.bbox_head.loss_result_process(
            cls_logit_new, bbox_preds_new, batch_img_metas)
        flatten_cls_logit_cross_new, flatten_bbox_preds_cross_new = self.cur_model.bbox_head.loss_result_process(
            cls_logit_new_cross, bbox_preds_new_cross, batch_img_metas)

        # cal kd loss
        loss_cls_kd_new, loss_reg_kd_new = self.distill_loss(flatten_cls_logit_cross_new, flatten_cls_logit_new,
                                                             flatten_bbox_preds_cross_new,
                                                             flatten_bbox_preds_new, batch_gt_instances,
                                                             is_old=False,
                                                             conf_thresh=self.loss_kd_score_threshold,
                                                             iou_thresh=self.loss_kd_iou_threshold,
                                                             ori_class_num = self.ori_num_classes)

        losses_target['loss_cls_kd_new'] = loss_cls_kd_new
        losses_target['loss_reg_kd_new'] = loss_reg_kd_new
        losses_target['loss_cls_kd_old'] = loss_cls_kd_old
        losses_target['loss_reg_kd_old'] = loss_reg_kd_old

        return losses_target


    def predict(self,
                batch_inputs: Tensor,
                batch_data_samples: SampleList,
                rescale: bool = True) -> SampleList:
        """Predict results from a batch of inputs and data samples with post-
        processing.
        """

        img_feats, txt_feats = self.extract_feat(batch_inputs,
                                                 batch_data_samples)

        self.bbox_head.num_classes = self.num_test_classes
        if self.reparameterized:
            results_list = self.bbox_head.predict(img_feats,
                                                  batch_data_samples,
                                                  rescale=rescale)
        else:
            results_list = self.bbox_head.predict(img_feats,
                                                  txt_feats,
                                                  batch_data_samples,
                                                  rescale=rescale)

        batch_data_samples = self.add_pred_to_datasample(
            batch_data_samples, results_list)
        return batch_data_samples

    def _forward(
            self,
            batch_inputs: Tensor,
            batch_data_samples: OptSampleList = None) -> Tuple[List[Tensor]]:
        """Network forward process. Usually includes backbone, neck and head
        forward without any post-processing.
        """
        img_feats, txt_feats = self.extract_feat(batch_inputs,
                                                 batch_data_samples)
        if self.reparameterized:
            results = self.bbox_head.forward(img_feats)
        else:
            results = self.bbox_head.forward(img_feats, txt_feats)
        return results

    def extract_feat(
            self, batch_inputs: Tensor,
            batch_data_samples: SampleList) -> Tuple[Tuple[Tensor], Tensor]:
        """Extract features."""
        # only image features
        img_feats, _ = self.backbone(batch_inputs, None)

        if not self.reparameterized:
            # use embeddings
            txt_feats = self.embeddings[None]
            if self.adapter is not None:
                txt_feats = self.adapter(txt_feats) + txt_feats
                txt_feats = nn.functional.normalize(txt_feats, dim=-1, p=2)
            txt_feats = txt_feats.repeat(img_feats[0].shape[0], 1, 1)
        else:
            txt_feats = None
        if self.with_neck:
            if self.mm_neck:
                img_feats = self.neck(img_feats, txt_feats)
            else:
                img_feats = self.neck(img_feats)
        return img_feats, txt_feats

    def distill_loss(self, cls_stus, cls_teas, bbox_stus, bbox_teas, batch_gt_instances,
                     conf_thresh=0.4, iou_thresh=0.5, is_old=True, ori_class_num=5):

        image_ids = batch_gt_instances[:, 0].long()  # 取出 image_id
        unique_ids = torch.unique(image_ids)  # 获取所有唯一的 image_id
        grouped_instances = [batch_gt_instances[image_ids == img_id] for img_id in unique_ids]
        grouped_instances_new = []
        for grouped_instance in grouped_instances:
            mask = grouped_instance[:, 1] > ori_class_num
            grouped_instances_new.append(grouped_instance[mask][:, -4:])

        loss_cls_kd_batch = []
        loss_reg_kd_batch = []

        for cls_stu, cls_tea, bbox_stu, bbox_tea, gt_instance in zip(cls_stus, cls_teas, bbox_stus, bbox_teas,
                                                                     grouped_instances_new):

            rescale = False
            # 筛选
            with torch.no_grad():

                scores_stu, labels_stu = cls_stu.max(1)
                scores_tea, labels_tea = cls_tea.max(1)

                mask = (scores_stu > conf_thresh) | (scores_tea > conf_thresh)  # 并集

                # 计算蒸馏损失
                score_stu_mask = cls_stu[mask]
                score_tea_mask = cls_tea[mask]
                bbox_stu_mask = bbox_stu[mask]
                bbox_tea_mask = bbox_tea[mask]

                if len(bbox_tea_mask) > 0 and len(gt_instance) > 0:
                    # 计算 bbox_stu 与 gt_instance 的最大 IoU
                    max_iou_tea = max_iou_per_bbox(bbox_tea_mask, gt_instance)  # (N,)

                    # 根据 is_old 规则筛选
                    if is_old:
                        mask_new = max_iou_tea <= iou_thresh
                    else:
                        mask_new = max_iou_tea >= iou_thresh

                    score_stu_mask = score_stu_mask[mask_new]
                    score_tea_mask = score_tea_mask[mask_new]
                    bbox_stu_mask = bbox_stu_mask[mask_new]
                    bbox_tea_mask = bbox_tea_mask[mask_new]

                if len(bbox_tea_mask) == 0:
                    rescale = True
                    score_stu_mask = cls_stu[:2]
                    score_tea_mask = cls_tea[:2]
                    bbox_stu_mask = bbox_stu[:2]
                    bbox_tea_mask = bbox_tea[:2]

            if rescale:
                loss_cls_kd_batch.append(kl_div_loss(score_stu_mask, score_tea_mask) * 0.01)
                loss_reg_kd_batch.append(bbox_l2_loss(bbox_stu_mask, bbox_tea_mask) * 0.01)
            else:
                loss_cls_kd_batch.append(kl_div_loss(score_stu_mask, score_tea_mask))
                loss_reg_kd_batch.append(bbox_l2_loss(bbox_stu_mask, bbox_tea_mask))

        if is_old:
            loss_cls_kd = sum(loss_cls_kd_batch) * self.loss_cls_kd_weight_old
            loss_reg_kd = sum(loss_reg_kd_batch) * self.loss_reg_kd_weight_old
        else:
            loss_cls_kd = sum(loss_cls_kd_batch) * self.loss_cls_kd_weight_new
            loss_reg_kd = sum(loss_reg_kd_batch) * self.loss_reg_kd_weight_new

        return loss_cls_kd, loss_reg_kd

    @staticmethod
    def align_scale(stu_feat, tea_feat):
        N, C, H, W = stu_feat.size()
        # normalize student feature
        stu_feat = stu_feat.permute(1, 0, 2, 3).reshape(C, -1)
        stu_mean = stu_feat.mean(dim=-1, keepdim=True)
        stu_std = stu_feat.std(dim=-1, keepdim=True)
        stu_feat = (stu_feat - stu_mean) / (stu_std + 1e-6)
        #
        tea_feat = tea_feat.permute(1, 0, 2, 3).reshape(C, -1)
        tea_mean = tea_feat.mean(dim=-1, keepdim=True)
        tea_std = tea_feat.std(dim=-1, keepdim=True)
        stu_feat = stu_feat * tea_std + tea_mean
        return stu_feat.reshape(C, N, H, W).permute(1, 0, 2, 3)

    @staticmethod
    def parameter_reorganization(ori_setting, cur_setting, load_from_weight):
        assert os.path.isfile(ori_setting['ckpt']), '{} is not a valid file'.format(ori_setting['ckpt'])
        assert os.path.isfile(cur_setting['ckpt']), '{} is not a valid file'.format(cur_setting['ckpt'])
        # ##### init original branches of new model #####
        # ori_model_load your models
        # 假设模型都是通过torch.load加载的或者已经实例化好的对象
        target_model_weight = torch.load(ori_setting['ckpt'], map_location='cpu')
        ori_model_weight = torch.load(ori_setting['ckpt'], map_location='cpu')
        cur_model_weight = torch.load(cur_setting['ckpt'], map_location='cpu')

        # ori_model_copy weights from source to target model
        target_model_state_dict = target_model_weight['state_dict']
        ori_model_state_dict = ori_model_weight['state_dict']
        cur_model_state_dict = cur_model_weight['state_dict']

        for key in ori_model_state_dict:
            target_model_state_dict[f'ori_model.{key}'] = ori_model_state_dict[key]
        for key in cur_model_state_dict:
            target_model_state_dict[f'cur_model.{key}'] = cur_model_state_dict[key]

        target_model_state_dict['embeddings'] = torch.cat(
            (ori_model_state_dict['embeddings'], cur_model_state_dict['embeddings']), dim=0)

        # ori_model_save the updated target model
        torch.save(target_model_weight, load_from_weight)
        print("Model weights copied successfully.")

    @staticmethod
    def freeze(model: nn.Module):
        """Freeze the model."""
        model.eval()
        for param in model.parameters():
            param.requires_grad = False


def calculate_iou_matrix(box, boxes):
    """
    计算单个框与一组框之间的 IoU。
    :param box: 单个边界框 [x1, y1, x2, y2] (Tensor)
    :param boxes: 一组边界框 (N, 4) 形状的 Tensor
    :return: 每个框的 IoU 值 Tensor
    """
    x_left = torch.maximum(box[0], boxes[:, 0])
    y_top = torch.maximum(box[1], boxes[:, 1])
    x_right = torch.minimum(box[2], boxes[:, 2])
    y_bottom = torch.minimum(box[3], boxes[:, 3])

    inter_width = torch.clamp(x_right - x_left, min=0)
    inter_height = torch.clamp(y_bottom - y_top, min=0)
    intersection = inter_width * inter_height

    box_area = (box[2] - box[0]) * (box[3] - box[1])
    boxes_area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    union = box_area + boxes_area - intersection

    iou = intersection / torch.clamp(union, min=1e-9)  # 避免除零错误
    return iou


def calculate_max_iou(bbox, bboxs):
    """
    计算一个框与一组框之间的最大 IoU。
    :param bbox: 单个边界框 [x1, y1, x2, y2] (Tensor)
    :param bboxs: 一组边界框 (N, 4) 形状的 Tensor
    :return: 最大 IoU 值
    """
    if bboxs.size(0) == 0:
        return torch.tensor(0.0, device=bbox.device)
    ious = calculate_iou_matrix(bbox, bboxs)
    return torch.max(ious)


def bbox_iou(bbox1, bbox2):
    """
    计算 bbox1 (N, 4) 和 bbox2 (M, 4) 之间的 IoU。
    bbox1 和 bbox2 格式为 (x1, y1, x2, y2)。
    返回 IoU 矩阵，形状为 (N, M)。
    """
    # 计算交集坐标
    inter_x1 = torch.max(bbox1[:, None, 0], bbox2[:, 0])  # (N, M)
    inter_y1 = torch.max(bbox1[:, None, 1], bbox2[:, 1])  # (N, M)
    inter_x2 = torch.min(bbox1[:, None, 2], bbox2[:, 2])  # (N, M)
    inter_y2 = torch.min(bbox1[:, None, 3], bbox2[:, 3])  # (N, M)

    # 计算交集面积
    inter_w = (inter_x2 - inter_x1).clamp(min=0)
    inter_h = (inter_y2 - inter_y1).clamp(min=0)
    inter_area = inter_w * inter_h  # (N, M)

    # 计算各自的面积
    area1 = (bbox1[:, 2] - bbox1[:, 0]) * (bbox1[:, 3] - bbox1[:, 1])  # (N,)
    area2 = (bbox2[:, 2] - bbox2[:, 0]) * (bbox2[:, 3] - bbox2[:, 1])  # (M,)

    # 计算 IoU
    iou = inter_area / (area1[:, None] + area2 - inter_area)  # (N, M)

    return iou


def max_iou_per_bbox(bbox1, bbox2):
    """
    计算 bbox1 中每个框与 bbox2 所有框的最大 IoU。
    返回最大 IoU 的 (N,) 张量。
    """
    iou_matrix = bbox_iou(bbox1, bbox2)  # (N, M)
    max_iou, _ = iou_matrix.max(dim=1)  # (N,)
    return max_iou


def kl_div_loss(score_stu_mask, score_tea_mask, temperature=1.0):
    """
    计算 KL 散度作为蒸馏损失

    :param score_stu_mask: 学生模型的预测分布 (未 softmax 的 logits)
    :param score_tea_mask: 教师模型的预测分布 (未 softmax 的 logits)
    :param temperature: 蒸馏温度
    :return: KL 散度损失
    """
    # 通过 softmax 获取概率分布，并应用温度缩放
    stu_prob = F.log_softmax(score_stu_mask / temperature, dim=-1)
    tea_prob = F.softmax(score_tea_mask / temperature, dim=-1)

    # 计算 KL 散度损失（reduction='batchmean' 计算样本均值）
    kl_loss = F.kl_div(stu_prob, tea_prob, reduction='batchmean') * (temperature ** 2)

    return kl_loss


def bbox_l2_loss(bbox_stu_mask, bbox_tea_mask):
    """
    计算 L2 损失（均方误差）作为目标框蒸馏损失
    :param bbox_stu_mask: 学生模型预测的目标框 (N, 4)
    :param bbox_tea_mask: 教师模型预测的目标框 (N, 4)
    :return: L2 蒸馏损失
    """
    loss = F.mse_loss(bbox_stu_mask, bbox_tea_mask, reduction='mean')
    return loss
