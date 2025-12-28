# Copyright (c) Tencent Inc. All rights reserved.

import os
from pathlib import Path
from typing import List, Union
from typing import Tuple

import torch
import torch.nn as nn
from mmdet.registry import MODELS
from mmdet.structures import OptSampleList
from mmdet.structures import SampleList
from mmdet.utils import (ConfigType, OptConfigType)
from mmengine.config import Config
from mmyolo.models.detectors import YOLODetector
from mmyolo.registry import MODELS
from torch import Tensor
from torchvision.ops import generalized_box_iou


@MODELS.register_module()
class YOLOWorldCrossKdDetector(YOLODetector):
    """Implementation of YOLOW Series"""

    def __init__(self,
                 *args,
                 mm_neck: bool = False,
                 num_train_classes=80,
                 num_test_classes=80,
                 load_from_weight=None,
                 ori_setting: ConfigType = None,
                 cur_setting: ConfigType = None,
                 kd_cfg: OptConfigType = None,
                 **kwargs) -> None:
        self.mm_neck = mm_neck
        self.num_train_classes = num_train_classes
        self.num_test_classes = num_test_classes
        super().__init__(*args, **kwargs)

        # Build old model
        assert isinstance(ori_setting.config, (str, Path)), 'ori_setting config must be str or Path'
        assert ori_setting.ckpt is not None, 'ori_setting ckpt must not be None'
        ori_config = Config.fromfile(ori_setting.config)
        ori_config["model"]["bbox_head"]["type"] = 'YOLOWorldCrossKdHead'
        ori_config["model"]["bbox_head"]["head_module"]["type"] = 'YOLOWorldCrossKdHeadModule'
        self.ori_model = MODELS.build(ori_config['model'])
        # Build cur model
        assert isinstance(cur_setting.config, (str, Path)), 'cur_setting config must be str or Path'
        assert cur_setting.ckpt is not None, 'cur_setting ckpt must not be None'
        cur_config = Config.fromfile(cur_setting.config)
        cur_config["model"]["bbox_head"]["type"] = 'YOLOWorldCrossKdHead'
        cur_config["model"]["bbox_head"]["head_module"]["type"] = 'YOLOWorldCrossKdHeadModule'
        self.cur_model = MODELS.build(cur_config['model'])
        # freeze
        self.freeze(self.cur_model)
        self.freeze(self.ori_model)
        # kd loss
        self.loss_cls_kd_temperature_old = kd_cfg['loss_cls_kd_temperature_old']
        self.loss_reg_kd_temperature_old = kd_cfg['loss_reg_kd_temperature_old']
        self.loss_cls_kd_temperature_new = kd_cfg['loss_cls_kd_temperature_new']
        self.loss_reg_kd_temperature_new = kd_cfg['loss_reg_kd_temperature_new']
        self.kd_new = kd_cfg['kd_new']
        self.class_names = kd_cfg.get('class_names', ())
        self.ori_num_classes = kd_cfg.get('ori_num_classes', 5)
        # 参数重组
        self.parameter_reorganization(ori_setting, cur_setting, load_from_weight)
        # kd_loss
        self.loss_cls_kd = MODELS.build(kd_cfg['loss_cls_kd'])
        self.loss_reg_kd = MODELS.build(kd_cfg['loss_reg_kd'])

    @staticmethod
    def parameter_reorganization(ori_setting, cur_setting, load_from_weight):
        assert os.path.isfile(ori_setting['ckpt']), '{} is not a valid file'.format(ori_setting['ckpt'])
        assert os.path.isfile(cur_setting['ckpt']), '{} is not a valid file'.format(cur_setting['ckpt'])

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

        # ori_model_save the updated target model
        torch.save(target_model_weight, load_from_weight)
        print("Model weights copied successfully.")

    @staticmethod
    def freeze(model: nn.Module):
        """Freeze the model."""
        model.eval()
        for param in model.parameters():
            param.requires_grad = False

    def loss(self, batch_inputs: Tensor,
             batch_data_samples: SampleList) -> Union[dict, list]:
        """Calculate losses from a batch of inputs and data samples."""
        self.bbox_head.num_classes = self.num_train_classes
        img_feats, txt_feats = self.extract_feat(batch_inputs,
                                                 batch_data_samples)

        losses = self.bbox_head.loss(img_feats, txt_feats, batch_data_samples)

        # old_model forward kd
        texts_old = list(self.class_names[:self.ori_num_classes])
        self.ori_model.bbox_head.num_classes = len(texts_old)
        img_feats_old, txt_feats_old = self.extract_feat(batch_inputs, batch_data_samples, ps_texts=texts_old,
                                                         ps_backbone=self.ori_model.backbone,
                                                         ps_neck=self.ori_model.neck)
        flatten_cls_embeds_old, flatten_cls_preds_old, flatten_pred_bboxes_old = self.ori_model.bbox_head.forward_with_feat(
            img_feats_old, txt_feats_old, batch_data_samples)

        image_feat_scale_old = [self.align_scale(im, im_old) for im, im_old in zip(img_feats, img_feats_old)]

        flatten_cls_embeds_target_old, flatten_cls_preds_target_old, flatten_pred_bboxes_target_old = self.ori_model.bbox_head.forward_with_feat(
            image_feat_scale_old, txt_feats_old, batch_data_samples)

        loss_cls_kd_old = self.loss_cls_kd(flatten_cls_preds_target_old, flatten_cls_preds_old)
        # 计算回归蒸馏损失
        loss_reg_kd_old = self.loss_bbox_iou_kd(
            pred_bboxes=flatten_pred_bboxes_target_old,
            gt_bboxes=flatten_pred_bboxes_old,
            cls_scores=flatten_cls_preds_old.max(dim=2)[0]  # max over classes -> shape (B, N)
        )

        losses['distill_loss_cls_old'] = loss_cls_kd_old * self.loss_cls_kd_temperature_old
        losses['distill_loss_reg_old'] = loss_reg_kd_old * self.loss_reg_kd_temperature_old

        if not self.kd_new:
            return losses

        # new model forward kd
        texts_new = list(self.class_names[self.ori_num_classes:])
        self.cur_model.bbox_head.num_classes = len(texts_new)
        img_feats_new, txt_feats_new = self.extract_feat(batch_inputs, batch_data_samples, ps_texts=texts_new,
                                                         ps_backbone=self.cur_model.backbone,
                                                         ps_neck=self.cur_model.neck)
        flatten_cls_embeds_new, flatten_cls_preds_new, flatten_pred_bboxes_new = self.cur_model.bbox_head.forward_with_feat(
            img_feats_new, txt_feats_new, batch_data_samples)

        image_feat_scale_new = [self.align_scale(im, im_new) for im, im_new in zip(img_feats, img_feats_new)]

        flatten_cls_embeds_target_new, flatten_cls_preds_target_new, flatten_pred_bboxes_target_new = self.cur_model.bbox_head.forward_with_feat(
            image_feat_scale_new, txt_feats_new, batch_data_samples)

        loss_cls_kd_new = self.loss_cls_kd(flatten_cls_preds_target_new, flatten_cls_preds_new)
        # 计算回归蒸馏损失
        loss_reg_kd_new = self.loss_bbox_iou_kd(
            pred_bboxes=flatten_pred_bboxes_target_new,
            gt_bboxes=flatten_pred_bboxes_new,
            cls_scores=flatten_cls_preds_new.max(dim=2)[0]  # max over classes -> shape (B, N)
        )

        losses['distill_loss_cls_new'] = loss_cls_kd_new * self.loss_cls_kd_temperature_new
        losses['distill_loss_reg_new'] = loss_reg_kd_new * self.loss_reg_kd_temperature_new

        return losses

    @staticmethod
    def loss_bbox_iou_kd(pred_bboxes: Tensor, gt_bboxes: Tensor, cls_scores: Tensor, iou_thresh: float = 0.1) -> Tensor:
        """
        Regression distillation loss.
        Args:
            pred_bboxes: (B, N, 4) - student predicted bboxes
            gt_bboxes: (B, N, 4) - teacher predicted bboxes
            cls_scores: (B, N) - classification logits from teacher
            iou_thresh: threshold for selecting boxes
        Returns:
            weighted IoU loss
        """
        # Sigmoid the classification scores
        cls_probs = cls_scores.sigmoid()  # (B, N)

        # Create mask for boxes with prob > 0.1
        mask = cls_probs > iou_thresh  # (B, N)

        total_loss = 0.0
        total_weight = 0.0

        B, N, _ = pred_bboxes.shape
        for b in range(B):
            valid_idx = mask[b].nonzero(as_tuple=False).squeeze(1)  # indices of valid boxes
            if valid_idx.numel() == 0:
                continue

            pb = pred_bboxes[b][valid_idx]  # (K, 4)
            gb = gt_bboxes[b][valid_idx]  # (K, 4)
            ws = cls_probs[b][valid_idx]  # (K,)

            # IoU loss: 1 - IoU
            iou = generalized_box_iou(pb, gb).diag()  # (K,)
            iou_loss = 1.0 - iou

            weighted_loss = (iou_loss * ws).sum()
            total_loss += weighted_loss
            total_weight += ws.sum()

        if total_weight > 0:
            return total_loss
        else:
            return torch.tensor(0.0, device=pred_bboxes.device)

    def predict(self,
                batch_inputs: Tensor,
                batch_data_samples: SampleList,
                rescale: bool = True) -> SampleList:
        """Predict results from a batch of inputs and data samples with post-
        processing.
        """

        img_feats, txt_feats = self.extract_feat(batch_inputs,
                                                 batch_data_samples)

        # self.bbox_head.num_classes = self.num_test_classes
        self.bbox_head.num_classes = txt_feats[0].shape[0]
        results_list = self.bbox_head.predict(img_feats,
                                              txt_feats,
                                              batch_data_samples,
                                              rescale=rescale)

        batch_data_samples = self.add_pred_to_datasample(
            batch_data_samples, results_list)
        return batch_data_samples

    def reparameterize(self, texts: List[List[str]]) -> None:
        # encode text embeddings into the detector
        self.texts = texts
        self.text_feats = self.backbone.forward_text(texts)

    def _forward(
            self,
            batch_inputs: Tensor,
            batch_data_samples: OptSampleList = None) -> Tuple[List[Tensor]]:
        """Network forward process. Usually includes backbone, neck and head
        forward without any post-processing.
        """
        img_feats, txt_feats = self.extract_feat(batch_inputs,
                                                 batch_data_samples)
        results = self.bbox_head.forward(img_feats, txt_feats)
        return results

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

    def extract_feat(
            self, batch_inputs: Tensor,
            batch_data_samples: SampleList, ps_texts=None, ps_backbone=None, ps_neck=None) -> Tuple[
        Tuple[Tensor], Tensor]:
        """Extract features."""
        txt_feats = None
        if batch_data_samples is None:
            texts = self.texts
            txt_feats = self.text_feats
        elif isinstance(batch_data_samples,
                        dict) and 'texts' in batch_data_samples:
            texts = batch_data_samples['texts']
        elif isinstance(batch_data_samples, list) and hasattr(
                batch_data_samples[0], 'texts'):
            texts = [data_sample.texts for data_sample in batch_data_samples]
        elif hasattr(self, 'text_feats'):
            texts = self.texts
            txt_feats = self.text_feats
        else:
            raise TypeError('batch_data_samples should be dict or list.')
        if txt_feats is not None:
            # forward image only
            img_feats = self.backbone.forward_image(batch_inputs)
        else:
            if ps_texts is not None:
                texts = [ps_texts for _ in range(len(texts))]
            if ps_backbone is not None:
                img_feats, txt_feats = ps_backbone(batch_inputs, texts)
            else:
                img_feats, txt_feats = self.backbone(batch_inputs, texts)
        if self.with_neck:
            if self.mm_neck:
                if ps_neck is not None:
                    img_feats = ps_neck(img_feats, txt_feats)
                else:
                    img_feats = self.neck(img_feats, txt_feats)
            else:
                img_feats = self.neck(img_feats)
        return img_feats, txt_feats
