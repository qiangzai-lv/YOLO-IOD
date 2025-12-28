# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmyolo.registry import MODELS


def knowledge_distillation_kl_div_loss(pred,
                                       soft_label,
                                       T,
                                       class_reduction='mean',
                                       detach_target=True):
    r"""Loss function for knowledge distilling using KL divergence.

    Args:
        pred (Tensor): Predicted logits with shape (N, n + 1).
        soft_label (Tensor): Target logits with shape (N, N + 1).
        T (int): Temperature for distillation.
        detach_target (bool): Remove soft_label from automatic differentiation

    Returns:
        torch.Tensor: Loss tensor with shape (N,).
    """
    assert pred.size() == soft_label.size()
    target = F.softmax(soft_label / T, dim=1)
    if detach_target:
        target = target.detach()

    kd_loss = F.kl_div(
        F.log_softmax(pred / T, dim=1), target, reduction='none')
    if class_reduction == 'mean':
        kd_loss = kd_loss.mean(1)
    elif class_reduction == 'sum':
        kd_loss = kd_loss.sum(1)
    else:
        raise NotImplementedError
    kd_loss = kd_loss * (T * T)
    return kd_loss


@MODELS.register_module()
class KnowledgeDistillationKLDivLoss(nn.Module):
    """Loss function for knowledge distilling using KL divergence.

    Args:
        reduction (str): Options are `'none'`, `'mean'` and `'sum'`.
        loss_weight (float): Loss weight of current loss.
        T (int): Temperature for distillation.
    """

    def __init__(self,
                 class_reduction='mean',
                 reduction='mean',
                 loss_weight=1.0,
                 T=10):
        super(KnowledgeDistillationKLDivLoss, self).__init__()
        assert T >= 1
        self.class_reduction = class_reduction
        self.reduction = reduction
        self.loss_weight = loss_weight
        self.T = T

    def forward(self,
                pred,
                soft_label,
                weight=None,
                avg_factor=None,
                reduction_override=None):
        """Forward function.

        Args:
            pred (Tensor): Predicted logits with shape (N, n + 1).
            soft_label (Tensor): Target logits with shape (N, N + 1).
            weight (torch.Tensor, optional): The weight of loss for each
                prediction. Defaults to None.
            avg_factor (int, optional): Average factor that is used to average
                the loss. Defaults to None.
            reduction_override (str, optional): The reduction method used to
                override the original reduction method of the loss.
                Defaults to None.
        """
        assert reduction_override in (None, 'none', 'mean', 'sum')

        reduction = (
            reduction_override if reduction_override else self.reduction)

        loss_kd = self.loss_weight * knowledge_distillation_kl_div_loss(
            pred,
            soft_label,
            weight,
            class_reduction=self.class_reduction,
            reduction=reduction,
            avg_factor=avg_factor,
            T=self.T)

        return loss_kd


@MODELS.register_module()
class KDQualityFocalLoss(nn.Module):
    def __init__(self, use_sigmoid=True, beta=1.0, reduction='mean', loss_weight=1.0):
        super().__init__()
        assert use_sigmoid, "Only sigmoid version of QFL is supported."
        self.beta = beta
        self.reduction = reduction
        self.loss_weight = loss_weight

    def forward(self,
                pred: torch.Tensor,  # (B, N, C)
                target: torch.Tensor):  # (B, N, C):

        # Apply sigmoid to target only (teacher output is usually logits)
        target_sigmoid = target.detach().sigmoid()

        # BCE loss per element
        loss = F.binary_cross_entropy_with_logits(pred, target_sigmoid, reduction='none')  # (B, N, C)

        # Focal weight: |p - t|^beta
        focal_weight = torch.abs(pred.sigmoid() - target_sigmoid).pow(self.beta)
        loss = loss * focal_weight  # (B, N, C)

        return loss.sum() * self.loss_weight