# Copyright (c) Tencent Inc. All rights reserved.
from .dynamic_loss import CoVMSELoss
from .kd_loss import KDQualityFocalLoss, KnowledgeDistillationKLDivLoss
from .erd_loss import KnowledgeDistillationERDLoss

__all__ = ['CoVMSELoss', 'KDQualityFocalLoss', 'KnowledgeDistillationERDLoss']
