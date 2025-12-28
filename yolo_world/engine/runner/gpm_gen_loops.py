# Copyright (c) OpenMMLab. All rights reserved.
import bisect
import logging
import os
from typing import Dict, List, Optional, Sequence, Tuple, Union
import torch.distributed as dist

import torch
import torch.nn.functional as F
from mmengine.logging import print_log
from mmengine.registry import LOOPS
from mmengine.runner.base_loop import BaseLoop
from torch import nn
from torch.utils.data import DataLoader

from .utils import calc_dynamic_intervals


def im2col_2d(input_tensor, kernel_size, stride=1, padding=0):
    return F.unfold(input_tensor, kernel_size=kernel_size, stride=stride, padding=padding)

def _freeze_all(model):
    """Freeze the model."""
    for name, param in model.named_parameters():
        param.requires_grad = False


def _freeze_all_linear(model):
    """Freeze the model."""
    for name, param in model.named_parameters():

        if 'module.backbone.text_model' in name:
            continue
        if 'module.backbone.image_model.stem' in name:
            param.requires_grad = False
        elif 'attn_block.guide_fc' in name:
            param.requires_grad = False
        elif 'cls_contrasts' in name:
            param.requires_grad = False


@LOOPS.register_module()
class EpochBasedTrainGPMGenLoop(BaseLoop):
    """Loop for epoch-based training.

    Args:
        runner (Runner): A reference of runner.
        dataloader (Dataloader or dict): A dataloader object or a dict to
            build a dataloader.
        max_epochs (int): Total training epochs.
        val_begin (int): The epoch that begins validating.
            Defaults to 1.
        val_interval (int): Validation interval. Defaults to 1.
        dynamic_intervals (List[Tuple[int, int]], optional): The
            first element in the tuple is a milestone and the second
            element is a interval. The interval is used after the
            corresponding milestone. Defaults to None.
    """

    def __init__(
            self,
            runner,
            dataloader: Union[DataLoader, Dict],
            max_epochs: int,
            val_begin: int = 1,
            val_interval: int = 1,
            gpm_cal_item_num: int = 1,
            gpm_mat_save_path: str = None,
            threshold_stage_1: float = 0.50,
            dynamic_intervals: Optional[List[Tuple[int, int]]] = None) -> None:
        super().__init__(runner, dataloader)
        self._max_epochs = int(max_epochs)
        assert self._max_epochs == max_epochs, \
            f'`max_epochs` should be a integer number, but get {max_epochs}.'
        self._max_iters = self._max_epochs * len(self.dataloader)
        self._epoch = 0
        self._iter = 0
        self.val_begin = val_begin
        self.val_interval = val_interval
        self.gpm_cal_item_num = gpm_cal_item_num
        # This attribute will be updated by `EarlyStoppingHook`
        # when it is enabled.
        self.stop_training = False
        if hasattr(self.dataloader.dataset, 'metainfo'):
            self.runner.visualizer.dataset_meta = \
                self.dataloader.dataset.metainfo
        else:
            print_log(
                f'Dataset {self.dataloader.dataset.__class__.__name__} has no '
                'metainfo. ``dataset_meta`` in visualizer will be '
                'None.',
                logger='current',
                level=logging.WARNING)

        self.dynamic_milestones, self.dynamic_intervals = \
            calc_dynamic_intervals(
                self.val_interval, dynamic_intervals)

        self.gpm_mat_save_path = gpm_mat_save_path
        self.threshold_stage_1 = threshold_stage_1
        self.mat_map = {}
        self.model_name_map = {}
        self.local_rank = int(os.environ.get('LOCAL_RANK', 0))
        self.world_size = dist.get_world_size()

        # freeze all
        _freeze_all(self.runner.model)


    @property
    def max_epochs(self):
        """int: Total epochs to train model."""
        return self._max_epochs

    @property
    def max_iters(self):
        """int: Total iterations to train model."""
        return self._max_iters

    @property
    def epoch(self):
        """int: Current epoch."""
        return self._epoch

    @property
    def iter(self):
        """int: Current iteration."""
        return self._iter

    def run(self) -> torch.nn.Module:
        """Execute loop."""
        self.runner.call_hook('before_train')
        self.run_get_representation_matrix_epoch()
        # self.runner.val_loop.run()
        self.runner.call_hook('after_train')
        return self.runner.model

    def run_get_representation_matrix_epoch(self) -> None:
        """Iterate one epoch."""
        self.runner.call_hook('before_train_epoch')
        self.runner.model.train()
        # TODO 样本筛选
        remaining_samples_iter = self.gpm_cal_item_num
        model = self.runner.model
        self.register_hooks(model)
        for idx, data_batch in enumerate(self.dataloader):
            print_log(f"remain iter:{remaining_samples_iter}", logger='current', level=logging.INFO)
            if remaining_samples_iter <= 0:
                break
            self.run_cal_representation_matrix_iter(idx, data_batch)
            remaining_samples_iter -= 1
            print_log("--------------------------------------------------------", logger='current', level=logging.INFO)
            gpm_mat_save_path = self.gpm_mat_save_path.replace('.pt', f'_{self.local_rank}_{idx}.pt')
            print_log("representation matrix cal finish save to {}".format(gpm_mat_save_path), logger='current',
                      level=logging.INFO)
            torch.save(self.mat_map, gpm_mat_save_path)
            print_log("--------------------------------------------------------", logger='current', level=logging.INFO)
            self.mat_map.clear()
            self.mat_map = {}

        self.runner.call_hook('after_train_epoch')

    def register_hooks(self, register_model):

        def cnn_hook_fn(module, inputs, output):
            # B, C_out, H, W = output.shape
            input_cols = im2col_2d(inputs[0], module.kernel_size, module.stride, module.padding)

            # svd stage1 消除背景
            B = input_cols.shape[0]
            svd_results_top = []
            for i in range(B):
                matrix = input_cols[i]  # shape: (C*kH*kW, L)
                U, S, Vh = torch.linalg.svd(matrix, full_matrices=False)  # U: (M, M), S: (min(M, N)), Vh: (N, N)
                # 计算累积能量比例
                sval_total = (S ** 2).sum()
                sval_ratio = (S ** 2) / sval_total
                cumulative_energy = torch.cumsum(sval_ratio, dim=0)

                # 找到前90%能量对应的主成分数量 k
                k = torch.searchsorted(cumulative_energy, self.threshold_stage_1).item() + 1  # +1 保证覆盖90%
                # 选取前 k 个主成分
                U90 = U[:, :k]
                svd_results_top.append(U90)
            svd_results_top = torch.cat(svd_results_top, dim=1).to("cpu")
            if self.local_rank == 0:
                print_log(f"{self.model_name_map[module]}, iter save svd: {svd_results_top.shape}")

            if self.model_name_map[module] not in self.mat_map:
                self.mat_map[self.model_name_map[module]] = svd_results_top
            else:
                self.mat_map[self.model_name_map[module]] = torch.cat(
                    (self.mat_map[self.model_name_map[module]], svd_results_top), dim=1)

        for name, module in register_model.named_modules():
            if name.startswith('module.backbone.text_model.model.text_model.encoder'):
                continue
            if isinstance(module, nn.Conv2d):
                print_log(f"Registering hook for: {name} ({module.__class__.__name__})")
                module.register_forward_hook(cnn_hook_fn)
                self.model_name_map[module] = name

    def run_cal_representation_matrix_iter(self, idx, data_batch: Sequence[dict]) -> None:
        """Iterate one min-batch.

        Args:
            idx:
            data_batch (Sequence[dict]): Batch of data from dataloader.
        """
        self.runner.call_hook(
            'before_train_iter', batch_idx=idx, data_batch=data_batch)

        data = self.runner.model.module.data_preprocessor(data_batch, training=True)
        losses = self.runner.model._run_forward(data, mode='loss')

        _, outputs = self.runner.model.module.parse_losses(losses)  # type: ignore

        self.runner.call_hook(
            'after_train_iter',
            batch_idx=idx,
            data_batch=data_batch,
            outputs=outputs)
        self._iter += 1

    def _decide_current_val_interval(self) -> None:
        """Dynamically modify the ``val_interval``."""
        step = bisect.bisect(self.dynamic_milestones, (self.epoch + 1))
        self.val_interval = self.dynamic_intervals[step - 1]
