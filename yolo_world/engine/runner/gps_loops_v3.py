# Copyright (c) OpenMMLab. All rights reserved.
import bisect
import copy
import logging
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
from mmengine.logging import print_log
from mmengine.registry import LOOPS
from mmengine.runner.base_loop import BaseLoop
from torch.utils.data import DataLoader

from .utils import calc_dynamic_intervals


def _freeze_all_linear(model):
    """Freeze the model."""
    for name, param in model.named_parameters():
        if 'module.backbone.text_model' in name:
            continue

@LOOPS.register_module()
class EpochBasedTrainGPSV3Loop(BaseLoop):
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
            gps_ratio: float,
            up_ratio: float,
            val_begin: int = 1,
            val_interval: int = 1,
            importance_save_path=None,
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

        self.grad_accum_map = {}
        self.grad_mask = {}
        self.gps_ratio = gps_ratio
        self.up_ratio = up_ratio

        _freeze_all_linear(self.runner.model)

        self.ori_model = None

        self.importance_save_path = importance_save_path

    def init_ori_model(self):
        model = copy.deepcopy(self.runner.model)
        for _, param in model.named_parameters():
            param.requires_grad = False
        self.ori_model = {name: param for name, param in model.named_parameters()}


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

    def run_cal_grad(self) -> torch.nn.Module:
        """Launch training."""
        self.runner.call_hook('before_train')
        self.run_cal_grad_epoch()
        # self.runner.val_loop.run()
        self.runner.call_hook('after_train')
        self._iter = 0
        return self.runner.model

    def run(self) -> torch.nn.Module:
        """Launch training."""
        self.runner.call_hook('before_train')

        if self.ori_model is None:
            self.init_ori_model()

        while self._epoch < self._max_epochs and not self.stop_training:
            self.run_epoch()

            self._decide_current_val_interval()
            if (self.runner.val_loop is not None
                    and self._epoch >= self.val_begin
                    and (self._epoch % self.val_interval == 0
                         or self._epoch == self._max_epochs)):
                self.runner.val_loop.run()

        self.runner.call_hook('after_train')
        return self.runner.model

    def run_epoch(self) -> None:
        """Iterate one epoch."""
        self.runner.call_hook('before_train_epoch')
        self.runner.model.train()
        for idx, data_batch in enumerate(self.dataloader):
            self.run_iter(idx, data_batch)

        self.runner.call_hook('after_train_epoch')
        self._epoch += 1

    def run_cal_grad_epoch(self) -> None:
        """Iterate one epoch."""
        self.runner.call_hook('before_train_epoch')
        self.runner.model.train()
        for idx, data_batch in enumerate(self.dataloader):
            self.run_cal_grad_iter(idx, data_batch)
        grad_accum_weight = self.prune_with_layer_weighting(self.grad_accum_map)
        self.grad_mask = self.prune_by_percentile_gradient_per_cell(self.grad_accum_map, weight_map=grad_accum_weight)
        self.runner.call_hook('after_train_epoch')

    @staticmethod
    def prune_with_layer_weighting(grad_accum_map):
        """
        对每一层的梯度根据其均值进行加权，再调用原始的梯度裁剪函数
        """

        # Step 1: 计算每一层的权重（均值绝对值作为权重指标）
        layer_weights = {}
        for name, param in grad_accum_map.items():
            grad_tensor = param['grad'].data.cpu().numpy()

            if grad_tensor is None or grad_tensor.size == 0:
                continue

            # 若为标量，跳过
            if len(grad_tensor.shape) <= 2:
                continue

            # 使用绝对值均值作为层的重要性
            importance = np.mean(np.abs(grad_tensor))
            layer_weights[name] = importance

        # Step 2: 归一化所有层的权重
        total_importance = sum(layer_weights.values()) / len(layer_weights)
        for name in layer_weights:
            layer_weights[name] /= total_importance

        # Step 4: 调用原始裁剪函数进行处理
        return layer_weights

    def prune_by_percentile_gradient_per_cell(self, grad_accum_map, weight_map=None):
        statistic = {}
        new_masks = {}
        model_importance_save = {}

        for name, param in grad_accum_map.items():

            new_mask = None
            grad_tensor = param['grad'].data.cpu().numpy()

            if len(grad_tensor.shape) == 0:
                continue

            ratio = min(weight_map[name] * self.gps_ratio, self.up_ratio)

            # bn and bias
            if len(grad_tensor.shape) <= 2:
                continue

            if len(grad_tensor.shape) == 4:

                # 参数定义
                total_channels = grad_tensor.shape[0] * grad_tensor.shape[1]
                top_n = int(max(1, int(ratio * total_channels)))

                # Step 1: reshape to [kernel_size, c_in, -1]
                grad_tensor_flat = grad_tensor.reshape(grad_tensor.shape[0], grad_tensor.shape[1],
                                                       -1)  # [kernel_size, c_in, K*K]

                # Step 2: compute importance per (kernel, channel) pair
                importance = np.mean(np.abs(grad_tensor_flat), axis=2)  # shape: [kernel_size, c_in]

                model_importance_save[name] = torch.from_numpy(importance).cuda()  # [k, c]

                # Step 3: flatten importance and get top-N indices
                importance_flat = importance.flatten()  # shape: [kernel_size * c_in]
                top_indices = np.argsort(importance_flat)[-top_n:]  # indices of most important kernel-channel pairs

                # Step 4: map back to (kernel, channel) indices
                kernel_indices, channel_indices = np.unravel_index(top_indices, importance.shape)

                # Step 5: initialize mask and set selected positions to 1
                new_mask = np.zeros_like(grad_tensor, dtype=np.float32)
                for k, c in zip(kernel_indices, channel_indices):
                    new_mask[k, c, :, :] = 1.0

            trainable_param = len(np.nonzero(new_mask)[0])
            try:
                total_para = len(new_mask.reshape(-1))
            except Exception as e:
                print(e)
                print('asfergdgftbf', name)
            statistic[name] = [trainable_param, total_para]
            print(name, ": ", trainable_param, "/", total_para, "(", np.round((trainable_param / total_para) * 100, 4),"%)", new_mask.shape)

            new_masks[name] = torch.from_numpy(new_mask).cuda()

        print_log("---------------------------------------------------------------", logger='current', level=logging.INFO)
        print_log(f"base_ratio: {self.gps_ratio}", logger='current', level=logging.INFO)

        trainable_head = 0
        total_head = 0
        for na, [trainable_p, t_p] in statistic.items():
            trainable_head = trainable_head + trainable_p
            total_head = total_head + t_p
        print_log("---------------------------------------------------------------", logger='current', level=logging.INFO)

        print_log("---------------------------------------------------------------", logger='current', level=logging.INFO)
        print_log(f"Trainable parameter / Total (total): {trainable_head} \\/ {total_head} : {np.round(((trainable_head) / (total_head)) * 100, 4)}", logger='current', level=logging.INFO)

        print_log("#######################################################################", logger='current', level=logging.INFO)

        if self.importance_save_path is not None:
            print_log(f"save importance to {self.importance_save_path}", logger='current', level=logging.INFO)
            torch.save(model_importance_save, self.importance_save_path)

        return new_masks

    def run_iter(self, idx, data_batch: Sequence[dict]) -> None:
        """Iterate one min-batch.

        Args:
            data_batch (Sequence[dict]): Batch of data from dataloader.
        """

        step_kwargs = {}
        zero_kwargs = {}

        self.runner.call_hook(
            'before_train_iter', batch_idx=idx, data_batch=data_batch)
        # Enable gradient accumulation mode and avoid unnecessary gradient
        # synchronization during gradient accumulation process.
        # outputs should be a dict of loss.
        with self.runner.optim_wrapper.optim_context(self):
            data = self.runner.model.module.data_preprocessor(data_batch, training=True)
            losses = self.runner.model._run_forward(data, mode='loss')

        parsed_losses, log_vars = self.runner.model.module.parse_losses(losses)  # type: ignore
        outputs = log_vars

        loss = self.runner.optim_wrapper.scale_loss(parsed_losses)
        self.runner.optim_wrapper.backward(loss)

        if self.runner.optim_wrapper.should_update():
            self.runner.optim_wrapper.step(**step_kwargs)

            # 参数mask赋值
            for param_name, param in self.runner.model.named_parameters():

                if not param.requires_grad:
                    continue

                if len(param.shape) <= 2:
                    continue

                # if param_name not in self.grad_mask:
                #     continue
                if len(param.shape) > 2:
                    with torch.no_grad():
                        ori_param = self.ori_model[param_name]
                        mask_ = self.grad_mask[param_name]
                        param.mul_(mask_).add_(ori_param * (1 - mask_))

            self.runner.optim_wrapper.zero_grad(**zero_kwargs)

        self.runner.call_hook(
            'after_train_iter',
            batch_idx=idx,
            data_batch=data_batch,
            outputs=outputs)
        self._iter += 1

    def run_cal_grad_iter(self, idx, data_batch: Sequence[dict]) -> None:
        """Iterate one min-batch.

        Args:
            data_batch (Sequence[dict]): Batch of data from dataloader.
        """
        step_kwargs = {}
        zero_kwargs = {}

        self.runner.call_hook(
            'before_train_iter', batch_idx=idx, data_batch=data_batch)
        # Enable gradient accumulation mode and avoid unnecessary gradient
        # synchronization during gradient accumulation process.
        # outputs should be a dict of loss.
        with self.runner.optim_wrapper.optim_context(self):
            data = self.runner.model.module.data_preprocessor(data_batch, training=True)
            losses = self.runner.model._run_forward(data, mode='loss')

        for k in losses.keys():
            if 'distill' in k:
                losses[k] = losses[k] * 0.00001

        parsed_losses, log_vars = self.runner.model.module.parse_losses(losses)  # type: ignore
        outputs = log_vars

        loss = self.runner.optim_wrapper.scale_loss(parsed_losses)
        self.runner.optim_wrapper.backward(loss)

        # === 使用 map 累加梯度 ===
        for name, param in self.runner.model.named_parameters():

            if param.grad is not None:
                if torch.isnan(param.grad).any() or torch.isinf(param.grad).any():
                    continue
                if len(param.shape) <= 2:
                    continue
                if name not in self.grad_accum_map:
                    self.grad_accum_map[name] = {'grad': torch.abs(param.grad.detach().clone()), 'param': param}
                else:
                    self.grad_accum_map[name]['grad'] += torch.abs(param.grad.detach())

        if self.runner.optim_wrapper.should_update():
            self.runner.optim_wrapper.step(**step_kwargs)
            self.runner.optim_wrapper.zero_grad(**zero_kwargs)

        # outputs = self.runner.model.train_step(
        #     data_batch, optim_wrapper=self.runner.optim_wrapper)

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