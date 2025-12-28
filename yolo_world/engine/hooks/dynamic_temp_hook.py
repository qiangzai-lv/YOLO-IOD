# your_project/hooks/dynamic_temp_hook.py
from mmengine import print_log
from mmengine.hooks import Hook
from mmengine.registry import HOOKS

@HOOKS.register_module()
class DynamicTemperatureHook(Hook):
    def __init__(self, total_epochs=100, loss_cls_kd_temperature_old = 0, loss_reg_kd_temperature_old = 0,):
        self.total_epochs = total_epochs


    def before_train_epoch(self, runner):
        # 当前 epoch
        cur_epoch = runner.epoch
        # old 线性衰减 5 - 0.5
        base_old_tem = 5
        final_old_tem = 0.5
        old_ratio = base_old_tem - (base_old_tem - final_old_tem) * (cur_epoch / self.total_epochs)
        runner.model.module.loss_cls_kd_temperature_old = old_ratio * runner.model.module.loss_cls_kd_temperature_old
        runner.model.module.loss_reg_kd_temperature_old = old_ratio * runner.model.module.loss_reg_kd_temperature_old
        runner.logger.info(f'[DynamicTemperatureHook] Epoch {cur_epoch} {old_ratio}: {runner.model.module.loss_cls_kd_temperature_old} {runner.model.module.loss_reg_kd_temperature_old}')

        # new 线性增 0.05 - 3
        base_new_tem = 0.05
        final_new_tem = 3
        new_ratio = base_new_tem + (final_new_tem - base_new_tem) * (cur_epoch / self.total_epochs)
        runner.model.module.loss_cls_kd_temperature_new = new_ratio * runner.model.module.loss_cls_kd_temperature_new
        runner.model.module.loss_reg_kd_temperature_new = new_ratio * runner.model.module.loss_reg_kd_temperature_new
        runner.logger.info(f'[DynamicTemperatureHook] Epoch {cur_epoch} {new_ratio}: {runner.model.module.loss_cls_kd_temperature_new} {runner.model.module.loss_reg_kd_temperature_new}')
