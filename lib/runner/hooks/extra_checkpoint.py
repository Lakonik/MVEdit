from mmcv.runner.hooks import CheckpointHook
from mmcv.runner import HOOKS
from mmcv.runner.dist_utils import allreduce_params


@HOOKS.register_module()
class ExtraCheckpointHook(CheckpointHook):
    def __init__(self, checkpoint_at: list):
        super().__init__(2 ** 30, False)
        self.checkpoint_at = checkpoint_at

    def after_train_iter(self, runner):
        if runner.iter + 1 in self.checkpoint_at:
            runner.logger.info(
                f'Saving checkpoint at {runner.iter + 1} iterations')
            if self.sync_buffer:
                allreduce_params(runner.model.buffers())
            self._save_checkpoint(runner)
