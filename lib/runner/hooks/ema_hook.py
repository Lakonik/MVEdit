from copy import deepcopy

import mmcv
import torch
from mmcv.parallel import is_module_wrapper
from mmcv.runner import HOOKS

from mmgen.core import ExponentialMovingAverageHook
from ...core import rgetattr, rhasattr


def get_ori_key(key):
    ori_key = key.split('.')
    ori_key[0] = ori_key[0][:-4]
    ori_key = '.'.join(ori_key)
    return ori_key


@HOOKS.register_module()
class ExponentialMovingAverageHookMod(ExponentialMovingAverageHook):

    def __init__(self,
                 module_keys,
                 trainable_only=True,
                 interp_mode='lerp',
                 interp_cfg=None,
                 interval=-1,
                 start_iter=0,
                 momentum_policy='fixed',
                 momentum_cfg=None):
        super(ExponentialMovingAverageHook, self).__init__()
        self.trainable_only = trainable_only
        # check args
        assert interp_mode in self._registered_interp_funcs, (
            'Supported '
            f'interpolation functions are {self._registered_interp_funcs}, '
            f'but got {interp_mode}')

        assert momentum_policy in self._registered_momentum_updaters, (
            'Supported momentum policy are'
            f'{self._registered_momentum_updaters},'
            f' but got {momentum_policy}')

        assert isinstance(module_keys, str) or mmcv.is_tuple_of(
            module_keys, str)
        self.module_keys = (module_keys, ) if isinstance(module_keys,
                                                         str) else module_keys
        # sanity check for the format of module keys
        for k in self.module_keys:
            assert k.split('.')[0].endswith(
                '_ema'), 'You should give keys that end with "_ema".'
        self.interp_mode = interp_mode
        self.interp_cfg = dict() if interp_cfg is None else deepcopy(
            interp_cfg)
        self.interval = interval
        self.start_iter = start_iter

        assert hasattr(
            self, interp_mode
        ), f'Currently, we do not support {self.interp_mode} for EMA.'
        self.interp_func = getattr(self, interp_mode)

        self.momentum_cfg = dict() if momentum_cfg is None else deepcopy(
            momentum_cfg)
        self.momentum_policy = momentum_policy
        if momentum_policy != 'fixed':
            assert hasattr(
                self, momentum_policy
            ), f'Currently, we do not support {self.momentum_policy} for EMA.'
            self.momentum_updater = getattr(self, momentum_policy)

    def after_train_iter(self, runner):
        if not self.every_n_iters(runner, self.interval):
            return

        with torch.no_grad():
            model = runner.model.module if is_module_wrapper(
                runner.model) else runner.model

            # update momentum
            _interp_cfg = deepcopy(self.interp_cfg)
            if self.momentum_policy != 'fixed':
                _updated_args = self.momentum_updater(runner, **self.momentum_cfg)
                _interp_cfg.update(_updated_args)

            for key in self.module_keys:
                # get current ema states
                ema_net = rgetattr(model, key)
                states_ema = ema_net.state_dict(keep_vars=False)
                # get currently original states
                net = rgetattr(model, get_ori_key(key))
                states_orig = net.state_dict(keep_vars=True)

                for k, v in states_orig.items():
                    if self.trainable_only and not v.requires_grad:
                        continue
                    if runner.iter < self.start_iter:
                        states_ema[k].data.copy_(v.data)
                    else:
                        states_ema[k].data.copy_(self.interp_func(
                            v,
                            states_ema[k],
                            trainable=v.requires_grad,
                            **_interp_cfg))
                # ema_net.load_state_dict(states_ema, strict=True)

    def before_run(self, runner):
        model = runner.model.module if is_module_wrapper(
            runner.model) else runner.model
        # sanity check for ema model
        for k in self.module_keys:
            if not rhasattr(model, k):
                raise RuntimeError(
                    f'Cannot find {k} network for EMA hook.')
