import inspect
from typing import List

import bitsandbytes

from mmcv.runner.optimizer.builder import OPTIMIZERS


def register_bitsandbytes_optimizers() -> List:
    bitsandbytes_optimizers = []
    for module_name in dir(bitsandbytes.optim):
        if module_name.startswith('__'):
            continue
        _optim = getattr(bitsandbytes.optim, module_name)
        if inspect.isclass(_optim) and issubclass(_optim, bitsandbytes.optim.optimizer.Optimizer2State) \
                and module_name not in OPTIMIZERS.module_dict:
            OPTIMIZERS.register_module()(_optim)
            bitsandbytes_optimizers.append(module_name)
    return bitsandbytes_optimizers


BNB_OPTIMIZERS = register_bitsandbytes_optimizers()
