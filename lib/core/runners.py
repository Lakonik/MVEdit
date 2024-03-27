import os.path as osp
import time
import platform
import shutil
import torch
import mmcv

from collections import OrderedDict
from tempfile import TemporaryDirectory
from torch.optim import Optimizer
from mmcv.runner import RUNNERS
from mmcv.parallel import is_module_wrapper
from mmcv.runner.checkpoint import weights_to_cpu
from mmgen.core import DynamicIterBasedRunner


def _save_to_state_dict(module, destination, prefix, keep_vars, trainable_only=False):
    for name, param in module._parameters.items():
        if param is not None and (not trainable_only or param.requires_grad):
            destination[prefix + name] = param if keep_vars else param.detach()
    for name, buf in module._buffers.items():
        # remove check of _non_persistent_buffers_set to allow nn.BatchNorm2d
        if buf is not None:
            destination[prefix + name] = buf if keep_vars else buf.detach()


def get_state_dict(module,
                   destination=None,
                   prefix='',
                   keep_vars=False,
                   trainable_only=False):
    # recursively check parallel module in case that the model has a
    # complicated structure, e.g., nn.Module(nn.Module(DDP))
    if is_module_wrapper(module):
        module = module.module

    # below is the same as torch.nn.Module.state_dict()
    if destination is None:
        destination = OrderedDict()
        destination._metadata = OrderedDict()  # type: ignore
    destination._metadata[prefix[:-1]] = local_metadata = dict(  # type: ignore
        version=module._version)
    _save_to_state_dict(module, destination, prefix, keep_vars, trainable_only=trainable_only)  # type: ignore
    for name, child in module._modules.items():
        if child is not None:
            get_state_dict(
                child, destination, prefix + name + '.', keep_vars=keep_vars, trainable_only=trainable_only)
    for hook in module._state_dict_hooks.values():
        hook_result = hook(module, destination, prefix, local_metadata)
        if hook_result is not None:
            destination = hook_result
    return destination  # type: ignore


def save_checkpoint(model,
                    filename,
                    optimizer=None,
                    loss_scaler=None,
                    save_apex_amp=False,
                    meta=None,
                    trainable_only=False):
    if meta is None:
        meta = {}
    elif not isinstance(meta, dict):
        raise TypeError(f'meta must be a dict or None, but got {type(meta)}')
    meta.update(mmcv_version=mmcv.__version__, time=time.asctime())

    if is_module_wrapper(model):
        model = model.module

    if hasattr(model, 'CLASSES') and model.CLASSES is not None:
        # save class name to the meta
        meta.update(CLASSES=model.CLASSES)

    checkpoint = {
        'meta': meta,
        'state_dict': weights_to_cpu(get_state_dict(model, trainable_only=trainable_only))
    }
    # save optimizer state dict in the checkpoint
    if isinstance(optimizer, Optimizer):
        checkpoint['optimizer'] = optimizer.state_dict()
    elif isinstance(optimizer, dict):
        checkpoint['optimizer'] = {}
        for name, optim in optimizer.items():
            checkpoint['optimizer'][name] = optim.state_dict()

    # save loss scaler for mixed-precision (FP16) training
    if loss_scaler is not None:
        checkpoint['loss_scaler'] = loss_scaler.state_dict()

    # save state_dict from apex.amp
    if save_apex_amp:
        from apex import amp
        checkpoint['amp'] = amp.state_dict()

    if filename.startswith('pavi://'):
        try:
            from pavi import modelcloud
            from pavi.exception import NodeNotFoundError
        except ImportError:
            raise ImportError(
                'Please install pavi to load checkpoint from modelcloud.')
        model_path = filename[7:]
        root = modelcloud.Folder()
        model_dir, model_name = osp.split(model_path)
        try:
            model = modelcloud.get(model_dir)
        except NodeNotFoundError:
            model = root.create_training_model(model_dir)
        with TemporaryDirectory() as tmp_dir:
            checkpoint_file = osp.join(tmp_dir, model_name)
            with open(checkpoint_file, 'wb') as f:
                torch.save(checkpoint, f)
                f.flush()
            model.create_file(checkpoint_file, name=model_name)
    else:
        mmcv.mkdir_or_exist(osp.dirname(filename))
        # immediately flush buffer
        with open(filename, 'wb') as f:
            torch.save(checkpoint, f)
            f.flush()


@RUNNERS.register_module()
class DynamicIterBasedRunnerMod(DynamicIterBasedRunner):
    
    def __init__(self, *args, ckpt_trainable_only=False, **kwargs):
        super(DynamicIterBasedRunnerMod, self).__init__(*args, **kwargs)
        self.ckpt_trainable_only = ckpt_trainable_only

    def save_checkpoint(self,
                        out_dir,
                        filename_tmpl='iter_{}.pth',
                        meta=None,
                        save_optimizer=True,
                        create_symlink=True):
        if meta is None:
            meta = dict(iter=self.iter + 1, epoch=self.epoch + 1)
        elif isinstance(meta, dict):
            meta.update(iter=self.iter + 1, epoch=self.epoch + 1)
        else:
            raise TypeError(
                f'meta should be a dict or None, but got {type(meta)}')
        if self.meta is not None:
            meta.update(self.meta)

        filename = filename_tmpl.format(self.iter + 1)
        filepath = osp.join(out_dir, filename)
        optimizer = self.optimizer if save_optimizer else None
        _loss_scaler = self.loss_scaler if self.with_fp16_grad_scaler else None
        save_checkpoint(
            self.model,
            filepath,
            optimizer=optimizer,
            loss_scaler=_loss_scaler,
            save_apex_amp=self.use_apex_amp,
            meta=meta,
            trainable_only=self.ckpt_trainable_only)
        # in some environments, `os.symlink` is not supported, you may need to
        # set `create_symlink` to False
        if create_symlink:
            dst_file = osp.join(out_dir, 'latest.pth')
            if platform.system() != 'Windows':
                mmcv.symlink(filename, dst_file)
            else:
                shutil.copy(filepath, dst_file)

    def resume(self,
               checkpoint,
               resume_optimizer=True,
               resume_loss_scaler=True,
               map_location='default'):
        if map_location == 'default':
            device_id = torch.cuda.current_device()
            checkpoint = self.load_checkpoint(
                checkpoint,
                map_location=lambda storage, loc: storage.cuda(device_id))
        else:
            checkpoint = self.load_checkpoint(
                checkpoint, map_location=map_location)

        self._epoch = checkpoint['meta']['epoch']
        self._iter = checkpoint['meta']['iter']
        self._inner_iter = checkpoint['meta']['iter']
        if 'optimizer' in checkpoint and resume_optimizer:
            if isinstance(self.optimizer, Optimizer):
                self.optimizer.load_state_dict(checkpoint['optimizer'])
            elif isinstance(self.optimizer, dict):
                for k in self.optimizer.keys():
                    self.optimizer[k].load_state_dict(
                        checkpoint['optimizer'][k])
            else:
                raise TypeError(
                    'Optimizer should be dict or torch.optim.Optimizer '
                    f'but got {type(self.optimizer)}')

        if 'loss_scaler' in checkpoint and resume_loss_scaler and hasattr(self, 'loss_scaler'):
            self.loss_scaler.load_state_dict(checkpoint['loss_scaler'])

        if self.use_apex_amp:
            from apex import amp
            amp.load_state_dict(checkpoint['amp'])

        self.logger.info(f'resumed from epoch: {self.epoch}, iter {self.iter}')
