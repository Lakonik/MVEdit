import logging
import re
import torch

from collections import OrderedDict
from typing import Callable, Optional, Union
from mmcv.runner import CheckpointLoader, get_dist_info, load_state_dict, _load_checkpoint
from mmcv.utils import load_url
from lib.core import rgetattr


@CheckpointLoader.register_scheme(prefixes='huggingface://')
def load_from_huggingface(filename, map_location=None):
    filename = filename.replace('huggingface://', '').split('/')
    outname = '_'.join(filename)
    filename.insert(2, 'resolve/main')
    url = 'https://huggingface.co/' + '/'.join(filename)
    rank, world_size = get_dist_info()
    if rank == 0:
        checkpoint = load_url(
            url, map_location=map_location, file_name=outname)
    if world_size > 1:
        torch.distributed.barrier()
        if rank > 0:
            checkpoint = load_url(
                url, map_location=map_location, file_name=outname)
    return checkpoint


def load_checkpoint(
        model: torch.nn.Module,
        filename: str,
        map_location: Union[str, Callable, None] = None,
        strict: bool = False,
        logger: Optional[logging.Logger] = None,
        revise_keys: list = [(r'^module\.', '')],
        convert_dtype: bool = False) -> Union[dict, OrderedDict]:
    checkpoint = _load_checkpoint(filename, map_location, logger)
    # OrderedDict is a subclass of dict
    if not isinstance(checkpoint, dict):
        raise RuntimeError(
            f'No state_dict found in checkpoint file {filename}')
    # get state_dict from checkpoint
    if 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        state_dict = checkpoint

    # strip prefix of state_dict
    metadata = getattr(state_dict, '_metadata', OrderedDict())
    for p, r in revise_keys:
        state_dict = OrderedDict(
            {re.sub(p, r, k): v
             for k, v in state_dict.items()})
    # Keep metadata in state_dict
    state_dict._metadata = metadata

    # load state_dict
    if convert_dtype:  # use state_dict dtype
        for key, value in state_dict.items():
            param = rgetattr(model, key, None)
            if param is not None:
                param.data = value.data.to(device=param.device)
    else:  # use model dtype
        load_state_dict(model, state_dict, strict, logger)
    return checkpoint
