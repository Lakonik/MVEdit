import torch
from mmcv.runner import CheckpointLoader, get_dist_info
from mmcv.utils import load_url


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
