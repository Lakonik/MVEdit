import logging
# import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.transforms.functional as F_t

from typing import List, Union
from mmcv.runner import load_checkpoint
from mmgen.utils import get_root_logger
from ..architecture.tracerb7.tracer import TracerDecoder
from ..architecture.tracerb7.efficientnet import EfficientEncoderB7


class TracerUniversalB7(nn.Module):

    def __init__(
            self,
            input_image_size: Union[List[int], int] = 640,
            batch_size: int = 8,
            torch_dtype='bfloat16',
            pretrained='huggingface://Carve/tracer_b7/tracer_b7.pth',
            erosion: int = 1):
        super(TracerUniversalB7, self).__init__()
        self.model = TracerDecoder(
            encoder=EfficientEncoderB7(),
            rfb_channel=[32, 64, 128],
            features_channels=[48, 80, 224, 640])
        self.batch_size = batch_size
        if isinstance(input_image_size, list):
            self.input_image_size = input_image_size[:2]
        else:
            self.input_image_size = (input_image_size, input_image_size)
        self.init_weights(pretrained)
        self.dtype = getattr(torch, torch_dtype) if isinstance(torch_dtype, str) else torch_dtype
        self.to(self.dtype)
        self.transform = transforms.Compose(
            [
                transforms.Resize(self.input_image_size, antialias=False),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )
        self.erosion = erosion
        self.eval()

    def init_weights(self, pretrained):
        if pretrained is None:
            raise NotImplementedError
        else:
            logger = get_root_logger(log_level=logging.ERROR)
            load_checkpoint(self.model, pretrained, map_location='cpu', strict=False, logger=logger)

    def forward(self, data):
        """
        Args:
            data (torch.Tensor): input images, shape (N, 3, H, W)
        """
        with torch.no_grad():
            ori_size = data.shape[-2:]
            masks = []
            for image_batch in data.to(self.dtype).split(self.batch_size, dim=0):
                image_batch = self.transform(image_batch)
                mask_batch = self.model(image_batch)
                mask_batch = -F.max_pool2d(-mask_batch, self.erosion * 2 + 1, stride=1, padding=self.erosion)
                masks.append(F_t.resize(mask_batch, ori_size, antialias=False))
            masks = torch.cat(masks, dim=0) if len(masks) > 1 else masks[0]
            failure_samples = (masks > 0.2).flatten(1).all(dim=1)
            failure_threshold = masks < 0.8
            masks.masked_fill_(failure_samples[:, None, None, None] & failure_threshold, 0)
            return masks

    # def rgb_to_rgba_np(self, rgb: np.ndarray) -> np.ndarray:
    #     """
    #     Args:
    #         rgb (np.ndarray): input images, shape (N, H, W, 3), dtype uint8
    #     """
    #     in_img_torch = torch.from_numpy(rgb.transpose(0, 3, 1, 2).astype(np.float32) / 255)
    #     mask = self()
    #     rgba = np.concatenate([
    #         rgb, (mask.permute(0, 2, 3, 1) * 255).round().to(torch.uint8).cpu().numpy()], axis=-1)
    #     return rgba
