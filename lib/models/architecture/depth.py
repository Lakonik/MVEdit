import torch
import torch.nn as nn
from mmgen.models.builder import MODULES
from transformers import DPTForDepthEstimation


@MODULES.register_module()
class MiDaS(nn.Module):
    def __init__(self, pretrained, subfolder=None) -> None:
        super().__init__()
        if subfolder is None:
            self.dpt = DPTForDepthEstimation.from_pretrained(pretrained)
        else:
            self.dpt = DPTForDepthEstimation.from_pretrained(pretrained, subfolder='depth_estimator')
    
    def forward(self, image):
        depth_map = self.dpt(image.permute(0, 3, 1, 2)).predicted_depth
        depth_min = torch.amin(depth_map, dim=[1, 2], keepdim=True)
        depth_max = torch.amax(depth_map, dim=[1, 2], keepdim=True)
        depth_map = 2.0 * (depth_map - depth_min) / (depth_max - depth_min + 1e-8) - 1.0
        depth_map = depth_map.unsqueeze(1).repeat(1, 3, 1, 1).permute(0, 2, 3, 1)
        return depth_map
