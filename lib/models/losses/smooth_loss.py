import torch.nn as nn
import torchvision.transforms.functional as F
from mmgen.models.builder import MODULES

from .pixelwise_loss import l1_loss_mod


@MODULES.register_module()
class SmoothLoss(nn.Module):

    def __init__(self,
                 kernel_size=17,
                 sigma=3,
                 loss_weight=1.0):
        super().__init__()
        self.loss_weight = loss_weight
        self.kernel_size = kernel_size
        self.sigma = sigma

    def forward(self, pred, weight=None, avg_factor=None):
        target = F.gaussian_blur(pred, kernel_size=self.kernel_size, sigma=self.sigma)
        return l1_loss_mod(
            pred, target, weight=weight, avg_factor=avg_factor
        ) * self.loss_weight
