import torch.nn as nn
from mmgen.models.builder import MODULES
from mmgen.models.losses.utils import weighted_loss

from ...core import eval_ssim


@weighted_loss
def ssim_loss(pred, target):
    return 1 - eval_ssim(pred, target, separate_filter=False)[0].flatten()


@MODULES.register_module()
class SSIMLoss(nn.Module):

    def __init__(self,
                 loss_weight=1.0):
        super().__init__()
        self.loss_weight = loss_weight

    def forward(self, pred, target, weight=None, avg_factor=None):
        return ssim_loss(
            pred, target, weight=weight, avg_factor=avg_factor) * self.loss_weight
