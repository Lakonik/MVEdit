import torch
import torch.nn as nn
import lpips
from mmgen.models.builder import MODULES
from mmgen.models.losses.utils import weighted_loss


@weighted_loss
def lpips_loss(pred, target, lpips_fun):
    return lpips_fun(pred, target).flatten()


@MODULES.register_module()
class LPIPSLoss(nn.Module):

    def __init__(self,
                 net='vgg',
                 lpips_list=None,
                 normalize_inputs=True,
                 loss_weight=1.0):
        super().__init__()
        self.net = net
        self.lpips = [] if (lpips_list is None or (len(lpips_list) > 0 and lpips_list[0].pnet_type != net)) \
            else lpips_list  # use a list to avoid registering the LPIPS model in state_dict
        self.normalize_inputs = normalize_inputs
        self.loss_weight = loss_weight

    def forward(self, pred, target, weight=None, avg_factor=None):
        dtype = pred.dtype
        lpips_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        if len(self.lpips) == 0:
            lpips_eval = lpips.LPIPS(
                net=self.net, eval_mode=True, pnet_tune=False).to(
                device=pred.device, dtype=lpips_dtype)
            self.lpips.append(lpips_eval)
        if self.normalize_inputs:
            pred = pred * 2 - 1
            target = target * 2 - 1
        return lpips_loss(
            pred.to(lpips_dtype), target.to(lpips_dtype), lpips_fun=self.lpips[0],
            weight=weight, avg_factor=avg_factor
        ).to(dtype) * self.loss_weight
