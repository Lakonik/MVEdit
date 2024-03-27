import torch
import torch.nn as nn
from mmgen.models.builder import MODULES
from mmgen.models.losses.utils import weighted_loss


@weighted_loss
def tv_loss(pred, target, dims, power=1, dense_weight=None):
    if target is not None:
        assert pred.size() == target.size()
    pred_diffs = []
    for dim in dims:
        pad_shape = list(pred.size())
        pad_shape[dim] = 1
        pred_diffs.append(torch.cat([torch.diff(pred, dim=dim), pred.new_zeros(pad_shape)], dim=dim))

    if dense_weight is not None:
        diff_weights = []
        for dim in dims:
            pad_shape = list(dense_weight.size())
            pad_shape[dim] = 1
            diff_weights.append(torch.cat([
                torch.minimum(
                    torch.narrow(dense_weight, dim=dim, start=0, length=dense_weight.size(dim) - 1),
                    torch.narrow(dense_weight, dim=dim, start=1, length=dense_weight.size(dim) - 1)),
                dense_weight.new_zeros(pad_shape)], dim=dim))

    if target is None:
        diff_loss = torch.stack(pred_diffs, dim=0)
    else:
        target_diffs = []
        for dim in dims:
            pad_shape = list(pred.size())
            pad_shape[dim] = 1
            target_diffs.append(torch.cat([torch.diff(target, dim=dim), target.new_zeros(pad_shape)], dim=dim))
        diff_loss = torch.stack(pred_diffs, dim=0) - torch.stack(target_diffs, dim=0)

    if dense_weight is not None:
        diff_loss = diff_loss * torch.stack(diff_weights, dim=0)

    return diff_loss.norm(dim=0).pow(power).mean(dim=dims)


@MODULES.register_module()
class TVLoss(nn.Module):

    def __init__(self,
                 dims=[-2, -1],
                 power=1,
                 loss_weight=1.0):
        super().__init__()
        self.dims = dims
        self.power = power
        self.loss_weight = loss_weight

    def forward(self, pred, target=None, weight=None, avg_factor=None):
        return tv_loss(
            pred, target, self.dims, power=self.power,
            dense_weight=weight, avg_factor=avg_factor
        ) * self.loss_weight
