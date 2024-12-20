import torch

from mmgen.models.builder import MODULES
from mmgen.models.diffusions import UniformTimeStepSampler


@MODULES.register_module()
class UniformTimeStepSamplerMod(UniformTimeStepSampler):
    def __init__(self, num_timesteps, **kwargs):
        super().__init__(num_timesteps)


@MODULES.register_module()
class SNRWeightedTimeStepSampler(UniformTimeStepSampler):
    def __init__(self,
                 num_timesteps, mean, std, mode,
                 power=1, power_2=0, min=-1, max=-1, bias=0, prob_power=0.0, data_noise=0.0):
        super(UniformTimeStepSampler, self).__init__()
        self.num_timesteps = num_timesteps
        self.power = power
        self.power_2 = power_2
        self.data_noise = data_noise

        snr = (mean / std) ** 2
        weight_x = snr ** (power + power_2) / (1 + data_noise * snr) ** power + bias

        if min > 0:
            weight_x = weight_x.clip(min=min)
        if max > 0:
            weight_x = weight_x.clip(max=max)

        assert mode in ['EPS', 'START_X', 'V']
        if mode == 'EPS':
            weight_raw = weight_x * (std / mean) ** 2
        elif mode == 'START_X':
            weight_raw = weight_x
        elif mode == 'V':
            weight_raw = weight_x * (std ** 2)
        else:
            raise AttributeError

        prob = weight_raw ** prob_power
        prob /= prob.sum()

        self.weight = torch.from_numpy(weight_raw / (prob * self.num_timesteps)).to(torch.float)
        self.prob = prob.tolist()
