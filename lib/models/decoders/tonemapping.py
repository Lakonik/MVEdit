import torch
import torch.nn as nn


class Tonemapping(nn.Module):
    def __init__(self,
                 exposure=0.0,
                 contrast=0.953,
                 bias=0.088,
                 sigmoid_gain=0.943,
                 log_gain=0.011,
                 lut_logx_min=-9,
                 lut_logx_max=3,
                 lut_steps=16):
        super().__init__()
        self.exposure = exposure
        self.contrast = contrast
        self.sigmoid_gain = sigmoid_gain
        self.log_gain = log_gain
        self.bias = bias
        self.register_buffer('lut_x', torch.linspace(lut_logx_min, lut_logx_max, lut_steps))
        self.register_buffer('lut_y', self.smooth_forward(self.lut_x))

    def smooth_forward(self, x, input_mode='log'):
        assert input_mode in ['log', 'linear']
        if input_mode == 'linear':
            x = x.clamp(min=1e-6).log2()
        x = (x + self.exposure) * self.contrast
        y = x.sigmoid() * self.sigmoid_gain + x * self.log_gain + self.bias
        return y

    def lut(self, x, input_mode='log'):
        assert input_mode in ['log', 'linear']
        dtype = x.dtype
        x = x.to(self.lut_x.dtype)
        if input_mode == 'linear':
            x = x.clamp(min=1e-6).log2()
        i = torch.bucketize(x, self.lut_x, right=True).clamp(min=1, max=len(self.lut_x) - 1)
        t = (x - self.lut_x[i - 1]) / (self.lut_x[i] - self.lut_x[i - 1])
        y = self.lut_y[i - 1] + (self.lut_y[i] - self.lut_y[i - 1]) * t
        return y.to(dtype)

    def inverse_lut(self, y, output_mode='log'):
        assert output_mode in ['log', 'linear']
        dtype = y.dtype
        y = y.to(self.lut_y.dtype)
        i = torch.bucketize(y, self.lut_y, right=True).clamp(min=1, max=len(self.lut_y) - 1)
        t = (y - self.lut_y[i - 1]) / (self.lut_y[i] - self.lut_y[i - 1])
        x = self.lut_x[i - 1] + (self.lut_x[i] - self.lut_x[i - 1]) * t
        if output_mode == 'linear':
            x = torch.exp2(x)
        return x.to(dtype)


# import matplotlib.pyplot as plt
#
# tm = Tonemapping()
#
# x = torch.linspace(-9, 3, 1000)
# y = tm.smooth_forward(x, input_mode='log')
# plt.plot(x, y)
# plt.ylim(0, 1)
# plt.show()
#
# y = torch.linspace(0, 1, 1000)
# x = tm.inverse_lut(y, output_mode='linear')
# plt.plot(x, y)
# plt.show()
