import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import tinycudann as tcnn
except ImportError:
    tcnn = None

import numpy as np

from copy import deepcopy
from mmcv.cnn import xavier_init, constant_init
from mmgen.models.builder import MODULES

from .base_volume_renderer import VolumeRenderer
from lib.ops import TruncExp


class MLP(nn.Module):
    def __init__(self, dim_in, dim_out, dim_hidden, num_layers, bias=True):
        super().__init__()
        self.dim_in = dim_in
        self.dim_out = dim_out
        self.dim_hidden = dim_hidden
        self.num_layers = num_layers

        net = []
        for l in range(num_layers):
            net.append(nn.Linear(self.dim_in if l == 0 else self.dim_hidden,
                                 self.dim_out if l == num_layers - 1 else self.dim_hidden, bias=bias))

        self.net = nn.ModuleList(net)

    def forward(self, x):
        for l in range(self.num_layers):
            x = self.net[l](x)
            if l != self.num_layers - 1:
                x = F.relu(x, inplace=True)
        return x


@MODULES.register_module()
class iNGPDecoder(VolumeRenderer):

    def __init__(self,
                 *args,
                 base_resolution=16,
                 max_resolution=320,
                 n_levels=12,
                 num_layers=2,
                 hidden_dim=64,
                 sigmoid_saturation=0.001,
                 blob_density=1.0,
                 blob_radius=0.2,
                 **kwargs):
        super().__init__(*args, **kwargs)

        self.base_resolution = base_resolution
        self.max_resolution = max_resolution
        self.n_levels = n_levels
        self.encoder = tcnn.Encoding(
            n_input_dims=3,
            encoding_config={
                "otype": "HashGrid",
                "n_levels": n_levels,
                "n_features_per_level": 2,
                "log2_hashmap_size": 19,
                "base_resolution": base_resolution,
                "interpolation": "Smoothstep",
                "per_level_scale": np.exp2(np.log2(max_resolution * self.bound / base_resolution) / (n_levels - 1)),
            },
            dtype=torch.float32,  # ENHANCE: default float16 seems unstable...
        )

        self.in_dim = self.encoder.n_output_dims
        self.mlp = MLP(self.in_dim, 4, hidden_dim, num_layers, bias=True)
        self.sigmoid_saturation = sigmoid_saturation
        self.blob_density = blob_density
        self.blob_radius = blob_radius
        self.sigma_activation = TruncExp()

        self.state_dict_bak = None

        self.init_weights()

    def init_weights(self):
        self.encoder.params.data.uniform_(-1e-4, 1e-4)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                xavier_init(m, distribution='uniform')

    def backup_state_dict(self):
        self.state_dict_bak = deepcopy(self.state_dict())

    def restore_state_dict(self):
        if self.state_dict_bak is None:
            raise RuntimeError("No backup state dict found")
        self.load_state_dict(self.state_dict_bak)

    def density_blob(self, x):
        d = (x ** 2).sum(-1).clamp(min=0.2)
        g = self.blob_density * torch.exp(-d / (2 * self.blob_radius ** 2))
        return g

    def point_decode(self, xyzs, dirs, code, density_only=False, use_2nd_order=False):
        """
        Args:
            xyzs: Shape (num_scenes, (num_points_per_scene, 3))
        """
        assert len(xyzs) == 1, "Multiple scenes not implemented"
        enc = self.encoder((xyzs[0] + self.bound) / (2 * self.bound)).float()
        h = self.mlp(enc)

        sigmas = self.sigma_activation(h[..., 0] + self.density_blob(xyzs[0]))
        rgbs = torch.sigmoid(h[..., 1:])
        if self.sigmoid_saturation > 0:
            rgbs = rgbs * (1 + self.sigmoid_saturation * 2) - self.sigmoid_saturation
        num_points = [len(xyzs[0])]
        return sigmas, rgbs, num_points

    def point_density_decode(self, xyzs, code, **kwargs):
        sigmas, _, num_points = self.point_decode(
            xyzs, None, code, density_only=True, **kwargs)
        return sigmas, num_points
