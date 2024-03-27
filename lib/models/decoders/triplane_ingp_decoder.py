import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import tinycudann as tcnn
except ImportError:
    tcnn = None
import numpy as np

from copy import deepcopy
from mmcv.cnn import constant_init
from mmgen.models.builder import MODULES

from .triplane_decoder import TriPlaneDecoder
from lib.ops import SHEncoder


@MODULES.register_module()
class TriPlaneiNGPDecoder(TriPlaneDecoder):

    def __init__(self,
                 *args,
                 plane_cfg=['xy', 'xz', 'yz'],
                 interp_mode='bilinear',
                 base_layers=[3 * 32, 128],
                 density_layers=[128, 1],
                 color_layers=[128, 128, 3],
                 base_resolution=16,
                 max_resolution=320,
                 n_levels=12,
                 use_dir_enc=True,
                 dir_layers=None,
                 scene_base_size=None,
                 scene_rand_dims=(0, 1),
                 activation='silu',
                 sigma_activation='trunc_exp',
                 sigmoid_saturation=0.001,
                 code_dropout=0.0,
                 flip_z=False,
                 ingp_base_layers=1,
                 zero_init_ingp=True,
                 **kwargs):
        super(TriPlaneDecoder, self).__init__(*args, **kwargs)
        self.plane_cfg = plane_cfg
        self.interp_mode = interp_mode
        self.in_chn = base_layers[0]
        self.use_dir_enc = use_dir_enc
        if scene_base_size is None:
            self.scene_base = None
        else:
            rand_size = [1 for _ in scene_base_size]
            for dim in scene_rand_dims:
                rand_size[dim] = scene_base_size[dim]
            init_base = torch.randn(rand_size).expand(scene_base_size).clone()
            self.scene_base = nn.Parameter(init_base)
        self.dir_encoder = SHEncoder() if use_dir_enc else None
        self.sigmoid_saturation = sigmoid_saturation

        activation_layer = self.activation_dict[activation.lower()]

        base_net = []
        for i in range(len(base_layers) - 1):
            base_net.append(nn.Linear(base_layers[i], base_layers[i + 1]))
            if i != len(base_layers) - 2:
                base_net.append(activation_layer())
        self.base_net = nn.Sequential(*base_net)
        self.base_activation = activation_layer()

        density_net = []
        for i in range(len(density_layers) - 1):
            density_net.append(nn.Linear(density_layers[i], density_layers[i + 1]))
            if i != len(density_layers) - 2:
                density_net.append(activation_layer())
        density_net.append(self.activation_dict[sigma_activation.lower()]())
        self.density_net = nn.Sequential(*density_net)

        self.dir_net = None
        color_net = []
        if use_dir_enc:
            if dir_layers is not None:
                dir_net = []
                for i in range(len(dir_layers) - 1):
                    dir_net.append(nn.Linear(dir_layers[i], dir_layers[i + 1]))
                    if i != len(dir_layers) - 2:
                        dir_net.append(activation_layer())
                self.dir_net = nn.Sequential(*dir_net)
            else:
                color_layers[0] = color_layers[0] + 16  # sh_encoding
        for i in range(len(color_layers) - 1):
            color_net.append(nn.Linear(color_layers[i], color_layers[i + 1]))
            if i != len(color_layers) - 2:
                color_net.append(activation_layer())
        color_net.append(nn.Sigmoid())
        self.color_net = nn.Sequential(*color_net)

        self.code_dropout = nn.Dropout2d(code_dropout) if code_dropout > 0 else None
        self.flip_z = flip_z

        self.zero_init_ingp = zero_init_ingp

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
            dtype=torch.float32,
        )
        hidden_dim = base_layers[-1]
        activation_layer = self.activation_dict[activation.lower()]

        ingp_base_net = [nn.Linear(self.encoder.n_output_dims, hidden_dim)]
        for i in range(ingp_base_layers - 1):
            ingp_base_net.append(activation_layer())
            ingp_base_net.append(nn.Linear(hidden_dim, hidden_dim))
        self.ingp_base_net = nn.Sequential(*ingp_base_net)

        self.state_dict_bak = None

        self.init_weights()

    def init_weights(self):
        super().init_weights()
        self.encoder.params.data.uniform_(-1e-4, 1e-4)
        if self.zero_init_ingp:
            constant_init(self.ingp_base_net[-1], 0)

    def backup_state_dict(self):
        self.state_dict_bak = deepcopy(self.state_dict())

    def restore_state_dict(self):
        if self.state_dict_bak is None:
            raise RuntimeError("No backup state dict found")
        self.load_state_dict(self.state_dict_bak)

    def point_decode(self, xyzs, dirs, code, density_only=False, use_2nd_order=False):
        """
        Args:
            xyzs: Shape (num_scenes, (num_points_per_scene, 3))
            dirs: Shape (num_scenes, (num_points_per_scene, 3))
            code: Shape (num_scenes, 3, n_channels, h, w)
        """
        assert len(xyzs) == 1, "Multiple scenes not implemented"
        ingp_enc = self.encoder((xyzs[0] + self.bound) / (2 * self.bound)).float()

        num_scenes, _, n_channels, h, w = code.size()
        if self.code_dropout is not None:
            code = self.code_dropout(
                code.reshape(num_scenes * 3, n_channels, h, w)
            ).reshape(num_scenes, 3, n_channels, h, w)
        if self.scene_base is not None:
            code = code + self.scene_base
        dtype = code.dtype
        if use_2nd_order:
            from lib.ops.cuda_gridsample import grid_sample_2d as grid_sample
        else:
            grid_sample = F.grid_sample
        if isinstance(xyzs, torch.Tensor):
            assert xyzs.dim() == 3
            num_points = xyzs.size(-2)
            point_code = grid_sample(
                code.reshape(num_scenes * 3, -1, h, w).float(),
                self.xyz_transform(xyzs),
                mode=self.interp_mode, padding_mode='border', align_corners=False
            ).reshape(num_scenes, 3, n_channels, num_points)
            point_code = point_code.to(dtype)
            point_code = point_code.permute(0, 3, 2, 1).reshape(
                num_scenes * num_points, n_channels * 3)
            num_points = [num_points] * num_scenes
        else:
            num_points = []
            point_code = []
            for code_single, xyzs_single in zip(code.float(), xyzs):
                num_points_per_scene = xyzs_single.size(-2)
                # (3, code_chn, num_points_per_scene)
                point_code_single = grid_sample(
                    code_single,
                    self.xyz_transform(xyzs_single),
                    mode=self.interp_mode, padding_mode='border', align_corners=False
                ).squeeze(-2)
                point_code_single = point_code_single.to(dtype)
                point_code_single = point_code_single.permute(2, 1, 0).reshape(
                    num_points_per_scene, n_channels * 3)
                num_points.append(num_points_per_scene)
                point_code.append(point_code_single)
            point_code = torch.cat(point_code, dim=0) if len(point_code) > 1 \
                else point_code[0]
        base_x = self.base_net(point_code) + self.ingp_base_net(ingp_enc)
        base_x_act = self.base_activation(base_x)
        sigmas = self.density_net(base_x_act).squeeze(-1)
        if density_only:
            rgbs = None
        else:
            if self.use_dir_enc:
                dirs = torch.cat(dirs, dim=0) if num_scenes > 1 else dirs[0]
                sh_enc = self.dir_encoder(dirs).to(base_x.dtype)
                if self.dir_net is not None:
                    color_in = self.base_activation(base_x + self.dir_net(sh_enc))
                else:
                    color_in = torch.cat([base_x_act, sh_enc], dim=-1)
            else:
                color_in = base_x_act
            rgbs = self.color_net(color_in)
            if self.sigmoid_saturation > 0:
                rgbs = rgbs * (1 + self.sigmoid_saturation * 2) - self.sigmoid_saturation
        return sigmas, rgbs, num_points
