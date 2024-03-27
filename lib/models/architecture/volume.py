import torch
import torch.nn as nn
import torch.nn.functional as F
import spconv.pytorch as spconv

from typing import Any, Dict, List, Optional, Tuple, Union
from mmcv.cnn.utils import constant_init, kaiming_init
from mmgen.models.builder import MODULES
from diffusers.models.attention_processor import Attention, AttnProcessor, AttentionProcessor
from diffusers.models.modeling_utils import ModelMixin
from diffusers.models.activations import get_activation
from diffusers.utils import is_xformers_available

from lib.ops import neighbor_spvolume_linear_interp


class UpsampleVolume(nn.Module):
    def __init__(self, channels, use_conv=False, use_conv_transpose=False, out_channels=None):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.use_conv_transpose = use_conv_transpose
        conv = None
        if use_conv_transpose:
            conv = nn.ConvTranspose3d(channels, self.out_channels, 4, 2, 1)
        elif use_conv:
            conv = nn.Conv3d(channels, self.out_channels, 3, padding=1)
        self.conv = conv

    def forward(self, hidden_states):
        assert hidden_states.shape[1] == self.channels

        if self.use_conv_transpose:
            return self.conv(hidden_states)

        dtype = hidden_states.dtype
        if dtype == torch.bfloat16:
            hidden_states = hidden_states.to(torch.float32)
        # upsample_nearest_nhwc fails with large batch sizes. see https://github.com/huggingface/diffusers/issues/984
        if hidden_states.shape[0] >= 64:
            hidden_states = hidden_states.contiguous()
        hidden_states = F.interpolate(hidden_states, scale_factor=2.0, mode='nearest')
        if dtype == torch.bfloat16:
            hidden_states = hidden_states.to(dtype)

        if self.use_conv:
            hidden_states = self.conv(hidden_states)

        return hidden_states


class DownsampleVolume(nn.Module):
    def __init__(self, channels, use_conv=False, out_channels=None):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        stride = 2
        if use_conv:
            conv = nn.Conv3d(self.channels, self.out_channels, 3, stride=stride, padding=1)
        else:
            assert self.channels == self.out_channels
            conv = nn.AvgPool3d(kernel_size=stride, stride=stride)
        self.conv = conv

    def forward(self, hidden_states):
        assert hidden_states.shape[1] == self.channels
        hidden_states = self.conv(hidden_states)
        return hidden_states


class ResnetBlockVolume(nn.Module):
    def __init__(
            self,
            in_channels,
            out_channels=None,
            conv_shortcut=False,
            dropout=0.0,
            groups=32,
            eps=1e-6,
            non_linearity='swish',
            use_in_shortcut=None,
            conv_shortcut_bias: bool = True):
        super().__init__()
        self.in_channels = in_channels
        out_channels = in_channels if out_channels is None else out_channels
        self.out_channels = out_channels
        self.use_conv_shortcut = conv_shortcut

        self.norm1 = torch.nn.GroupNorm(num_groups=groups, num_channels=in_channels, eps=eps, affine=True)
        self.conv1 = torch.nn.Conv3d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.norm2 = torch.nn.GroupNorm(num_groups=groups, num_channels=out_channels, eps=eps, affine=True)
        self.dropout = torch.nn.Dropout(dropout)
        self.conv2 = torch.nn.Conv3d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)

        self.nonlinearity = get_activation(non_linearity)

        self.use_in_shortcut = self.in_channels != out_channels if use_in_shortcut is None else use_in_shortcut

        self.conv_shortcut = None
        if self.use_in_shortcut:
            self.conv_shortcut = torch.nn.Conv3d(
                in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=conv_shortcut_bias
            )

    def forward(self, input_tensor):
        hidden_states = input_tensor
        hidden_states = self.norm1(hidden_states)
        hidden_states = self.nonlinearity(hidden_states)
        hidden_states = self.conv1(hidden_states)
        hidden_states = self.norm2(hidden_states)
        hidden_states = self.nonlinearity(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.conv2(hidden_states)

        if self.conv_shortcut is not None:
            input_tensor = self.conv_shortcut(input_tensor)

        output_tensor = input_tensor + hidden_states

        return output_tensor


class DownBlockVolume(nn.Module):
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            dropout: float = 0.0,
            num_layers: int = 1,
            resnet_eps: float = 1e-6,
            resnet_act_fn: str = 'swish',
            resnet_groups: int = 32,
            add_downsample=True):
        super().__init__()
        resnets = []

        for i in range(num_layers):
            in_channels = in_channels if i == 0 else out_channels
            resnets.append(
                ResnetBlockVolume(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    eps=resnet_eps,
                    groups=resnet_groups,
                    dropout=dropout,
                    non_linearity=resnet_act_fn))

        self.resnets = nn.ModuleList(resnets)

        if add_downsample:
            self.downsamplers = nn.ModuleList([DownsampleVolume(
                out_channels, use_conv=True, out_channels=out_channels)])
        else:
            self.downsamplers = None

    def forward(self, hidden_states):
        output_states = ()

        for resnet in self.resnets:
            hidden_states = resnet(hidden_states)
            output_states += (hidden_states,)

        if self.downsamplers is not None:
            for downsampler in self.downsamplers:
                hidden_states = downsampler(hidden_states)
            output_states += (hidden_states,)

        return hidden_states, output_states


class UNetMidBlockVolume(nn.Module):
    def __init__(
            self,
            in_channels: int,
            dropout: float = 0.0,
            num_layers: int = 1,
            resnet_eps: float = 1e-6,
            resnet_act_fn: str = 'swish',
            resnet_groups: int = 32,
            add_attention: bool = True,
            attn_num_head_channels=1):
        super().__init__()
        resnet_groups = resnet_groups if resnet_groups is not None else min(in_channels // 4, 32)
        self.add_attention = add_attention

        # there is always at least one resnet
        resnets = [
            ResnetBlockVolume(
                in_channels=in_channels,
                out_channels=in_channels,
                eps=resnet_eps,
                groups=resnet_groups,
                dropout=dropout,
                non_linearity=resnet_act_fn)]
        attentions = []

        for _ in range(num_layers):
            if self.add_attention:
                attentions.append(
                    Attention(
                        in_channels,
                        heads=in_channels // attn_num_head_channels if attn_num_head_channels is not None else 1,
                        dim_head=attn_num_head_channels if attn_num_head_channels is not None else in_channels,
                        eps=resnet_eps,
                        norm_num_groups=resnet_groups,
                        residual_connection=True,
                        bias=True,
                        upcast_softmax=True))
            else:
                attentions.append(None)

            resnets.append(
                ResnetBlockVolume(
                    in_channels=in_channels,
                    out_channels=in_channels,
                    eps=resnet_eps,
                    groups=resnet_groups,
                    dropout=dropout,
                    non_linearity=resnet_act_fn))

        self.attentions = nn.ModuleList(attentions)
        self.resnets = nn.ModuleList(resnets)

    def forward(self, hidden_states):
        hidden_states = self.resnets[0](hidden_states)
        for attn, resnet in zip(self.attentions, self.resnets[1:]):
            if attn is not None:
                hidden_states = attn(hidden_states)
            hidden_states = resnet(hidden_states)

        return hidden_states


class UpBlockVolume(nn.Module):
    def __init__(
            self,
            in_channels: int,
            prev_output_channel: int,
            out_channels: int,
            dropout: float = 0.0,
            num_layers: int = 1,
            resnet_eps: float = 1e-6,
            resnet_act_fn: str = 'swish',
            resnet_groups: int = 32,
            add_upsample=True):
        super().__init__()
        resnets = []

        for i in range(num_layers):
            res_skip_channels = in_channels if (i == num_layers - 1) else out_channels
            resnet_in_channels = prev_output_channel if i == 0 else out_channels

            resnets.append(
                ResnetBlockVolume(
                    in_channels=resnet_in_channels + res_skip_channels,
                    out_channels=out_channels,
                    eps=resnet_eps,
                    groups=resnet_groups,
                    dropout=dropout,
                    non_linearity=resnet_act_fn))

        self.resnets = nn.ModuleList(resnets)

        if add_upsample:
            self.upsamplers = nn.ModuleList([UpsampleVolume(out_channels, use_conv=True, out_channels=out_channels)])
        else:
            self.upsamplers = None

    def forward(self, hidden_states, res_hidden_states_tuple):
        for resnet in self.resnets:
            res_hidden_states = res_hidden_states_tuple[-1]
            res_hidden_states_tuple = res_hidden_states_tuple[:-1]
            hidden_states = torch.cat([hidden_states, res_hidden_states], dim=1)

            hidden_states = resnet(hidden_states)

        if self.upsamplers is not None:
            for upsampler in self.upsamplers:
                hidden_states = upsampler(hidden_states)

        return hidden_states


@MODULES.register_module()
class UNetVolume(ModelMixin):
    def __init__(
        self,
            sample_size: Optional[int] = None,
            in_channels: int = 4,
            out_channels: Optional[int] = None,
            block_out_channels: Tuple[int] = (320, 640, 1280, 1280),
            layers_per_block: Union[int, Tuple[int]] = 2,
            encoder_block_out_channels: Optional[Tuple[int]] = None,
            encoder_layers_per_block: Optional[Union[int, Tuple[int]]] = 2,
            act_fn: str = 'silu',
            norm_num_groups: Optional[int] = 32,
            norm_eps: float = 1e-5,
            attention_head_dim: Union[int, Tuple[int]] = 8,
            conv_in_kernel: int = 3,
            conv_out_kernel: int = 3,
            zero_init_residual: bool = True):
        super().__init__()

        self.sample_size = sample_size

        conv_in_padding = (conv_in_kernel - 1) // 2
        self.conv_in = nn.Conv3d(
            in_channels,
            block_out_channels[0] if encoder_block_out_channels is None else encoder_block_out_channels[0],
            kernel_size=conv_in_kernel, padding=conv_in_padding
        )

        self.encoder_blocks = nn.ModuleList([])
        if encoder_block_out_channels is not None:
            output_channel = encoder_block_out_channels[0]
            for i in range(len(encoder_block_out_channels)):
                input_channel = output_channel
                output_channel = encoder_block_out_channels[i]
                encoder_block = DownBlockVolume(
                    num_layers=encoder_layers_per_block[i],
                    in_channels=input_channel,
                    out_channels=output_channel,
                    add_downsample=True,
                    resnet_eps=norm_eps,
                    resnet_act_fn=act_fn,
                    resnet_groups=norm_num_groups)
                self.encoder_blocks.append(encoder_block)

        self.down_blocks = nn.ModuleList([])
        self.up_blocks = nn.ModuleList([])

        if isinstance(attention_head_dim, int):
            attention_head_dim = (attention_head_dim,) * len(block_out_channels)

        if isinstance(layers_per_block, int):
            layers_per_block = [layers_per_block] * len(block_out_channels)

        output_channel = block_out_channels[0]
        for i in range(len(block_out_channels)):
            input_channel = output_channel
            output_channel = block_out_channels[i]
            is_final_block = i == len(block_out_channels) - 1

            down_block = DownBlockVolume(
                num_layers=layers_per_block[i],
                in_channels=input_channel,
                out_channels=output_channel,
                add_downsample=not is_final_block,
                resnet_eps=norm_eps,
                resnet_act_fn=act_fn,
                resnet_groups=norm_num_groups)
            self.down_blocks.append(down_block)

        self.mid_block = UNetMidBlockVolume(
            in_channels=block_out_channels[-1],
            resnet_eps=norm_eps,
            resnet_act_fn=act_fn,
            attn_num_head_channels=attention_head_dim[-1],
            resnet_groups=norm_num_groups)

        reversed_block_out_channels = list(reversed(block_out_channels))
        reversed_layers_per_block = list(reversed(layers_per_block))

        output_channel = reversed_block_out_channels[0]
        for i in range(len(block_out_channels)):
            is_final_block = i == len(block_out_channels) - 1

            prev_output_channel = output_channel
            output_channel = reversed_block_out_channels[i]
            input_channel = reversed_block_out_channels[min(i + 1, len(block_out_channels) - 1)]

            if not is_final_block:
                add_upsample = True
            else:
                add_upsample = False

            up_block = UpBlockVolume(
                num_layers=reversed_layers_per_block[i] + 1,
                in_channels=input_channel,
                out_channels=output_channel,
                prev_output_channel=prev_output_channel,
                add_upsample=add_upsample,
                resnet_eps=norm_eps,
                resnet_act_fn=act_fn,
                resnet_groups=norm_num_groups)
            self.up_blocks.append(up_block)

        if norm_num_groups is not None:
            self.conv_norm_out = nn.GroupNorm(
                num_channels=block_out_channels[0], num_groups=norm_num_groups, eps=norm_eps
            )
            self.conv_act = get_activation(act_fn)
        else:
            self.conv_norm_out = None
            self.conv_act = None

        conv_out_padding = (conv_out_kernel - 1) // 2

        if out_channels is not None:
            self.conv_out = nn.Conv3d(
                block_out_channels[0], out_channels, kernel_size=conv_out_kernel, padding=conv_out_padding
            )
        else:
            self.conv_out = None

        if is_xformers_available():
            self.enable_xformers_memory_efficient_attention()

        self.zero_init_residual = zero_init_residual
        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                kaiming_init(m)
            elif isinstance(m, nn.GroupNorm):
                constant_init(m, 1)

        if self.zero_init_residual:
            for m in self.modules():
                if isinstance(m, ResnetBlockVolume):
                    constant_init(m.conv2, 0)
                elif isinstance(m, Attention):
                    constant_init(m.to_out[0], 0)

    @property
    def attn_processors(self):
        processors = {}

        def fn_recursive_add_processors(name: str, module: torch.nn.Module, processors: Dict[str, AttentionProcessor]):
            if hasattr(module, "set_processor"):
                processors[f"{name}.processor"] = module.processor

            for sub_name, child in module.named_children():
                fn_recursive_add_processors(f"{name}.{sub_name}", child, processors)

            return processors

        for name, module in self.named_children():
            fn_recursive_add_processors(name, module, processors)

        return processors

    def set_attn_processor(self, processor):
        count = len(self.attn_processors.keys())

        if isinstance(processor, dict) and len(processor) != count:
            raise ValueError(
                f"A dict of processors was passed, but the number of processors {len(processor)} does not match the"
                f" number of attention layers: {count}. Please make sure to pass {count} processor classes."
            )

        def fn_recursive_attn_processor(name: str, module: torch.nn.Module, processor):
            if hasattr(module, "set_processor"):
                if not isinstance(processor, dict):
                    module.set_processor(processor)
                else:
                    module.set_processor(processor.pop(f"{name}.processor"))

            for sub_name, child in module.named_children():
                fn_recursive_attn_processor(f"{name}.{sub_name}", child, processor)

        for name, module in self.named_children():
            fn_recursive_attn_processor(name, module, processor)

    def set_default_attn_processor(self):
        self.set_attn_processor(AttnProcessor())

    def forward(self, sample):
        sample = self.conv_in(sample)

        extra_down_block_res_samples = (sample,)
        for encoder_block in self.encoder_blocks:
            sample, res_samples = encoder_block(sample)
            extra_down_block_res_samples += res_samples

        down_block_res_samples = (sample,)
        for downsample_block in self.down_blocks:
            sample, res_samples = downsample_block(sample)
            down_block_res_samples += res_samples

        sample = self.mid_block(sample)

        for i, upsample_block in enumerate(self.up_blocks):
            res_samples = down_block_res_samples[-len(upsample_block.resnets):]
            down_block_res_samples = down_block_res_samples[:-len(upsample_block.resnets)]
            sample = upsample_block(sample, res_samples)

        if self.conv_norm_out:
            sample = self.conv_norm_out(sample)
            sample = self.conv_act(sample)
        if self.conv_out:
            sample = self.conv_out(sample)

        return sample, extra_down_block_res_samples


class SpGroupNorm(spconv.SparseModule):
    def __init__(self, num_groups: int, num_channels: int, eps: float = 1e-5, affine: bool = True,
                 device=None, dtype=None) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super(SpGroupNorm, self).__init__()
        if num_channels % num_groups != 0:
            raise ValueError('num_channels must be divisible by num_groups')

        self.num_groups = num_groups
        self.num_channels = num_channels
        self.eps = eps
        self.affine = affine
        if self.affine:
            self.weight = nn.Parameter(torch.empty(num_channels, **factory_kwargs))
            self.bias = nn.Parameter(torch.empty(num_channels, **factory_kwargs))
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        if self.affine:
            nn.init.ones_(self.weight)
            nn.init.zeros_(self.bias)

    def forward(self, input):
        if input.batch_size > 1:
            feats = input.features.split(input.voxel_num, dim=0)
            out_feats = torch.cat(
                [F.group_norm(feats_single.T[None], self.num_groups, self.weight, self.bias, self.eps).squeeze(0).T
                 for feats_single in feats],
                dim=0)
        else:
            out_feats = F.group_norm(
                input.features.T[None], self.num_groups, self.weight, self.bias, self.eps).squeeze(0).T
        return input.replace_feature(out_feats)


class ResnetBlockSpVolume(spconv.SparseModule):
    def __init__(
            self,
            in_channels,
            out_channels=None,
            conv_shortcut=False,
            dropout=0.0,
            groups=32,
            eps=1e-6,
            non_linearity='swish',
            use_in_shortcut=None,
            conv_shortcut_bias: bool = True,
            indice_key=None,
            algo=None):
        super().__init__()
        self.in_channels = in_channels
        out_channels = in_channels if out_channels is None else out_channels
        self.out_channels = out_channels
        self.use_conv_shortcut = conv_shortcut

        self.norm1 = SpGroupNorm(num_groups=groups, num_channels=in_channels, eps=eps, affine=True)
        self.conv1 = spconv.SubMConv3d(
            in_channels, out_channels, kernel_size=3, stride=1, padding=1,
            indice_key=indice_key, algo=algo)
        self.norm2 = SpGroupNorm(num_groups=groups, num_channels=out_channels, eps=eps, affine=True)
        self.dropout = torch.nn.Dropout(dropout)
        self.conv2 = spconv.SubMConv3d(
            out_channels, out_channels, kernel_size=3, stride=1, padding=1,
            indice_key=indice_key, algo=algo)

        self.nonlinearity = get_activation(non_linearity)

        self.use_in_shortcut = self.in_channels != out_channels if use_in_shortcut is None else use_in_shortcut

        self.conv_shortcut = None
        if self.use_in_shortcut:
            self.conv_shortcut = spconv.SubMConv3d(
                in_channels, out_channels, kernel_size=1, stride=1, padding=0,
                bias=conv_shortcut_bias, indice_key=indice_key, algo=algo)

    def forward(self, input_tensor):
        hidden_states = input_tensor

        hidden_states = self.norm1(hidden_states)
        hidden_states = hidden_states.replace_feature(
            self.nonlinearity(hidden_states.features))
        hidden_states = self.conv1(hidden_states)
        hidden_states = self.norm2(hidden_states)
        hidden_states = input_tensor.replace_feature(
            self.dropout(self.nonlinearity(hidden_states.features)))
        hidden_states = self.conv2(hidden_states)

        if self.conv_shortcut is not None:
            input_tensor = self.conv_shortcut(input_tensor)

        output_tensor = input_tensor + hidden_states

        return output_tensor


class UpsampleSpVolume(spconv.SparseModule):
    def __init__(self, channels, use_conv=False, out_channels=None, indice_key=None):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        conv = None
        if use_conv:
            conv = spconv.SubMConv3d(channels, self.out_channels, 3, padding=1, indice_key=indice_key)
        self.conv = conv

    def forward(self, hidden_states, indices=None, voxel_num=None):
        """
        Args:
            hidden_states (spconv.SparseConvTensor): Spatial shape (Di, Hi, Wi)
            indices (torch.Tensor | None): (num_pts, 4) in [batch_idx, x_Do, y_Ho, z_Wo]
        """
        if indices is None:
            # Todo: upsample within the original manifold
            raise NotImplementedError
        else:
            batch_ids = indices[:, :1]
            out_spatial_shape = [s * 2 for s in hidden_states.spatial_shape]
            out_spatial_shape_t = hidden_states.features.new_tensor(out_spatial_shape)
            pts = indices[:, 1:].to(hidden_states.features.dtype) * (2 / out_spatial_shape_t) \
                  + (1 / out_spatial_shape_t - 1)
            out_feats, valid_pts_mask = neighbor_spvolume_linear_interp(hidden_states, pts, batch_ids, prune=True)
            hidden_states = spconv.SparseConvTensor(
                features=out_feats, indices=indices[valid_pts_mask], spatial_shape=out_spatial_shape,
                batch_size=hidden_states.batch_size, voxel_num=voxel_num)

        if self.use_conv:
            hidden_states = self.conv(hidden_states)

        return hidden_states


class UpBlockSpVolume(spconv.SparseModule):
    def __init__(
            self,
            in_channels: int,
            prev_output_channel: int,
            out_channels: int,
            dropout: float = 0.0,
            num_layers: int = 1,
            resnet_eps: float = 1e-6,
            resnet_act_fn: str = 'swish',
            resnet_groups: int = 32,
            add_upsample=True,
            indice_key=None):
        super().__init__()
        resnets = []

        for i in range(num_layers):
            res_skip_channels = in_channels if (i == num_layers - 1) else out_channels
            resnet_in_channels = prev_output_channel if i == 0 else out_channels

            resnets.append(
                ResnetBlockSpVolume(
                    in_channels=resnet_in_channels + res_skip_channels,
                    out_channels=out_channels,
                    eps=resnet_eps,
                    groups=resnet_groups,
                    dropout=dropout,
                    non_linearity=resnet_act_fn,
                    indice_key=indice_key))

        self.resnets = nn.ModuleList(resnets)

        if add_upsample:
            self.upsamplers = nn.ModuleList([UpsampleSpVolume(
                prev_output_channel, use_conv=True, out_channels=prev_output_channel, indice_key=indice_key)])
        else:
            self.upsamplers = None

    def forward(self, hidden_states, res_hidden_states_tuple, indices=None, voxel_num=None):
        """
        Args:
            hidden_states (spconv.SparseConvTensor | torch.Tensor)
        """
        if self.upsamplers is not None:
            for upsampler in self.upsamplers:
                hidden_states = upsampler(hidden_states, indices, voxel_num)

        for resnet in self.resnets:
            res_hidden_states = res_hidden_states_tuple[-1]
            res_hidden_states_tuple = res_hidden_states_tuple[:-1]
            if isinstance(res_hidden_states, spconv.SparseConvTensor):
                hidden_states = hidden_states.replace_feature(
                    torch.cat([hidden_states.features, res_hidden_states.features], dim=1))
            else:
                if res_hidden_states.dim() == 5:  # dense feats
                    res_hidden_states = res_hidden_states[
                        indices[:, 0], :, indices[:, 1], indices[:, 2], indices[:, 3]]
                hidden_states = hidden_states.replace_feature(
                    torch.cat([hidden_states.features, res_hidden_states], dim=1))

            hidden_states = resnet(hidden_states)

        return hidden_states
