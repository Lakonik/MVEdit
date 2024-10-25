import os
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Dict, Any, Optional, Union, Tuple
from collections import OrderedDict
from copy import deepcopy
from transformers import CLIPTextModel as _CLIPTextModel
from diffusers.models import AutoencoderKL
from diffusers.models.unet_2d_condition import UNet2DConditionModel as _UNet2DConditionModel
from diffusers.models import ControlNetModel
from diffusers.models.attention_processor import Attention, LoRALinearLayer, LoRAXFormersAttnProcessor, LoRAAttnProcessor, LoRAAttnProcessor2_0, AttnProcessor, AttnProcessor2_0, XFormersAttnProcessor
try:
    from diffusers.models.vae import Decoder
except ModuleNotFoundError:
    from diffusers.models.autoencoders.vae import Decoder
from diffusers.models.resnet import ResnetBlock2D
from diffusers.models.modeling_utils import ModelMixin
from diffusers.utils.import_utils import is_xformers_available

from mmcv.runner import load_checkpoint, _load_checkpoint, load_state_dict
from mmcv.cnn.utils import constant_init, kaiming_init
from mmgen.models.builder import MODULES, build_module
from mmgen.utils import get_root_logger

from ...core import rgetattr


TEXT_ENCODER_ATTN_MODULE = '.self_attn'


def ceildiv(a, b):
    return -(a // -b)


MODULES.register_module(name='LoRAXFormersAttnProcessor', module=LoRAXFormersAttnProcessor)
MODULES.register_module(name='LoRAAttnProcessor', module=LoRAAttnProcessor)
MODULES.register_module(name='LoRAAttnProcessor2_0', module=LoRAAttnProcessor2_0)


def autocast_patch(module, dtype=None, enabled=True):

    def make_new_forward(old_forward, dtype, enabled):
        def new_forward(*args, **kwargs):
            with torch.autocast(device_type='cuda', dtype=dtype, enabled=enabled):
                result = old_forward(*args, **kwargs)
            return result

        return new_forward

    module.forward = make_new_forward(module.forward, dtype, enabled)


def unet_enc(
        unet,
        sample: torch.FloatTensor,
        timestep: Union[torch.Tensor, float, int],
        encoder_hidden_states: torch.Tensor,
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,
        added_cond_kwargs=None):
    # 0. center input if necessary
    if unet.config.center_input_sample:
        sample = 2 * sample - 1.0

    # 1. time
    t_emb = unet.get_time_embed(sample=sample, timestep=timestep)
    emb = unet.time_embedding(t_emb)
    aug_emb = unet.get_aug_embed(
        emb=emb, encoder_hidden_states=encoder_hidden_states, added_cond_kwargs=added_cond_kwargs)
    emb = emb + aug_emb if aug_emb is not None else emb

    if unet.time_embed_act is not None:
        emb = unet.time_embed_act(emb)

    encoder_hidden_states = unet.process_encoder_hidden_states(
        encoder_hidden_states=encoder_hidden_states, added_cond_kwargs=added_cond_kwargs)

    # 2. pre-process
    sample = unet.conv_in(sample)

    # 3. down
    down_block_res_samples = (sample,)
    for downsample_block in unet.down_blocks:
        if hasattr(downsample_block, "has_cross_attention") and downsample_block.has_cross_attention:
            sample, res_samples = downsample_block(
                hidden_states=sample,
                temb=emb,
                encoder_hidden_states=encoder_hidden_states,
                cross_attention_kwargs=cross_attention_kwargs,
            )
        else:
            sample, res_samples = downsample_block(hidden_states=sample, temb=emb)

        down_block_res_samples += res_samples

    return emb, down_block_res_samples, sample


def unet_dec(
        unet,
        emb,
        down_block_res_samples,
        sample,
        encoder_hidden_states: torch.Tensor,
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,
        down_block_additional_residuals: Optional[Tuple[torch.Tensor]] = None,
        mid_block_additional_residual: Optional[torch.Tensor] = None):
    is_controlnet = mid_block_additional_residual is not None and down_block_additional_residuals is not None

    if is_controlnet:
        new_down_block_res_samples = ()

        for down_block_res_sample, down_block_additional_residual in zip(
                down_block_res_samples, down_block_additional_residuals):
            down_block_res_sample = down_block_res_sample + down_block_additional_residual
            new_down_block_res_samples = new_down_block_res_samples + (down_block_res_sample,)

        down_block_res_samples = new_down_block_res_samples

    # 4. mid
    if unet.mid_block is not None:
        if hasattr(unet.mid_block, "has_cross_attention") and unet.mid_block.has_cross_attention:
            sample = unet.mid_block(
                sample,
                emb,
                encoder_hidden_states=encoder_hidden_states,
                cross_attention_kwargs=cross_attention_kwargs,
            )
        else:
            sample = unet.mid_block(sample, emb)

    if is_controlnet:
        sample = sample + mid_block_additional_residual

    # 5. up
    for i, upsample_block in enumerate(unet.up_blocks):
        res_samples = down_block_res_samples[-len(upsample_block.resnets):]
        down_block_res_samples = down_block_res_samples[: -len(upsample_block.resnets)]

        if hasattr(upsample_block, 'has_cross_attention') and upsample_block.has_cross_attention:
            sample = upsample_block(
                hidden_states=sample,
                temb=emb,
                res_hidden_states_tuple=res_samples,
                encoder_hidden_states=encoder_hidden_states,
                cross_attention_kwargs=cross_attention_kwargs,
            )
        else:
            sample = upsample_block(
                hidden_states=sample,
                temb=emb,
                res_hidden_states_tuple=res_samples,
            )

    # 6. post-process
    if unet.conv_norm_out:
        sample = unet.conv_norm_out(sample)
        sample = unet.conv_act(sample)
    sample = unet.conv_out(sample)

    return sample


@MODULES.register_module()
class UNet2DConditionModel(_UNet2DConditionModel):
    def __init__(self,
                 *args,
                 freeze=True,
                 freeze_exclude=[],
                 pretrained=None,
                 torch_dtype='float32',
                 freeze_exclude_fp32=True,
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.freeze = freeze
        if self.freeze:
            self.requires_grad_(False)
            for attr in freeze_exclude:
                rgetattr(self, attr).requires_grad_(True)

        self.init_weights(pretrained)
        if torch_dtype is not None:
            self.to(getattr(torch, torch_dtype))

        self.set_use_memory_efficient_attention_xformers(
            not hasattr(torch.nn.functional, 'scaled_dot_product_attention'))

        self.freeze_exclude_fp32 = freeze_exclude_fp32
        if self.freeze_exclude_fp32:
            for attr in freeze_exclude:
                m = rgetattr(self, attr)
                assert isinstance(m, nn.Module)
                m.to(torch.float32)
                autocast_patch(m, enabled=False)

    def init_weights(self, pretrained):
        if pretrained is not None:
            logger = get_root_logger()
            # load_checkpoint(self, pretrained, map_location='cpu', strict=False, logger=logger)
            checkpoint = _load_checkpoint(pretrained, map_location='cpu', logger=logger)
            if 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
            else:
                state_dict = checkpoint
            metadata = getattr(state_dict, '_metadata', OrderedDict())
            state_dict._metadata = metadata
            assert self.conv_in.weight.shape[1] == self.conv_out.weight.shape[0]
            if state_dict['conv_in.weight'].size() != self.conv_in.weight.size():
                assert state_dict['conv_in.weight'].shape[1] == state_dict['conv_out.weight'].shape[0]
                src_chn = state_dict['conv_in.weight'].shape[1]
                tgt_chn = self.conv_in.weight.shape[1]
                assert src_chn < tgt_chn
                convert_mat_out = torch.tile(torch.eye(src_chn), (ceildiv(tgt_chn, src_chn), 1))
                convert_mat_out = convert_mat_out[:tgt_chn]
                convert_mat_in = F.normalize(convert_mat_out.pinverse(), dim=-1)
                state_dict['conv_out.weight'] = torch.einsum(
                    'ts,scxy->tcxy', convert_mat_out, state_dict['conv_out.weight'])
                state_dict['conv_out.bias'] = torch.einsum(
                    'ts,s->t', convert_mat_out, state_dict['conv_out.bias'])
                state_dict['conv_in.weight'] = torch.einsum(
                    'st,csxy->ctxy', convert_mat_in, state_dict['conv_in.weight'])
            load_state_dict(self, state_dict, logger=logger)

    def forward(self, sample, timestep, encoder_hidden_states, **kwargs):
        dtype = sample.dtype
        return super().forward(
            sample, timestep, encoder_hidden_states, return_dict=False, **kwargs)[0].to(dtype)

    def forward_enc(self, sample, timestep, encoder_hidden_states, **kwargs):
        return unet_enc(self, sample, timestep, encoder_hidden_states, **kwargs)

    def forward_dec(self, emb, down_block_res_samples, sample, encoder_hidden_states, **kwargs):
        return unet_dec(self, emb, down_block_res_samples, sample, encoder_hidden_states, **kwargs)


@MODULES.register_module()
class UNetLoRAWrapper(nn.Module):
    def __init__(
            self,
            unet,
            lora_layers=dict(
                type='LoRAXFormersAttnProcessor',
                rank=4)):
        super().__init__()
        self.unet: UNet2DConditionModel = build_module(unet)
        lora_attn_procs = {}
        for name in self.unet.attn_processors.keys():
            cross_attention_dim = None if name.endswith('attn1.processor') \
                else self.unet.config.cross_attention_dim
            if name.startswith('mid_block'):
                hidden_size = self.unet.config.block_out_channels[-1]
            elif name.startswith('up_blocks'):
                block_id = int(name[len("up_blocks.")])
                hidden_size = list(reversed(self.unet.config.block_out_channels))[block_id]
            elif name.startswith('down_blocks'):
                block_id = int(name[len('down_blocks.')])
                hidden_size = self.unet.config.block_out_channels[block_id]
            lora_cfg = deepcopy(lora_layers)
            lora_cfg.update(dict(
                hidden_size=hidden_size,
                cross_attention_dim=cross_attention_dim))
            m = build_module(lora_cfg)
            if isinstance(m, ReferenceAttnProc):
                p = m.chained_proc
            else:
                p = m
            autocast_patch(p.to_k_lora, enabled=False)
            autocast_patch(p.to_v_lora, enabled=False)
            autocast_patch(p.to_q_lora, enabled=False)
            autocast_patch(p.to_out_lora, enabled=False)
            lora_attn_procs[name] = m
        self.unet.set_attn_processor(lora_attn_procs)
        # self.lora_layers = torch.nn.ModuleList(self.unet.attn_processors.values())

    @torch.no_grad()
    def bake_lora_weights(self):
        if torch.__version__ >= '2.0':
            ordinary_attn = AttnProcessor2_0()
        elif is_xformers_available():
            ordinary_attn = XFormersAttnProcessor()
        else:
            ordinary_attn = AttnProcessor()

        def expansion(lora: LoRALinearLayer):
            # in x in -> in x out
            eye = torch.eye(lora.down.weight.size(1))
            eye = eye.to(lora.down.weight)
            return lora(eye).T

        def traverse(name: str, module: Attention):
            if hasattr(module, "set_processor"):
                baked_processor = module.processor
                if isinstance(module.processor, ReferenceAttnProc):
                    baked_processor = baked_processor.chained_proc
                assert isinstance(baked_processor, (LoRAAttnProcessor, LoRAAttnProcessor2_0, LoRAXFormersAttnProcessor))
                module.to_k.weight += expansion(baked_processor.to_k_lora)
                module.to_v.weight += expansion(baked_processor.to_v_lora)
                module.to_q.weight += expansion(baked_processor.to_q_lora)
                module.to_out[0].weight += expansion(baked_processor.to_out_lora)
                if isinstance(module.processor, ReferenceAttnProc):
                    module.processor._modules.pop('chained_proc')
                    module.processor.chained_proc = ordinary_attn
                else:
                    module.set_processor(ordinary_attn)
    
            for sub_name, child in module.named_children():
                traverse(f"{name}.{sub_name}", child)

        for name, module in self.named_children():
            traverse(name, module)

    def forward(self, *args, **kwargs):
        return self.unet.forward(*args, **kwargs)


@MODULES.register_module()
class CLIPTextModel(_CLIPTextModel):
    def __init__(
            self,
            *args,
            freeze=True,
            pretrained=None,
            attention_dropout=0.0,
            bos_token_id=0,
            dropout=0.0,
            eos_token_id=2,
            hidden_act='gelu',
            hidden_size=1024,
            initializer_factor=1.0,
            initializer_range=0.02,
            intermediate_size=4096,
            layer_norm_eps=1e-05,
            max_position_embeddings=77,
            model_type='clip_text_model',
            num_attention_heads=16,
            num_hidden_layers=23,
            pad_token_id=1,
            projection_dim=512,
            torch_dtype='float32',
            transformers_version='4.25.0.dev0',
            vocab_size=49408,
            **kwargs):
        cfg = _CLIPTextModel.config_class(
            *args,
            attention_dropout=attention_dropout,
            bos_token_id=bos_token_id,
            dropout=dropout,
            eos_token_id=eos_token_id,
            hidden_act=hidden_act,
            hidden_size=hidden_size,
            initializer_factor=initializer_factor,
            initializer_range=initializer_range,
            intermediate_size=intermediate_size,
            layer_norm_eps=layer_norm_eps,
            max_position_embeddings=max_position_embeddings,
            model_type=model_type,
            num_attention_heads=num_attention_heads,
            num_hidden_layers=num_hidden_layers,
            pad_token_id=pad_token_id,
            projection_dim=projection_dim,
            torch_dtype=torch_dtype,
            transformers_version=transformers_version,
            vocab_size=vocab_size,
            **kwargs)
        self.pretrained = pretrained
        super(CLIPTextModel, self).__init__(cfg)
        self.to(cfg.torch_dtype)
        self.freeze = freeze
        if self.freeze:
            self.requires_grad_(False)

    def init_weights(self):
        if self.pretrained is None:
            super(CLIPTextModel, self).init_weights()
        else:
            logger = get_root_logger()
            load_checkpoint(self, self.pretrained, map_location='cpu', strict=False, logger=logger)


@MODULES.register_module()
class CLIPLoRAWrapper(nn.Module):
    def __init__(
            self,
            text_encoder,
            lora_layers=dict(
                type='LoRAXFormersAttnProcessor',
                rank=4)):
        super().__init__()
        self.text_encoder = build_module(text_encoder)
        self.lora_scale = 1.0
        lora_attn_procs = {}
        for name, module in self.text_encoder.named_modules():
            if name.endswith(TEXT_ENCODER_ATTN_MODULE):
                lora_cfg = deepcopy(lora_layers)
                lora_cfg.update(dict(
                    hidden_size=module.out_proj.out_features,
                    cross_attention_dim=None))
                m = build_module(lora_cfg)
                autocast_patch(m.to_k_lora, enabled=False)
                autocast_patch(m.to_v_lora, enabled=False)
                autocast_patch(m.to_q_lora, enabled=False)
                autocast_patch(m.to_out_lora, enabled=False)
                lora_attn_procs[name] = m
        self._modify_text_encoder(lora_attn_procs)
        self.lora_layers = nn.ModuleList(lora_attn_procs.values())

    @property
    def _lora_attn_processor_attr_to_text_encoder_attr(self):
        return {
            'to_q_lora': 'q_proj',
            'to_k_lora': 'k_proj',
            'to_v_lora': 'v_proj',
            'to_out_lora': 'out_proj',
        }

    def _remove_text_encoder_monkey_patch(self):
        # Loop over the CLIPAttention module of text_encoder
        for name, attn_module in self.text_encoder.named_modules():
            if name.endswith(TEXT_ENCODER_ATTN_MODULE):
                # Loop over the LoRA layers
                for _, text_encoder_attr in self._lora_attn_processor_attr_to_text_encoder_attr.items():
                    # Retrieve the q/k/v/out projection of CLIPAttention
                    module = attn_module.get_submodule(text_encoder_attr)
                    if hasattr(module, "old_forward"):
                        # restore original `forward` to remove monkey-patch
                        module.forward = module.old_forward
                        delattr(module, "old_forward")

    def _modify_text_encoder(self, attn_processors):
        r"""
        Monkey-patches the forward passes of attention modules of the text encoder.

        Parameters:
            attn_processors: Dict[str, `LoRAAttnProcessor`]:
                A dictionary mapping the module names and their corresponding [`~LoRAAttnProcessor`].
        """

        # First, remove any monkey-patch that might have been applied before
        self._remove_text_encoder_monkey_patch()

        # Loop over the CLIPAttention module of text_encoder
        for name, attn_module in self.text_encoder.named_modules():
            if name.endswith(TEXT_ENCODER_ATTN_MODULE):
                # Loop over the LoRA layers
                for attn_proc_attr, text_encoder_attr in self._lora_attn_processor_attr_to_text_encoder_attr.items():
                    # Retrieve the q/k/v/out projection of CLIPAttention and its corresponding LoRA layer.
                    module = attn_module.get_submodule(text_encoder_attr)
                    lora_layer = attn_processors[name].get_submodule(attn_proc_attr)

                    # save old_forward to module that can be used to remove monkey-patch
                    old_forward = module.old_forward = module.forward

                    # create a new scope that locks in the old_forward, lora_layer value for each new_forward function
                    # for more detail, see https://github.com/huggingface/diffusers/pull/3490#issuecomment-1555059060
                    def make_new_forward(old_forward, lora_layer):
                        def new_forward(x):
                            result = old_forward(x) + self.lora_scale * lora_layer(x)
                            return result

                        return new_forward

                    # Monkey-patch.
                    module.forward = make_new_forward(old_forward, lora_layer)

    def forward(self, *args, **kwargs):
        return self.text_encoder.forward(*args, **kwargs)


@MODULES.register_module()
class VAEDecoder(Decoder, ModelMixin):
    def __init__(
            self,
            in_channels=12,
            out_channels=24,
            up_block_types=('UpDecoderBlock2D',),
            block_out_channels=(64,),
            layers_per_block=2,
            norm_num_groups=32,
            act_fn='silu',
            norm_type='group',
            zero_init_residual=True):
        super(VAEDecoder, self).__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            up_block_types=up_block_types,
            block_out_channels=block_out_channels,
            layers_per_block=layers_per_block,
            norm_num_groups=norm_num_groups,
            act_fn=act_fn,
            norm_type=norm_type)
        self.set_use_memory_efficient_attention_xformers(
            not hasattr(torch.nn.functional, 'scaled_dot_product_attention'))
        self.zero_init_residual = zero_init_residual
        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                kaiming_init(m)
            elif isinstance(m, nn.GroupNorm):
                constant_init(m, 1)

        if self.zero_init_residual:
            for m in self.modules():
                if isinstance(m, ResnetBlock2D):
                    constant_init(m.conv2, 0)
                elif isinstance(m, Attention):
                    constant_init(m.to_out[0], 0)


@MODULES.register_module()
class PretrainedUNet(nn.Module):
    def __init__(self,
                 from_pretrained=None,
                 freeze=True,
                 torch_dtype='float32',
                 checkpointing=True,
                 use_ref_attn=False,
                 **kwargs):
        super().__init__()
        if torch_dtype is not None:
            kwargs.update(torch_dtype=getattr(torch, torch_dtype))
        self.unet = _UNet2DConditionModel.from_pretrained(
            from_pretrained, **kwargs)
        self.freeze = freeze
        if self.freeze:
            self.requires_grad_(False)
        self.unet.set_use_memory_efficient_attention_xformers(
            not hasattr(torch.nn.functional, 'scaled_dot_product_attention'))
        if use_ref_attn:
            unet_attn_procs = dict()
            for name, attn_proc in self.unet.attn_processors.items():
                unet_attn_procs[name] = ReferenceAttnProc(
                    attn_proc, enabled=name.endswith('attn1.processor'), name=name)
            self.unet.set_attn_processor(unet_attn_procs)
        if checkpointing:
            self.unet.enable_gradient_checkpointing()

    def forward(self, sample, timestep, encoder_hidden_states, **kwargs):
        return self.unet(
            sample, timestep, encoder_hidden_states, return_dict=False, **kwargs)[0]

    def forward_enc(self, sample, timestep, encoder_hidden_states, **kwargs):
        return unet_enc(self.unet, sample, timestep, encoder_hidden_states, **kwargs)

    def forward_dec(self, emb, down_block_res_samples, sample, encoder_hidden_states, **kwargs):
        return unet_dec(self.unet, emb, down_block_res_samples, sample, encoder_hidden_states, **kwargs)


@MODULES.register_module()
class PretrainedControlNet(nn.Module):
    def __init__(self,
                 from_pretrained=None,
                 freeze=True,
                 torch_dtype='float32',
                 checkpointing=True,
                 **kwargs):
        super().__init__()
        if torch_dtype is not None:
            kwargs.update(torch_dtype=getattr(torch, torch_dtype))
        self.controlnet = ControlNetModel.from_pretrained(
            from_pretrained, **kwargs)
        self.freeze = freeze
        if self.freeze:
            self.requires_grad_(False)
        self.controlnet.set_use_memory_efficient_attention_xformers(
            not hasattr(torch.nn.functional, 'scaled_dot_product_attention'))
        if checkpointing:
            self.controlnet.enable_gradient_checkpointing()

    def forward(self, *args, **kwargs):
        return self.controlnet.forward(*args, **kwargs)


@MODULES.register_module()
class PretrainedVAE(nn.Module):
    def __init__(self,
                 from_pretrained=None,
                 del_encoder=False,
                 del_decoder=False,
                 use_slicing=False,
                 freeze=True,
                 torch_dtype='float32',
                 **kwargs):
        super().__init__()
        if torch_dtype is not None:
            kwargs.update(torch_dtype=getattr(torch, torch_dtype))
        self.vae = AutoencoderKL.from_pretrained(
            from_pretrained, **kwargs)
        if del_encoder:
            del self.vae.encoder
        if del_decoder:
            del self.vae.decoder
        if use_slicing:
            self.vae.enable_slicing()
        self.freeze = freeze
        if self.freeze:
            self.requires_grad_(False)
        self.vae.set_use_memory_efficient_attention_xformers(
            not hasattr(torch.nn.functional, 'scaled_dot_product_attention'))

    def forward(self, *args, **kwargs):
        return self.vae.forward(*args, **kwargs)[0]

    def encode(self, img):
        return self.vae.encode(img).latent_dist.sample() * self.vae.config.scaling_factor

    def decode(self, code):
        return self.vae.decode(code / self.vae.config.scaling_factor, return_dict=False)[0]

    def visualize(self, code, scene_name, viz_dir):
        code_viz = (self(code).to(dtype=torch.float32, device='cpu').numpy().transpose(0, 2, 3, 1) / 2 + 0.5).clip(
            min=0, max=1)
        code_viz = np.round(code_viz * 255).astype(np.uint8)
        for code_viz_single, scene_name_single in zip(code_viz, scene_name):
            plt.imsave(os.path.join(viz_dir, 'scene_' + scene_name_single + '_dec.png'), code_viz_single)


@MODULES.register_module()
class PretrainedVAEDecoder(PretrainedVAE):
    def __init__(self, **kwargs):
        super().__init__(
            del_encoder=True,
            del_decoder=False,
            **kwargs)

    def forward(self, code):
        return super().decode(code)


@MODULES.register_module()
class PretrainedVAEEncoder(PretrainedVAE):
    def __init__(self, **kwargs):
        super().__init__(
            del_encoder=False,
            del_decoder=True,
            **kwargs)

    def forward(self, img):
        return super().encode(img)


class ReferenceAttnProc(torch.nn.Module):
    def __init__(
            self,
            chained_proc,
            enabled=False,
            name=None
    ) -> None:
        super().__init__()
        self.enabled = enabled
        self.chained_proc = chained_proc
        self.name = name

    def __call__(
            self, attn: Attention, hidden_states, encoder_hidden_states=None, attention_mask=None,
            mode='w', ref_dict: dict = None):
        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        if self.enabled:
            if mode == 'w':
                ref_dict[self.name] = encoder_hidden_states
            elif mode == 'r':
                encoder_hidden_states = torch.cat([encoder_hidden_states, ref_dict.pop(self.name)], dim=1)
            elif mode == 'm':
                encoder_hidden_states = torch.cat([encoder_hidden_states, ref_dict[self.name]], dim=1)
            else:
                assert False, mode
        res = self.chained_proc(attn, hidden_states, encoder_hidden_states, attention_mask)
        return res
