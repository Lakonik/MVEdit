import sys
import inspect
import torch
import torch.nn as nn
import mmcv
import diffusers

from copy import deepcopy
from transformers import CLIPTokenizer
from transformers.models.clip.modeling_clip import CLIPTextTransformer
from mmgen.models.architectures.common import get_module_device
from mmgen.models.builder import MODULES, build_module
from mmgen.models.diffusions.utils import _get_noise_batch
from lib.core import get_noise_scales
from .gaussian_diffusion import GaussianDiffusion
from ..architecture import CLIPLoRAWrapper, CLIPTextModel


def _make_causal_mask(
    input_ids_shape: torch.Size, dtype: torch.dtype, device: torch.device, past_key_values_length: int = 0
):
    """
    Make causal mask used for bi-directional self-attention.
    """
    bsz, tgt_len = input_ids_shape
    mask = torch.full((tgt_len, tgt_len), torch.finfo(dtype).min, device=device)
    mask_cond = torch.arange(mask.size(-1), device=device)
    mask.masked_fill_(mask_cond < (mask_cond + 1).view(mask.size(-1), 1), 0)
    mask = mask.to(dtype)

    if past_key_values_length > 0:
        mask = torch.cat([torch.zeros(tgt_len, past_key_values_length, dtype=dtype, device=device), mask], dim=-1)
    return mask[None, None, :, :].expand(bsz, 1, tgt_len, tgt_len + past_key_values_length)


@MODULES.register_module()
class GaussianDiffusionText(GaussianDiffusion):

    def __init__(self,
                 denoising,
                 text_encoder=None,
                 tokenizer=None,
                 inversion_init=None,
                 ddpm_loss=dict(
                     type='DDPMMSELoss',
                     log_cfgs=dict(
                         type='quartile', prefix_name='loss_mse', total_timesteps=1000)),
                 betas_cfg=dict(type='cosine'),
                 num_timesteps=1000,
                 num_classes=0,
                 sample_method='ddim',
                 timestep_sampler=dict(type='UniformTimeStepSampler'),
                 denoising_mean_mode='V',
                 guidance_embeddings=None,
                 train_cfg=None,
                 test_cfg=None):
        super(GaussianDiffusion, self).__init__()
        # build denoising module in this function
        self.num_classes = num_classes
        self.num_timesteps = num_timesteps
        self.sample_method = sample_method
        self.denoising = build_module(denoising)
        self.text_encoder = build_module(text_encoder)
        assert isinstance(self.text_encoder, (CLIPTextModel, CLIPLoRAWrapper))
        assert isinstance(self.clip_text_model.text_model, CLIPTextTransformer)
        self.tokenizer = CLIPTokenizer.from_pretrained(tokenizer, subfolder='tokenizer')

        self.textual_inversions = None

        # get output-related configs from denoising
        self.denoising_mean_mode = denoising_mean_mode

        self.betas_cfg = deepcopy(betas_cfg)
        self.prepare_diffusion_vars()

        # build sampler
        self.timestep_sampler = build_module(
            timestep_sampler,
            default_args=dict(
                num_timesteps=num_timesteps,
                mean=self.sqrt_alphas_bar,
                std=self.sqrt_one_minus_alphas_bar,
                mode=self.denoising_mean_mode))
        self.ddpm_loss = build_module(ddpm_loss, default_args=dict(sampler=self.timestep_sampler))

        self.train_cfg = deepcopy(train_cfg) if train_cfg is not None else dict()
        self.test_cfg = deepcopy(test_cfg) if test_cfg is not None else dict()

        self.guidance_embeddings = None
        if guidance_embeddings is not None:
            embeds = torch.load(guidance_embeddings)
            self.guidance_embeddings = embeds['pos'] - embeds['neg']

        self.init_weights(inversion_init)

    @property
    def clip_text_model(self):
        return self.text_encoder.text_encoder if isinstance(
            self.text_encoder, CLIPLoRAWrapper) else self.text_encoder

    def init_weights(self, inversion_init):
        if isinstance(inversion_init, str):
            token_ids = self.tokenizer.encode(inversion_init, add_special_tokens=False)
            init_embeddings = self.clip_text_model.get_input_embeddings().weight.data[token_ids]
            self.textual_inversions = nn.Embedding.from_pretrained(init_embeddings, freeze=False)

    def _encode_text_prompt(self, prompt):
        """
        Modify the encoding pipeline to inject textual inversions
        """
        device = get_module_device(self)
        text_inputs = self.tokenizer(
            prompt,
            padding='max_length',
            max_length=self.tokenizer.model_max_length - (
                self.textual_inversions.weight.size(0) if self.textual_inversions is not None else 0),
            truncation=True,
            return_tensors='pt')
        input_ids = text_inputs.input_ids.to(device)
        if hasattr(self.clip_text_model.config, 'use_attention_mask') and self.clip_text_model.config.use_attention_mask:
            attention_mask = text_inputs.attention_mask.to(device)
        else:
            attention_mask = None
        bs, seq_len = input_ids.size()
        hidden_states = self.clip_text_model.text_model.embeddings(input_ids=input_ids)
        if self.textual_inversions is not None:
            hidden_states = torch.cat([
                hidden_states[:, :1],
                self.textual_inversions.weight.unsqueeze(0).expand(bs, -1, -1),
                hidden_states[:, 1:]], dim=1)
            seq_len = seq_len + self.textual_inversions.weight.size(0)
            if attention_mask is not None:
                attention_mask = torch.cat(
                    [attention_mask[:, :1],
                     torch.ones((bs, self.textual_inversions.weight.size(0)), dtype=attention_mask.dtype, device=device),
                     attention_mask[:, 1:]], dim=1)
        causal_attention_mask = _make_causal_mask((bs, seq_len), hidden_states.dtype, device=device)
        assert attention_mask is None
        encoder_outputs = self.clip_text_model.text_model.encoder(
            inputs_embeds=hidden_states,
            attention_mask=attention_mask,
            causal_attention_mask=causal_attention_mask)
        last_hidden_state = encoder_outputs[0]
        last_hidden_state = self.clip_text_model.text_model.final_layer_norm(last_hidden_state)
        return last_hidden_state

    def pred_x_0(self, x_t, t, encoder_hidden_states=None, uncond_hidden_states=None,
                 grad_guide_fn=None, concat_cond=None, cfg=dict(), update_denoising_output=False):
        assert encoder_hidden_states is not None
        assert concat_cond is None
        clip_denoised = cfg.get('clip_denoised', True)
        clip_range = cfg.get('clip_range', [-1, 1])
        guidance_gain = cfg.get('guidance_gain', 1.0)
        grad_through_unet = cfg.get('grad_through_unet', True)
        snr_weight_power = cfg.get('snr_weight_power', 0.5)
        cfg_scale = cfg.get('cfg_scale', 1.0)

        num_batches = x_t.size(0)
        if t.dim() == 0 or len(t) != num_batches:
            t = t.expand(num_batches)
        sqrt_alpha_bar_t, sqrt_one_minus_alpha_bar_t = get_noise_scales(
            self.alphas_bar, t.reshape(-1, 1, 1, 1), self.num_timesteps, dtype=x_t.dtype)

        if grad_guide_fn is not None and grad_through_unet:
            x_t.requires_grad = True
            grad_enabled_prev = torch.is_grad_enabled()
            torch.set_grad_enabled(True)

        if cfg_scale != 1:
            assert uncond_hidden_states is not None
            bs, c, h, w = x_t.size()
            x_t_ = x_t[None].expand(2, -1, -1, -1, -1).reshape(2 * bs, c, h, w)
            t_ = t[None].expand(2, -1).reshape(2 * bs)
            hidden_states = torch.cat(
                [encoder_hidden_states, uncond_hidden_states], dim=0)
            denoising_output_ = self.denoising(x_t_, t_, hidden_states).reshape(2, bs, c, h, w)
            denoising_output = denoising_output_[0] * cfg_scale + (1 - cfg_scale) * denoising_output_[1]
        else:
            denoising_output = self.denoising(x_t, t, encoder_hidden_states)

        x_0_pred = self.convert_to_x_0(x_t, denoising_output, sqrt_alpha_bar_t, sqrt_one_minus_alpha_bar_t)

        if grad_guide_fn is not None:
            if clip_denoised:
                x_0_pred = x_0_pred.clamp(*clip_range)
            if grad_through_unet:
                loss = grad_guide_fn(x_0_pred)
                grad = torch.autograd.grad(loss, x_t)[0]
            else:
                x_0_pred.requires_grad = True
                grad_enabled_prev = torch.is_grad_enabled()
                torch.set_grad_enabled(True)
                loss = grad_guide_fn(x_0_pred)
                grad = torch.autograd.grad(loss, x_0_pred)[0]
            torch.set_grad_enabled(grad_enabled_prev)
            x_0_pred = x_0_pred - grad * (
                (sqrt_one_minus_alpha_bar_t ** (2 - snr_weight_power * 2))
                * (sqrt_alpha_bar_t ** (snr_weight_power * 2 - 1))
                * guidance_gain)
            if self.denoising_mean_mode.upper() == 'EPS':
                denoising_output = denoising_output + grad * (
                    (sqrt_one_minus_alpha_bar_t ** (1 - snr_weight_power * 2))
                    * (sqrt_alpha_bar_t ** (snr_weight_power * 2))
                    * guidance_gain)
            elif self.denoising_mean_mode.upper() == 'START_X':
                denoising_output = x_0_pred
            elif self.denoising_mean_mode.upper() == 'V':
                denoising_output = denoising_output + grad * (
                    (sqrt_one_minus_alpha_bar_t ** (1 - snr_weight_power * 2))
                    * (sqrt_alpha_bar_t ** (snr_weight_power * 2 - 1))
                    * guidance_gain)

        if clip_denoised:
            x_0_pred = x_0_pred.clamp(*clip_range)
            if self.denoising_mean_mode.upper() == 'EPS':
                mid = x_t / sqrt_one_minus_alpha_bar_t
                denoising_output = torch.minimum(torch.maximum(
                    denoising_output,
                    mid - (clip_range[1] * sqrt_alpha_bar_t / sqrt_one_minus_alpha_bar_t)),
                    mid - (clip_range[0] * sqrt_alpha_bar_t / sqrt_one_minus_alpha_bar_t))
            elif self.denoising_mean_mode.upper() == 'START_X':
                denoising_output = x_0_pred
            elif self.denoising_mean_mode.upper() == 'V':
                mid = x_t * (sqrt_alpha_bar_t / sqrt_one_minus_alpha_bar_t)
                denoising_output = torch.minimum(torch.maximum(
                    denoising_output,
                    mid - (clip_range[1] / sqrt_one_minus_alpha_bar_t)),
                    mid - (clip_range[0] / sqrt_one_minus_alpha_bar_t))

        return x_0_pred, denoising_output

    def sample_from_noise(self, noise, prompts=None, neg_prompts=None, show_pbar=False, concat_cond=None,
                          save_intermediates=False, **kwargs):
        sampler_class = None
        for sampler in self.available_samplers:
            if self.sample_method.lower() == sampler.lower():
                sampler_class = getattr(diffusers.schedulers, sampler + 'Scheduler')
                break
        if sampler_class is None:
            raise AttributeError(f'Cannot find sampler [{self.sample_method}].')
        sampler_kwargs = dict()
        signatures = inspect.signature(sampler_class).parameters.keys()
        if 'timestep_spacing' in signatures:
            sampler_kwargs['timestep_spacing'] = 'trailing'
        if 'clip_sample' in signatures:
            sampler_kwargs['clip_sample'] = False
        if 'use_karras_sigmas' in signatures:
            sampler_kwargs['use_karras_sigmas'] = True
        sampler = sampler_class(
            self.num_timesteps, trained_betas=self.betas,
            prediction_type=self.prediction_type_mapping[self.denoising_mean_mode.upper()],
            **sampler_kwargs)

        sampler.set_timesteps(self.test_cfg.get('num_timesteps', self.num_timesteps), device=noise.device)
        timesteps = sampler.timesteps
        langevin_steps = self.test_cfg.get('langevin_steps', 0)
        langevin_t_range = self.test_cfg.get('langevin_t_range', [0, self.num_timesteps - 1])

        x_t = noise * sampler.init_noise_sigma

        if prompts is None:
            prompts = ['' for _ in range(x_t.size(0))]
        encoder_hidden_states = self._encode_text_prompt(prompts)
        if self.test_cfg.get('cfg_scale', 1.0) != 1.0:
            if neg_prompts is None:
                neg_prompts = ['' for _ in range(x_t.size(0))]
            uncond_hidden_states = self._encode_text_prompt(neg_prompts)
            if self.guidance_embeddings is not None:
                uncond_hidden_states = (
                    uncond_hidden_states - self.test_cfg.get('embed_guidance_scale', 0.0)
                    * self.guidance_embeddings.to(encoder_hidden_states))
        else:
            uncond_hidden_states = None

        if show_pbar:
            pbar = mmcv.ProgressBar(len(timesteps))
        cond_step = 0
        self.intermediate_samples = []

        for t in timesteps:
            if langevin_steps > 0 and langevin_t_range[0] < t < langevin_t_range[1]:
                assert sampler.init_noise_sigma == 1.0, \
                    'Langevin sampling only works with variance-preserving schedulers.'
                for _ in range(langevin_steps):
                    cc = concat_cond[:, cond_step % concat_cond.size(1)] if concat_cond is not None else None
                    x_t = self.p_sample_langevin(
                        x_t, t,
                        encoder_hidden_states=encoder_hidden_states, uncond_hidden_states=uncond_hidden_states,
                        concat_cond=cc, cfg=self.test_cfg, **kwargs)
                    cond_step += 1

            cc = concat_cond[:, cond_step % concat_cond.size(1)] if concat_cond is not None else None
            x_0_pred, denoising_output = self.pred_x_0(
                sampler.scale_model_input(x_t, t), t,
                encoder_hidden_states=encoder_hidden_states, uncond_hidden_states=uncond_hidden_states,
                concat_cond=cc, cfg=self.test_cfg, **kwargs)
            cond_step += 1

            if save_intermediates:
                self.intermediate_samples.append(x_0_pred.detach().cpu())

            x_t = sampler.step(denoising_output, t, x_t, return_dict=False)[0]

            if show_pbar:
                pbar.update()

        if show_pbar:
            sys.stdout.write('\n')

        return x_t

    def forward_train(self, x_0, prompts=None,
                      concat_cond=None, grad_guide_fn=None, cfg=dict(),
                      x_t_detach=False, **kwargs):
        device = get_module_device(self)

        assert x_0.dim() == 4
        num_batches, num_channels, h, w = x_0.size()

        t = self.timestep_sampler(num_batches).to(device)

        noise = _get_noise_batch(
            None, x_0.shape[-3:],
            num_timesteps=self.num_timesteps,
            num_batches=x_0.size(0),
            timesteps_noise=False
        ).to(x_0)  # (num_batches, num_channels, h, w)
        x_t, mean, std = self.q_sample(x_0, t, noise)
        if x_t_detach:
            x_t.detach_()

        if prompts is None:
            prompts = ['' for _ in range(num_batches)]
        encoder_hidden_states = self._encode_text_prompt(prompts)

        cfg = deepcopy(cfg)
        cfg.update(clip_denoised=False)
        _, denoising_output = self.pred_x_0(
            x_t, t, encoder_hidden_states=encoder_hidden_states,
            grad_guide_fn=grad_guide_fn, concat_cond=concat_cond, cfg=cfg)
        loss = self.loss(denoising_output, x_0, noise, t, mean, std)
        log_vars = self.ddpm_loss.log_vars
        log_vars.update(loss_ddpm_mse=float(loss))

        return loss, log_vars
