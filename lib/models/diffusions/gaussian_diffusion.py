import sys
import inspect
import math
import numpy as np
import torch
import torch.nn as nn
import mmcv
import diffusers

from copy import deepcopy
from mmgen.models.architectures.common import get_module_device
from mmgen.models.builder import MODULES, build_module
from mmgen.models.diffusions.utils import var_to_tensor, _get_noise_batch
from lib.core import get_noise_scales


@MODULES.register_module()
class GaussianDiffusion(nn.Module):

    prediction_type_mapping = dict(
        EPS='epsilon',
        START_X='sample',
        V='v_prediction')

    available_samplers = [
        'DDIM',
        'DDPM',
        'DEISMultistep',
        'DPMSolverMultistep',
        'DPMSolverSDE',
        'DPMSolverSinglestep',
        'EulerAncestralDiscrete',
        'EulerDiscrete',
        'HeunDiscrete',
        'KDPM2AncestralDiscrete',
        'KDPM2Discrete',
        'LMSDiscrete',
        'PNDM',
        'UniPCMultistep',
    ]

    # @property
    # def available_samplers(self):
    #     available_samplers = []
    #     for scheduler in dir(diffusers.schedulers):
    #         if scheduler.endswith('Scheduler'):
    #             available_samplers.append(scheduler[:-9])
    #     return available_samplers

    def __init__(self,
                 denoising,
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
                 train_cfg=None,
                 test_cfg=None):
        super().__init__()
        # build denoising module in this function
        self.num_classes = num_classes
        self.num_timesteps = num_timesteps
        self.sample_method = sample_method
        self._denoising_cfg = deepcopy(denoising)
        self.denoising = build_module(
            denoising,
            default_args=dict(
                num_classes=num_classes, num_timesteps=num_timesteps))
        self.denoising_mean_mode = denoising_mean_mode

        self.betas_cfg = deepcopy(betas_cfg)

        self.train_cfg = deepcopy(train_cfg) if train_cfg is not None else dict()
        self.test_cfg = deepcopy(test_cfg) if test_cfg is not None else dict()

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

        self.intermediate_samples = []

    @staticmethod
    def linear_beta_schedule(diffusion_timesteps, beta_0=1e-4, beta_T=2e-2):
        r"""Linear schedule from Ho et al, extended to work for any number of
        diffusion steps.

        Args:
            diffusion_timesteps (int): The number of betas to produce.
            beta_0 (float, optional): `\beta` at timestep 0. Defaults to 1e-4.
            beta_T (float, optional): `\beta` at timestep `T` (the final
                diffusion timestep). Defaults to 2e-2.

        Returns:
            np.ndarray: Betas used in diffusion process.
        """
        scale = 1000 / diffusion_timesteps
        beta_0 = scale * beta_0
        beta_T = scale * beta_T
        return np.linspace(
            beta_0, beta_T, diffusion_timesteps, dtype=np.float64)

    @staticmethod
    def cosine_beta_schedule(diffusion_timesteps, max_beta=0.999, s=0.008):
        r"""Create a beta schedule that discretizes the given alpha_t_bar
        function, which defines the cumulative product of `(1-\beta)` over time
        from `t = [0, 1]`.

        Args:
            diffusion_timesteps (int): The number of betas to produce.
            max_beta (float, optional): The maximum beta to use; use values
                lower than 1 to prevent singularities. Defaults to 0.999.
            s (float, optional): Small offset to prevent `\beta` from being too
                small near `t = 0` Defaults to 0.008.

        Returns:
            np.ndarray: Betas used in diffusion process.
        """

        def f(t, T, s):
            return np.cos((t / T + s) / (1 + s) * np.pi / 2) ** 2

        betas = []
        for t in range(diffusion_timesteps):
            alpha_bar_t = f(t + 1, diffusion_timesteps, s)
            alpha_bar_t_1 = f(t, diffusion_timesteps, s)
            betas_t = 1 - alpha_bar_t / alpha_bar_t_1
            betas.append(min(betas_t, max_beta))
        return np.array(betas)

    def get_betas(self):
        """Get betas by defined schedule method in diffusion process."""
        self.betas_schedule = self.betas_cfg.pop('type')
        if self.betas_schedule == 'linear':
            return self.linear_beta_schedule(self.num_timesteps,
                                             **self.betas_cfg)
        elif self.betas_schedule == 'cosine':
            return self.cosine_beta_schedule(self.num_timesteps,
                                             **self.betas_cfg)
        elif self.betas_schedule == 'scaled_linear':
            return np.linspace(
                self.betas_cfg.get('beta_start', 0.0001) ** 0.5,
                self.betas_cfg.get('beta_end', 0.02) ** 0.5,
                self.num_timesteps,
                dtype=np.float64) ** 2
        else:
            raise AttributeError(f'Unknown method name {self.beta_schedule}'
                                 'for beta schedule.')

    def prepare_diffusion_vars(self):
        self.betas = self.get_betas()
        self.alphas = 1.0 - self.betas
        self.alphas_bar = np.cumproduct(self.alphas, axis=0)

        # calculations for diffusion q(x_t | x_0) and others
        self.sqrt_alphas_bar = np.sqrt(self.alphas_bar)
        self.sqrt_one_minus_alphas_bar = np.sqrt(1.0 - self.alphas_bar)

    def q_sample(self, x_0, t, noise=None):
        tar_shape = x_0.shape
        if noise is None:
            noise = _get_noise_batch(
                None, x_0.shape[-3:],
                num_timesteps=self.num_timesteps,
                num_batches=x_0.size(0),
                timesteps_noise=False
            ).to(x_0)  # (num_batches, num_channels, h, w)
        mean = var_to_tensor(self.sqrt_alphas_bar, t.cpu(), tar_shape).to(x_0)
        std = var_to_tensor(self.sqrt_one_minus_alphas_bar, t.cpu(), tar_shape).to(x_0)
        return x_0 * mean + noise * std, mean, std

    def pred_x_0(self, x_t, t, grad_guide_fn=None, concat_cond=None, cfg=dict()):
        clip_denoised = cfg.get('clip_denoised', True)
        clip_range = cfg.get('clip_range', [-1, 1])
        guidance_gain = cfg.get('guidance_gain', 1.0)
        grad_through_unet = cfg.get('grad_through_unet', True)
        snr_weight_power = cfg.get('snr_weight_power', 0.5)

        num_batches = x_t.size(0)
        if t.dim() == 0 or len(t) != num_batches:
            t = t.expand(num_batches)
        sqrt_alpha_bar_t, sqrt_one_minus_alpha_bar_t = get_noise_scales(
            self.alphas_bar, t.reshape(-1, 1, 1, 1), self.num_timesteps, dtype=x_t.dtype)

        if grad_guide_fn is not None and grad_through_unet:
            x_t.requires_grad = True
            grad_enabled_prev = torch.is_grad_enabled()
            torch.set_grad_enabled(True)

        denoising_output = self.denoising(x_t, t, concat_cond=concat_cond)

        if self.denoising_mean_mode.upper() == 'EPS':
            x_0_pred = (x_t - sqrt_one_minus_alpha_bar_t * denoising_output) / sqrt_alpha_bar_t
        elif self.denoising_mean_mode.upper() == 'START_X':
            x_0_pred = denoising_output
        elif self.denoising_mean_mode.upper() == 'V':
            x_0_pred = sqrt_alpha_bar_t * x_t - sqrt_one_minus_alpha_bar_t * denoising_output
        else:
            raise AttributeError('Unknown denoising mean output type '
                                 f'[{self.denoising_mean_mode}].')

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

    def p_sample_langevin(self,
                          x_t,
                          t,
                          noise=None,
                          cfg=dict(),
                          grad_guide_fn=None,
                          **kwargs):
        device = get_module_device(self)
        langevin_delta = cfg.get('langevin_delta', 0.1)
        sigma = self.sqrt_one_minus_alphas_bar[t]
        x_0_pred, _ = self.pred_x_0(x_t, t, grad_guide_fn=grad_guide_fn, cfg=cfg, **kwargs)
        eps_t_pred = (x_t - self.sqrt_alphas_bar[t] * x_0_pred) / sigma
        if noise is None:
            noise = _get_noise_batch(
                None, x_t.shape[-3:],
                num_timesteps=self.num_timesteps,
                num_batches=x_t.size(0),
                timesteps_noise=False
            ).to(device)  # (num_batches, num_channels, h, w)
        x_t = x_t - 0.5 * langevin_delta * sigma * eps_t_pred + math.sqrt(langevin_delta) * sigma * noise
        return x_t

    def sample_from_noise(self, noise, show_pbar=False, concat_cond=None, save_intermediates=False, **kwargs):
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
                        x_t, t, concat_cond=cc, cfg=self.test_cfg, **kwargs)
                    cond_step += 1

            cc = concat_cond[:, cond_step % concat_cond.size(1)] if concat_cond is not None else None
            x_0_pred, denoising_output = self.pred_x_0(
                sampler.scale_model_input(x_t, t), t, concat_cond=cc, cfg=self.test_cfg, **kwargs)
            cond_step += 1

            if save_intermediates:
                self.intermediate_samples.append(x_0_pred.detach().cpu())

            x_t = sampler.step(denoising_output, t, x_t, return_dict=False)[0]

            if show_pbar:
                pbar.update()

        if show_pbar:
            sys.stdout.write('\n')

        return x_t

    def loss(self, denoising_output, x_0, noise, t, mean, std):
        if self.denoising_mean_mode.upper() == 'EPS':
            loss_kwargs = dict(eps_t_pred=denoising_output)
        elif self.denoising_mean_mode.upper() == 'START_X':
            loss_kwargs = dict(x_0_pred=denoising_output)
        elif self.denoising_mean_mode.upper() == 'V':
            loss_kwargs = dict(v_t_pred=denoising_output)
        else:
            raise AttributeError('Unknown denoising mean output type '
                                 f'[{self.denoising_mean_mode}].')
        loss_kwargs.update(
            x_0=x_0,
            noise=noise,
            timesteps=t)
        if 'v_t_pred' in loss_kwargs:
            loss_kwargs.update(v_t=mean * noise - std * x_0)
        return self.ddpm_loss(loss_kwargs)

    def forward_train(self, x_0, concat_cond=None, grad_guide_fn=None, cfg=dict(),
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

        cfg = deepcopy(cfg)
        cfg.update(clip_denoised=False)
        _, denoising_output = self.pred_x_0(
            x_t, t, grad_guide_fn=grad_guide_fn, concat_cond=concat_cond, cfg=cfg)
        loss = self.loss(denoising_output, x_0, noise, t, mean, std)
        log_vars = self.ddpm_loss.log_vars
        log_vars.update(loss_ddpm_mse=float(loss))

        return loss, log_vars

    def forward_test(self, data, **kwargs):
        """Testing function for Diffusion Denosing Probability Models.

        Args:
            data (torch.Tensor | dict | None): Input data. This data will be
                passed to different methods.
        """
        assert data.dim() == 4
        return self.sample_from_noise(data, **kwargs)

    def forward(self, data, return_loss=False, **kwargs):
        if return_loss:
            return self.forward_train(data, **kwargs)

        return self.forward_test(data, **kwargs)
