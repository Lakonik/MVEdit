import os
import numpy as np
import torch
import torch.distributed as dist
import mmcv

from torch.nn.parallel.distributed import DistributedDataParallel
from mmcv.runner import get_dist_info
from mmgen.models.builder import MODELS, build_module
from mmgen.models.architectures.common import get_module_device

from ...core import eval_psnr, rgetattr, module_requires_grad
from .base_nerf import get_cam_rays
from .diffusion_nerf import DiffusionNeRF


@MODELS.register_module()
class DiffusionNeRFText(DiffusionNeRF):

    def __init__(self,
                 *args,
                 decoder_debug=None,
                 **kwargs):
        super(DiffusionNeRFText, self).__init__(*args, **kwargs)
        self.decoder_debug = build_module(decoder_debug) if decoder_debug is not None else None

    def train_step(self, data, optimizer, loss_scaler=None, running_status=None):
        diffusion = self.diffusion
        decoder = self.decoder_ema if self.freeze_decoder and self.decoder_use_ema else self.decoder

        num_scenes = len(data['scene_id'])
        extra_scene_step = self.train_cfg.get('extra_scene_step', 0)

        if 'optimizer' in self.train_cfg:
            code_list_, code_optimizers, density_grid, density_bitfield = self.load_cache(
                data, freeze_code=self.train_cfg.get('freeze_code', False))
            code = self.code_activation(torch.stack(code_list_, dim=0), update_stats=True)
        else:
            assert 'code' in data
            code, density_grid, density_bitfield = self.load_scene(
                data, load_density='decoder' in optimizer)
            code_optimizers = []

        for key in optimizer.keys():
            if key.startswith('diffusion'):
                optimizer[key].zero_grad()
        for code_optimizer in code_optimizers:
            code_optimizer.zero_grad()
        if 'decoder' in optimizer:
            optimizer['decoder'].zero_grad()

        concat_cond = None
        if 'cond_imgs' in data:
            cond_imgs = data['cond_imgs']  # (num_scenes, num_imgs, h, w, 3)
            cond_intrinsics = data['cond_intrinsics']  # (num_scenes, num_imgs, 4), in [fx, fy, cx, cy]
            cond_poses = data['cond_poses']

            num_scenes, num_imgs, h, w, _ = cond_imgs.size()
            # (num_scenes, num_imgs, h, w, 3)
            cond_rays_o, cond_rays_d = get_cam_rays(cond_poses, cond_intrinsics, h, w)
            dt_gamma_scale = self.train_cfg.get('dt_gamma_scale', 0.0)
            # (num_scenes,)
            dt_gamma = dt_gamma_scale / cond_intrinsics[..., :2].mean(dim=(-2, -1))

            if self.image_cond:
                cond_inds = torch.randint(num_imgs, size=(num_scenes,))  # (num_scenes,)
                concat_cond = cond_imgs[range(num_scenes), cond_inds].permute(0, 3, 1, 2)  # (num_scenes, 3, h, w)
                diff_image_size = rgetattr(diffusion, 'denoising.image_size')
                assert diff_image_size[0] % concat_cond.size(-2) == 0
                assert diff_image_size[1] % concat_cond.size(-1) == 0
                concat_cond = concat_cond.tile((diff_image_size[0] // concat_cond.size(-2),
                                                diff_image_size[1] // concat_cond.size(-1)))

        prompts = data['prompts']
        uncond_prob = self.train_cfg.get('uncond_prob', 0.0)
        if uncond_prob > 0.0:
            uncond_mask = np.random.rand(num_scenes) < uncond_prob
            prompts = ['' if uncond else prompt for uncond, prompt in zip(uncond_mask, prompts)]

        x_t_detach = self.train_cfg.get('x_t_detach', False)

        with torch.autocast(
                device_type='cuda',
                enabled=self.autocast_dtype is not None,
                dtype=getattr(torch, self.autocast_dtype) if self.autocast_dtype is not None else None):
            loss_diffusion, log_vars = diffusion(
                self.code_diff_pr(code), concat_cond=concat_cond, return_loss=True,
                x_t_detach=x_t_detach, cfg=self.train_cfg, prompts=prompts)
        loss_diffusion.backward() if loss_scaler is None else loss_scaler.scale(loss_diffusion).backward()
        for key in optimizer.keys():
            if key.startswith('diffusion'):
                if loss_scaler is None:
                    optimizer[key].step()
                else:
                    loss_scaler.unscale_(optimizer[key])
                    loss_scaler.step(optimizer[key])
        if loss_scaler is not None:
            for code_optimizer in code_optimizers:
                loss_scaler.unscale_(code_optimizer)

        if extra_scene_step > 0:
            assert len(code_optimizers) > 0
            prior_grad = [code_.grad.data.clone() for code_ in code_list_]
            cfg = self.train_cfg.copy()
            cfg['n_inverse_steps'] = extra_scene_step
            code, _, _, loss_decoder, loss_dict_decoder, out_rgbs, target_rgbs = self.inverse_code(
                decoder, cond_imgs, cond_rays_o, cond_rays_d, dt_gamma=dt_gamma, cfg=cfg,
                code_=code_list_,
                density_grid=density_grid,
                density_bitfield=density_bitfield,
                code_optimizer=code_optimizers,
                prior_grad=prior_grad)
            for k, v in loss_dict_decoder.items():
                log_vars.update({k: float(v)})
        else:
            prior_grad = None

        if 'decoder' in optimizer or len(code_optimizers) > 0:
            if len(code_optimizers) > 0:
                code = self.code_activation(torch.stack(code_list_, dim=0))

            loss_decoder, log_vars_decoder, out_rgbs, target_rgbs = self.loss_decoder(
                decoder, code, density_bitfield, cond_rays_o, cond_rays_d,
                cond_imgs, dt_gamma, cfg=self.train_cfg,
                update_extra_state=self.update_extra_iters,
                extra_args=(density_grid, density_bitfield, 0),
                extra_kwargs=dict(
                    density_thresh=self.train_cfg['density_thresh']
                ) if 'density_thresh' in self.train_cfg else dict())
            log_vars.update(log_vars_decoder)

            if prior_grad is not None:
                for code_, prior_grad_single in zip(code_list_, prior_grad):
                    code_.grad.copy_(prior_grad_single)
            loss_decoder.backward()

            if 'decoder' in optimizer:
                if self.train_cfg.get('decoder_grad_clip', 0.0) > 0.0:
                    decoder_grad_norm = torch.nn.utils.clip_grad_norm_(
                        decoder.parameters(), self.train_cfg['decoder_grad_clip'])
                    log_vars.update(decoder_grad_norm=float(decoder_grad_norm))
                optimizer['decoder'].step()
            for code_optimizer in code_optimizers:
                code_optimizer.step()

            # ==== save cache ====
            self.save_cache(
                code_list_, code_optimizers,
                density_grid, density_bitfield, data['scene_id'], data['scene_name'])

            # ==== evaluate reconstruction ====
            with torch.no_grad():
                if len(code_optimizers) > 0:
                    self.mean_ema_update(code)
                train_psnr = eval_psnr(out_rgbs, target_rgbs)
                code_rms = code.square().flatten(1).mean().sqrt()
                log_vars.update(train_psnr=float(train_psnr.mean()),
                                code_rms=float(code_rms.mean()))
                if 'test_imgs' in data and data['test_imgs'] is not None:
                    log_vars.update(self.eval_and_viz(
                        data, self.decoder, code, density_bitfield, cfg=self.train_cfg)[0])
                if self.train_cfg.get('debug_viz_step') and running_status is not None:
                    data['test_imgs'] = data['cond_imgs'][:, :1]
                    data['test_poses'] = data['cond_poses'][:, :1]
                    data['test_intrinsics'] = data['cond_intrinsics'][:, :1]
                    no_vizdir_cfg = dict(self.train_cfg)
                    no_vizdir_cfg['viz_dir'] = None
                    log_vars.update(super().eval_and_viz(
                        data, self.decoder, code, density_bitfield, cfg=no_vizdir_cfg)[0])
                    if running_status.get('iteration') != 1 and running_status.get('iteration') % self.train_cfg.get('debug_viz_step') == 1:
                        rank, ws = get_dist_info()
                        if rank == 0:
                            if os.path.exists(self.train_cfg.get('viz_dir')):
                                for f in os.listdir(self.train_cfg.get('viz_dir')):
                                    os.remove(os.path.join(self.train_cfg.get('viz_dir'), f))
                        if ws > 1:
                            dist.barrier()
                        super().eval_and_viz(data, self.decoder, code, density_bitfield, cfg=self.train_cfg)

        # ==== outputs ====
        if 'decoder' in optimizer or len(code_optimizers) > 0:
            log_vars.update(loss_decoder=float(loss_decoder))
        outputs_dict = dict(
            log_vars=log_vars, num_samples=num_scenes)

        return outputs_dict

    def val_text(self, data, show_pbar=False, **kwargs):
        diffusion = self.diffusion_ema if self.diffusion_use_ema else self.diffusion
        decoder = self.decoder_ema if self.decoder_use_ema else self.decoder

        num_batches = len(data['scene_id'])
        noise = data.get('noise', None)
        if noise is None:
            noise = torch.randn(
                (num_batches, *self.code_size), device=get_module_device(self))

        with torch.autocast(
                device_type='cuda',
                enabled=self.autocast_dtype is not None,
                dtype=getattr(torch, self.autocast_dtype) if self.autocast_dtype is not None else None):
            dtype = next(diffusion.parameters()).dtype
            code_out = diffusion(
                self.code_diff_pr(noise).to(dtype), return_loss=False,
                prompts=data['prompts'], neg_prompts=data.get('neg_prompts', None),
                show_pbar=show_pbar, **kwargs)
        code = self.code_diff_pr_inv(code_out)
        density_grid, density_bitfield = self.get_density(decoder, code, cfg=self.test_cfg)
        return code, density_grid, density_bitfield

    def val_guide(self, data, **kwargs):
        device = get_module_device(self)
        diffusion = self.diffusion_ema if self.diffusion_use_ema else self.diffusion
        decoder = self.decoder_ema if self.decoder_use_ema else self.decoder

        cond_imgs = data['cond_imgs']  # (num_scenes, num_imgs, h, w, 3)
        cond_intrinsics = data['cond_intrinsics']  # (num_scenes, num_imgs, 4), in [fx, fy, cx, cy]
        cond_poses = data['cond_poses']

        num_scenes, num_imgs, h, w, _ = cond_imgs.size()
        # (num_scenes, num_imgs, h, w, 3)
        cond_rays_o, cond_rays_d = get_cam_rays(cond_poses, cond_intrinsics, h, w)
        dt_gamma_scale = self.test_cfg.get('dt_gamma_scale', 0.0)
        # (num_scenes,)
        dt_gamma = dt_gamma_scale / cond_intrinsics[..., :2].mean(dim=(-2, -1))

        if self.image_cond:
            concat_cond = cond_imgs.permute(0, 1, 4, 2, 3)  # (num_scenes, num_imgs, 3, h, w)
            if num_imgs > 1:
                cond_inds = torch.stack([torch.randperm(num_imgs, device=device) for _ in range(num_scenes)], dim=0)
                scene_arange = torch.arange(num_scenes, device=device)[:, None]
                concat_cond = concat_cond[scene_arange, cond_inds]  # (num_scenes, num_imgs, 3, h, w)
            diff_image_size = rgetattr(diffusion, 'denoising.image_size')
            assert diff_image_size[0] % concat_cond.size(-2) == 0
            assert diff_image_size[1] % concat_cond.size(-1) == 0
            concat_cond = concat_cond.tile((diff_image_size[0] // concat_cond.size(-2),
                                            diff_image_size[1] // concat_cond.size(-1)))
        else:
            concat_cond = None

        decoder_training_prev = decoder.training
        decoder.train(True)

        with module_requires_grad(diffusion, False), module_requires_grad(decoder, False):
            n_inverse_rays = self.test_cfg.get('n_inverse_rays', 4096)
            raybatch_inds, num_raybatch = self.get_raybatch_inds(cond_imgs, n_inverse_rays)

            density_grid = torch.zeros((num_scenes, self.grid_size ** 3), device=device)
            density_bitfield = torch.zeros((num_scenes, self.grid_size ** 3 // 8), dtype=torch.uint8, device=device)
            inverse_step_id = torch.zeros((1, ), dtype=torch.long, device=device)

            def grad_guide_fn(x_0_pred):
                code_pred = self.code_diff_pr_inv(x_0_pred)
                inds = raybatch_inds[inverse_step_id % num_raybatch] if raybatch_inds is not None else None
                rays_o, rays_d, target_rgbs = self.ray_sample(
                    cond_rays_o, cond_rays_d, cond_imgs, n_inverse_rays, sample_inds=inds)
                _, loss, _ = self.loss(
                    decoder, code_pred, density_bitfield,
                    target_rgbs, rays_o, rays_d, dt_gamma,
                    scale_num_ray=target_rgbs.numel() // (num_scenes * 3),  # actual n_samples
                    cfg=self.test_cfg,
                    update_extra_state=self.update_extra_iters,
                    extra_args=(density_grid, density_bitfield, 0),
                    extra_kwargs=dict(
                        density_thresh=self.train_cfg['density_thresh']
                    ) if 'density_thresh' in self.train_cfg else dict())
                inverse_step_id[:] += 1
                return loss * num_scenes

            noise = data.get('noise', None)
            if noise is None:
                noise = torch.randn(
                    (num_scenes, *self.code_size), device=get_module_device(self))

            with torch.autocast(
                    device_type='cuda',
                    enabled=self.autocast_dtype is not None,
                    dtype=getattr(torch, self.autocast_dtype) if self.autocast_dtype is not None else None):
                dtype = next(diffusion.parameters()).dtype
                code = diffusion(
                    self.code_diff_pr(noise).to(dtype), return_loss=False,
                    prompts=data['prompts'], neg_prompts=data.get('neg_prompts', None),
                    grad_guide_fn=grad_guide_fn, concat_cond=concat_cond, **kwargs)

        decoder.train(decoder_training_prev)

        return self.code_diff_pr_inv(code), density_grid, density_bitfield

    def val_optim(self, data, code_=None,
                  density_grid=None, density_bitfield=None, show_pbar=False, **kwargs):
        device = get_module_device(self)
        decoder = self.decoder_ema if self.decoder_use_ema else self.decoder
        diffusion = self.diffusion_ema if self.diffusion_use_ema else self.diffusion

        cond_imgs = data['cond_imgs']  # (num_scenes, num_imgs, h, w, 3)
        cond_intrinsics = data['cond_intrinsics']  # (num_scenes, num_imgs, 4), in [fx, fy, cx, cy]
        cond_poses = data['cond_poses']

        num_scenes, num_imgs, h, w, _ = cond_imgs.size()
        # (num_scenes, num_imgs, h, w, 3)
        cond_rays_o, cond_rays_d = get_cam_rays(cond_poses, cond_intrinsics, h, w)
        dt_gamma_scale = self.test_cfg.get('dt_gamma_scale', 0.0)
        # (num_scenes,)
        dt_gamma = dt_gamma_scale / cond_intrinsics[..., :2].mean(dim=(-2, -1))

        if self.image_cond:
            concat_cond = cond_imgs.permute(0, 1, 4, 2, 3)  # (num_scenes, num_imgs, 3, h, w)
            if num_imgs > 1:
                cond_inds = torch.stack([torch.randperm(num_imgs, device=device) for _ in range(num_scenes)], dim=0)
                scene_arange = torch.arange(num_scenes, device=device)[:, None]
                concat_cond = concat_cond[scene_arange, cond_inds]  # (num_scenes, num_imgs, 3, h, w)
            diff_image_size = rgetattr(diffusion, 'denoising.image_size')
            assert diff_image_size[0] % concat_cond.size(-2) == 0
            assert diff_image_size[1] % concat_cond.size(-1) == 0
            concat_cond = concat_cond.tile((diff_image_size[0] // concat_cond.size(-2),
                                            diff_image_size[1] // concat_cond.size(-1)))
        else:
            concat_cond = None

        decoder_training_prev = decoder.training
        decoder.train(True)

        extra_scene_step = self.test_cfg.get('extra_scene_step', 0)
        n_inverse_steps = self.test_cfg.get('n_inverse_steps', 100)
        assert n_inverse_steps > 0
        if show_pbar:
            pbar = mmcv.ProgressBar(n_inverse_steps)

        with module_requires_grad(diffusion, False), module_requires_grad(decoder, False), torch.enable_grad():
            if code_ is None:
                code_ = self.get_init_code_(num_scenes, cond_imgs.device)
            else:
                code_.requires_grad_(True)
            if density_grid is None:
                density_grid = self.get_init_density_grid(num_scenes, cond_imgs.device)
            if density_bitfield is None:
                density_bitfield = self.get_init_density_bitfield(num_scenes, cond_imgs.device)
            code_optimizer = self.build_optimizer(code_, self.test_cfg)
            code_scheduler = self.build_scheduler(code_optimizer, self.test_cfg)

            for inverse_step_id in range(n_inverse_steps):
                code_optimizer.zero_grad()
                code = self.code_activation(code_)
                with torch.autocast(
                        device_type='cuda',
                        enabled=self.autocast_dtype is not None,
                        dtype=getattr(torch, self.autocast_dtype) if self.autocast_dtype is not None else None):
                    dtype = next(diffusion.parameters()).dtype
                    loss, log_vars = diffusion(
                        self.code_diff_pr(code).to(dtype), return_loss=True,
                        prompts=data['prompts'], neg_prompts=data.get('neg_prompts', None),
                        concat_cond=concat_cond[:, inverse_step_id % num_imgs] if concat_cond is not None else None,
                        x_t_detach=self.test_cfg.get('x_t_detach', False), cfg=self.test_cfg, **kwargs)
                loss.backward()

                if extra_scene_step > 0:
                    prior_grad = code_.grad.data.clone()
                    cfg = self.test_cfg.copy()
                    cfg['n_inverse_steps'] = extra_scene_step + 1
                    self.inverse_code(
                        decoder, cond_imgs, cond_rays_o, cond_rays_d, dt_gamma=dt_gamma, cfg=cfg,
                        code_=code_,
                        density_grid=density_grid,
                        density_bitfield=density_bitfield,
                        code_optimizer=code_optimizer,
                        code_scheduler=code_scheduler,
                        prior_grad=prior_grad)
                else:  # avoid cloning the grad
                    code = self.code_activation(code_)
                    loss_decoder, log_vars_decoder, out_rgbs, target_rgbs = self.loss_decoder(
                        decoder, code, density_bitfield, cond_rays_o, cond_rays_d,
                        cond_imgs, dt_gamma, cfg=self.test_cfg)
                    loss_decoder.backward()
                    code_optimizer.step()
                    if code_scheduler is not None:
                        code_scheduler.step()

                if show_pbar:
                    pbar.update()

        decoder.train(decoder_training_prev)

        return self.code_activation(code_), density_grid, density_bitfield

    def val_step(self, data, viz_dir=None, viz_dir_guide=None, **kwargs):
        decoder = self.decoder_ema if self.decoder_use_ema else self.decoder

        with torch.no_grad():
            if 'code' in data:
                code, density_grid, density_bitfield = self.load_scene(
                    data, load_density=True)
            elif 'cond_imgs' in data:
                cond_mode = self.test_cfg.get('cond_mode', 'guide')
                if cond_mode == 'guide':
                    code, density_grid, density_bitfield = self.val_guide(data, **kwargs)
                elif cond_mode == 'optim':
                    code, density_grid, density_bitfield = self.val_optim(data, **kwargs)
                elif cond_mode == 'guide_optim':
                    code, density_grid, density_bitfield = self.val_guide(data, **kwargs)
                    if viz_dir_guide is not None and 'test_poses' in data:
                        self.eval_and_viz(
                            data, decoder, code, density_bitfield,
                            viz_dir=viz_dir_guide, cfg=self.test_cfg)
                    code, density_grid, density_bitfield = self.val_optim(
                        data,
                        code_=self.code_activation.inverse(code).requires_grad_(True),
                        density_grid=density_grid,
                        density_bitfield=density_bitfield,
                        **kwargs)
                else:
                    raise AttributeError
            else:
                code, density_grid, density_bitfield = self.val_text(data, **kwargs)

            # ==== evaluate reconstruction ====
            torch.cuda.empty_cache()
            if 'test_poses' in data:
                log_vars, pred_imgs = self.eval_and_viz(
                    data, decoder, code, density_bitfield,
                    viz_dir=viz_dir, cfg=self.test_cfg)
            else:
                log_vars = dict()
                pred_imgs = None
                if viz_dir is None:
                    viz_dir = self.test_cfg.get('viz_dir', None)
                if viz_dir is not None:
                    if isinstance(decoder, DistributedDataParallel):
                        decoder = decoder.module
                    decoder.visualize(
                        code, data['scene_name'],
                        viz_dir, code_range=self.test_cfg.get('clip_range', [-1, 1]))

        # ==== save 3D code ====
        save_dir = self.test_cfg.get('save_dir', None)
        if save_dir is not None:
            self.save_scene(save_dir, code, density_grid, density_bitfield, data['scene_name'])
            save_mesh = self.test_cfg.get('save_mesh', False)
            if save_mesh:
                mesh_resolution = self.test_cfg.get('mesh_resolution', 256)
                mesh_threshold = self.test_cfg.get('mesh_threshold', 10)
                self.save_mesh(save_dir, decoder, code, data['scene_name'], mesh_resolution, mesh_threshold)

        # ==== outputs ====
        outputs_dict = dict(
            log_vars=log_vars,
            num_samples=len(data['scene_name']),
            pred_imgs=pred_imgs)

        return outputs_dict

    def eval_and_viz(self, data, decoder, code, density_bitfield, viz_dir=None, cfg=dict()):
        data['scene_name'] = [
            scene_name_single + '_' + prompt_single
            for scene_name_single, prompt_single in zip(data['scene_name'], data['prompts'])]
        ret = super(DiffusionNeRFText, self).eval_and_viz(
            data, decoder, code, density_bitfield, viz_dir=viz_dir, cfg=cfg)
        if viz_dir is None:
            viz_dir = cfg.get('viz_dir', None)
        if viz_dir is not None and self.decoder_debug is not None:
            with torch.autocast(
                    device_type='cuda',
                    enabled=self.autocast_dtype is not None,
                    dtype=getattr(torch, self.autocast_dtype) if self.autocast_dtype is not None else None):
                self.decoder_debug.visualize(
                    self.code_diff_pr(code),
                    data['scene_name'],
                    viz_dir)
        return ret
