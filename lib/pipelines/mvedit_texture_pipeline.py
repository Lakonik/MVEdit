import numpy as np
import PIL
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
import torch.distributed

from tqdm import tqdm
from typing import Dict, Optional, Union, List, Tuple, Callable
from PIL import Image
from transformers import CLIPTextModel, CLIPTokenizer
from diffusers import AutoencoderKL, UNet2DConditionModel, StableDiffusionControlNetPipeline, \
    DPMSolverMultistepScheduler, DDIMScheduler
from diffusers.models import ControlNetModel
from diffusers.pipelines.controlnet import MultiControlNetModel
from diffusers.schedulers import KarrasDiffusionSchedulers
from copy import deepcopy

from lib.core import depth_to_normal, get_noise_scales
from lib.models import SRVGGNetCompact
from lib.models.autoencoders.base_nerf import BaseNeRF, get_ray_directions
from lib.models.autoencoders.base_mesh import Mesh
from lib.models.decoders.base_mesh_renderer import MeshRenderer
from lib.models.architecture.ip_adapter import IPAdapter
from lib.models.architecture.joint_attn import apply_cross_image_attn_proc
from .mvedit_3d_pipeline import MVEdit3DPipeline, join_prompts, \
    default_tile_weight, default_depth_weight, default_blend_weight, plot_weights, normalize_depth


def default_lr_schedule(progress, start_lr=0.01, end_lr=0.01):
    return start_lr - (start_lr - end_lr) * progress


def default_patch_rgb_weight(progress, start_weight=0.1, end_weight=0.1):
    return start_weight + (end_weight - start_weight) * progress


def default_max_num_views(progress, start_num=32, end_num=7, power=2):
    return (start_num - end_num) * (1 - progress) ** power + end_num


def camera_dense_weighting(intrinsics, intrinsics_size, render_size, alphas, depths):
    directions = get_ray_directions(
        render_size, render_size,
        intrinsics[None] * (render_size / intrinsics_size),
        norm=False, device=intrinsics.device).squeeze(0)
    normals = depth_to_normal(depths, directions, format='opencv') * 2 - 1
    cos = (normals[..., None, :] @ F.normalize(directions[..., :, None], dim=-2)).squeeze(-1)
    alphas_eroded = -F.avg_pool2d(F.max_pool2d(
        -alphas.permute(0, 3, 1, 2), 5, stride=1, padding=2), 5, stride=1, padding=2).permute(0, 2, 3, 1)
    return alphas_eroded * (-cos).clamp(min=0)


class MVEditTexturePipeline(MVEdit3DPipeline):

    _optional_components = []

    def __init__(
            self,
            vae: AutoencoderKL,
            text_encoder: CLIPTextModel,
            tokenizer: CLIPTokenizer,
            unet: UNet2DConditionModel,
            controlnet: Union[List[ControlNetModel], Tuple[ControlNetModel]],  # must be tile + depth
            scheduler: KarrasDiffusionSchedulers,
            nerf: BaseNeRF,
            mesh_renderer: MeshRenderer):
        super(StableDiffusionControlNetPipeline, self).__init__()
        assert isinstance(controlnet, (list, tuple))
        controlnet = MultiControlNetModel(controlnet)
        self.register_modules(
            vae=vae,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            unet=unet,
            controlnet=controlnet,
            scheduler=scheduler,
            nerf=nerf,
            mesh_renderer=mesh_renderer)
        self.clip_img_size = 224
        self.clip_img_mean = [0.48145466, 0.4578275, 0.40821073]
        self.clip_img_std = [0.26862954, 0.26130258, 0.27577711]
        self.cross_image_attn_enabled = False
        self.bg_color = 0.5

    @staticmethod
    def save_tiled_viz(out_dir, i, images, tgt_images=None):
        for img_id, image in enumerate(images.permute(0, 2, 3, 1)):
            if tgt_images is not None:
                image = torch.cat([tgt_images[0, img_id], image], dim=1)
            plt.imsave(out_dir + '/{:03d}_{:03d}.png'.format(i, img_id),
                       (image.clamp(min=0, max=1) * 255).to(torch.uint8).cpu().numpy())

    def texture_optim(
            self, tgt_images,  # input images
            optimizer, lr, inverse_steps, render_bs, patch_bs,  # optimization settings
            patch_rgb_weight,  # loss weights
            nerf_code, in_mesh,  # mesh model
            render_size, intrinsics, intrinsics_size, camera_poses, cam_weights_dense, patch_size):
        device = self.unet.device

        decoder_training_prev = self.nerf.decoder.training
        self.nerf.decoder.train(True)

        with torch.enable_grad():
            optimizer.param_groups[0]['lr'] = lr

            camera_perm = torch.randperm(camera_poses.size(0), device=device)
            pose_batches = camera_poses[camera_perm].split(render_bs, dim=0)
            intrinsics_batches = intrinsics[camera_perm].split(render_bs, dim=0)
            tgt_image_batches = tgt_images.squeeze(0)[camera_perm].split(render_bs, dim=0)
            num_pose_batches = len(pose_batches)
            cam_weights_batches = cam_weights_dense[camera_perm].split(render_bs, dim=0)

            pixel_rgb_loss_ = []
            patch_rgb_loss_ = []
            for inverse_step_id in range(inverse_steps):
                pose_batch = pose_batches[inverse_step_id % num_pose_batches]
                intrinsics_batch = intrinsics_batches[inverse_step_id % num_pose_batches]
                target_rgbs = tgt_image_batches[inverse_step_id % num_pose_batches]
                target_w = cam_weights_batches[inverse_step_id % num_pose_batches]

                render_out = self.mesh_renderer(
                    [in_mesh],
                    pose_batch[None],
                    intrinsics_batch[None] * (render_size / intrinsics_size),
                    render_size, render_size,
                    self.make_nerf_albedo_shading_fun(nerf_code))
                out_rgbs = (render_out['rgba'][..., :3]
                            + (1 - render_out['rgba'][..., 3:].clamp(min=1e-3)) * self.bg_color).squeeze(0)
                loss = pixel_rgb_loss = self.nerf.pixel_loss(
                    out_rgbs.reshape(target_rgbs.size()), target_rgbs,
                    weight=target_w) * 2
                pixel_rgb_loss_.append(pixel_rgb_loss.item())

                if patch_rgb_weight > 0:
                    out_rgb_patch = out_rgbs.reshape(
                        -1, render_size // patch_size, patch_size, render_size // patch_size, patch_size, 3
                    ).permute(0, 1, 3, 5, 2, 4).reshape(-1, 3, patch_size, patch_size)
                    tgt_rgb_patch = target_rgbs.reshape(
                        -1, render_size // patch_size, patch_size, render_size // patch_size, patch_size, 3
                    ).permute(0, 1, 3, 5, 2, 4).reshape(-1, 3, patch_size, patch_size)
                    target_w_patch = target_w.reshape(
                        -1, render_size // patch_size, patch_size, render_size // patch_size, patch_size, 1
                    ).permute(0, 1, 3, 5, 2, 4).reshape(-1, 1, patch_size, patch_size)
                    patch_batch = torch.randperm(out_rgb_patch.size(0), device=device)[:patch_bs]
                    target_w_patch_ = target_w_patch[patch_batch].amax(dim=(1, 2, 3))
                    patch_rgb_loss = self.nerf.patch_loss(
                        out_rgb_patch[patch_batch],  # (num_patches, 3, h, w)
                        tgt_rgb_patch[patch_batch],  # (num_patches, 3, h, w)
                        weight=target_w_patch_
                    ) * patch_rgb_weight
                    loss = loss + patch_rgb_loss
                    patch_rgb_loss_.append(patch_rgb_loss.item())

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            print(f'\npixel_rgb_loss: {np.mean(pixel_rgb_loss_):.4f}, '
                  f'patch_rgb_loss: {np.mean(patch_rgb_loss_):.4f}')

        self.nerf.decoder.train(decoder_training_prev)

    @torch.no_grad()
    def __call__(
            self,
            prompt: Union[str, List[str]] = '',
            negative_prompt: Union[str, List[str]] = '',
            in_model: Optional[Union[str, Mesh]] = None,
            ingp_states: Optional[Dict] = None,
            init_images: Optional[List[PIL.Image.Image]] = None,
            cond_images: Optional[List[PIL.Image.Image]] = None,
            extra_control_images: Optional[List[List[PIL.Image.Image]]] = None,
            nerf_code: Optional[torch.tensor] = None,
            camera_poses: List[np.ndarray] = None,
            intrinsics: torch.Tensor = None,  # (num_cameras, 4)
            intrinsics_size: Union[int, float] = 256,
            use_reference: bool = True,
            cam_weights: List[float] = None,
            keep_views: List[int] = None,
            guidance_scale: float = 7,
            num_inference_steps: int = 24,
            denoising_strength: Optional[float] = 0.6,
            patch_size: int = 512,
            patch_bs: int = 1,
            diff_bs: int = 12,
            render_bs: int = 8,
            n_inverse_steps: int = 64,
            init_inverse_steps: int = 64,
            ip_adapter: Optional[IPAdapter] = None,
            tile_weight: Callable = default_tile_weight,
            depth_weight: Callable = default_depth_weight,
            blend_weight: Callable = default_blend_weight,
            lr_schedule: Callable = default_lr_schedule,
            max_num_views: Callable = default_max_num_views,
            patch_rgb_weight: Callable = default_patch_rgb_weight,
            ablation_nodiff: bool = False,
            debug: bool = False,
            out_dir: Optional[str] = None,
            save_interval: Optional[int] = None,
            save_all_interval: Optional[int] = None,
            default_prompt='high-res, best quality, 3d model, cg, rendering, extremely detailed, '
                           'photorealistic, RAW photo, 4k uhd, dslr, high quality',
            default_neg_prompt='depth of field, out of focus, lowres, worst quality, low quality, drawing, '
                               'illustration, painting, blurry, jpeg artifacts, macro',
            bake_texture=True,
            force_auto_uv=False,
            map_size=2048,
            prog_bar=tqdm):

        if not self.cross_image_attn_enabled:
            apply_cross_image_attn_proc(self.unet)
            self.cross_image_attn_enabled = True

        device = self.unet.device
        torch.set_grad_enabled(False)

        render_size = 512
        
        # ================= Pre-process inputs =================
        if debug:
            plot_weights(tile_weight, blend_weight, depth_weight, out_dir)

        if isinstance(camera_poses, torch.Tensor):
            camera_poses = camera_poses.to(device=device, dtype=torch.float32)
        else:
            camera_poses = torch.from_numpy(np.stack(camera_poses, axis=0)).to(device=device, dtype=torch.float32)
        num_cameras = len(camera_poses)

        if intrinsics.dim() == 1:
            intrinsics = intrinsics[None].expand(num_cameras, -1)

        assert in_model is not None
        in_mesh, images, alphas, depths = self.load_init_mesh(
            in_model, camera_poses, intrinsics, intrinsics_size, render_bs)
        in_images = images.permute(0, 3, 1, 2).to(dtype=self.unet.dtype)
        ctrl_depths = normalize_depth(depths, alphas).to(self.unet.dtype).unsqueeze(1).repeat(1, 3, 1, 1)
        if init_images is not None:
            in_images, _ = self.load_init_images(init_images, ret_masks=False)

        if cam_weights is None:
            cam_weights = [1.0] * num_cameras
        cam_weights = camera_poses.new_tensor(cam_weights)
        cam_weights_dense = cam_weights[:, None, None, None] * camera_dense_weighting(
            intrinsics, intrinsics_size, render_size, alphas, depths)

        if not isinstance(prompt, list):
            prompt = [prompt] * num_cameras
        if not isinstance(negative_prompt, list):
            negative_prompt = [negative_prompt] * num_cameras

        cond_images, extra_control_images = self.load_cond_images(in_images, cond_images, extra_control_images)

        rgb_prompt = [join_prompts(prompt_single, default_prompt) for prompt_single in prompt]
        rgb_negative_prompt = [join_prompts(negative_prompt_single, default_neg_prompt)
                               for negative_prompt_single in negative_prompt]

        # ================= Initialize scheduler =================
        # assert isinstance(self.scheduler, (DPMSolverMultistepScheduler, DDIMScheduler))
        betas = self.scheduler.betas.cpu().numpy()
        alphas = 1.0 - betas
        alphas_bar = np.cumproduct(alphas, axis=0)

        self.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = self.scheduler.timesteps
        if denoising_strength is not None:
            timesteps = timesteps[min(
                int(round(len(timesteps) * (1 - denoising_strength) / self.scheduler.order)) * self.scheduler.order,
                len(timesteps) - 1):]
        print(f"Timesteps = {timesteps}")

        # ================= Initilize embeds and latents =================
        if not ablation_nodiff:  # for ip_adapter and cross_image_attn, in_images are used when cond_images is None
            prompt_embeds = self.get_prompt_embeds(
                in_images, rgb_prompt, rgb_negative_prompt, ip_adapter=ip_adapter, cond_images=cond_images)

            init_latents = []
            for in_images_batch in in_images.split(diff_bs, dim=0):
                init_latents.append(self.vae.encode(
                    in_images_batch * 2 - 1).latent_dist.sample() * self.vae.config.scaling_factor)
            init_latents = torch.cat(init_latents, dim=0)
            if use_reference:
                if cond_images is None:
                    ref_latents = init_latents
                else:
                    ref_images = []
                    for cond_image in cond_images:
                        ref_images.append(F.interpolate(cond_image, size=(512, 512), mode='bilinear'))
                    ref_images = torch.cat(ref_images, dim=0)
                    ref_latents = []
                    for ref_images_batch in ref_images.split(diff_bs, dim=0):
                        ref_latents.append(self.vae.encode(
                            ref_images_batch * 2 - 1).latent_dist.sample() * self.vae.config.scaling_factor)
                    ref_latents = torch.cat(ref_latents, dim=0)
            else:
                ref_latents = None

        # ================= Initialize texture field =================
        if self.nerf.decoder.state_dict_bak is None:
            self.nerf.decoder.backup_state_dict()
        if ingp_states is not None:
            self.nerf.decoder.load_state_dict(
                ingp_states if isinstance(ingp_states, dict) else
                torch.load(ingp_states, map_location='cpu'), strict=False)

        optimizer = torch.optim.Adam(self.nerf.decoder.parameters(), lr=0.01)
        if nerf_code is None:
            nerf_code = [None]

        # ================= Ancestral sampling loop =================
        for i, t in enumerate(prog_bar([None] + list(timesteps))):
            progress = i / len(timesteps)

            # ================= Update cameras =================
            if t is None:  # initialization
                if keep_views is None:
                    keep_views = []
                reorder_ids = keep_views
                num_keep_views = len(keep_views)
                for cam_id in range(num_cameras):
                    if cam_id not in keep_views:
                        reorder_ids.append(cam_id)
                reorder_ids = torch.tensor(reorder_ids, device=device)
                in_images = in_images[reorder_ids]
                ctrl_depths = ctrl_depths[reorder_ids]
                camera_poses = camera_poses[reorder_ids]
                intrinsics = intrinsics[reorder_ids]
                cam_weights_dense = cam_weights_dense[reorder_ids]
                extra_control_images = [
                    extra_control_image[reorder_ids] for extra_control_image in extra_control_images]
                if not ablation_nodiff:
                    init_latents = init_latents[reorder_ids]
                    ref_latents = ref_latents[reorder_ids] if use_reference else None
                    reorder_ids_ = torch.cat([reorder_ids, reorder_ids + num_cameras], dim=0)
                    prompt_embeds = prompt_embeds[reorder_ids_]
                cam_positions = camera_poses[:, :3, 3]  # (num_cameras, 3)
                # (num_cameras, num_cameras)
                dists = torch.cdist(
                    cam_positions[None], cam_positions[None], compute_mode='donot_use_mm_for_euclid_dist'
                ).squeeze(0) * cam_weights[:, None]
                dists = dists + 999999 * torch.eye(len(dists), dtype=dists.dtype, device=device)

            else:
                max_num_cameras = max(int(round(max_num_views(progress))), num_keep_views)
                if max_num_cameras < num_cameras:
                    keep_ids = torch.arange(num_cameras, device=device)
                    pixel_dist = (ctrl_images - in_images).square().flatten(1).mean(dim=1)
                    for _ in range(num_cameras - max_num_cameras):
                        view_importance = dists[num_keep_views:].amin(dim=1) - pixel_dist[num_keep_views:]
                        remove_id = view_importance.argmin() + num_keep_views
                        keep_mask = torch.arange(len(keep_ids), device=device) != remove_id
                        keep_ids = keep_ids[keep_mask]
                        dists = dists[keep_mask][:, keep_mask]
                        pixel_dist = pixel_dist[keep_mask]
                    in_images = in_images[keep_ids]
                    camera_poses = camera_poses[keep_ids]
                    intrinsics = intrinsics[keep_ids]
                    cam_weights_dense = cam_weights_dense[keep_ids]
                    ctrl_images = ctrl_images[keep_ids]
                    ctrl_depths = ctrl_depths[keep_ids]
                    extra_control_images = [
                        extra_control_image[keep_ids] for extra_control_image in extra_control_images]
                    if not ablation_nodiff:
                        latents = latents[keep_ids]
                        ref_latents = ref_latents[keep_ids] if use_reference else None
                        keep_ids_ = torch.cat([keep_ids, keep_ids + num_cameras], dim=0)
                        prompt_embeds = prompt_embeds[keep_ids_]
                    if isinstance(self.scheduler, DPMSolverMultistepScheduler):
                        self.scheduler.model_outputs = [output[keep_ids] if output is not None else None for output in
                                                        self.scheduler.model_outputs]
                    num_cameras = max_num_cameras

            # ================= Denoising step =================
            if t is not None and not ablation_nodiff:
                latents_scaled = self.scheduler.scale_model_input(latents, t)
                if use_reference:
                    latent_batches = latents_scaled[:, :, -64:].split(diff_bs, dim=0) \
                                     + latents_scaled.split(diff_bs, dim=0)
                    prompt_embeds_batches = prompt_embeds[:num_cameras].split(diff_bs, dim=0) \
                                            + prompt_embeds[-num_cameras:].split(diff_bs, dim=0)
                    ctrl_images_batches = ctrl_images.split(diff_bs, dim=0) * 2
                    ctrl_depths_batches = ctrl_depths.split(diff_bs, dim=0) * 2
                    extra_control_batches = [extra_control_image_group.split(diff_bs, dim=0) * 2
                                             for extra_control_image_group in extra_control_images]
                else:
                    latent_batches = torch.cat([latents_scaled] * 2, dim=0).split(diff_bs, dim=0)
                    prompt_embeds_batches = prompt_embeds.split(diff_bs, dim=0)
                    ctrl_images_batches = torch.cat([ctrl_images] * 2, dim=0).split(diff_bs, dim=0)
                    ctrl_depths_batches = torch.cat([ctrl_depths] * 2, dim=0).split(diff_bs, dim=0)
                    extra_control_batches = [torch.cat([extra_control_image_group] * 2, dim=0).split(diff_bs, dim=0)
                                             for extra_control_image_group in extra_control_images]

                noise_pred = self.get_noise_pred(
                    latent_batches, prompt_embeds_batches, ctrl_images_batches, ctrl_depths_batches,
                    t, progress, tile_weight, depth_weight, guidance_scale,
                    extra_control_batches=extra_control_batches)

                sqrt_alpha_bar_t, sqrt_one_minus_alpha_bar_t = get_noise_scales(
                    alphas_bar, t, self.scheduler.num_train_timesteps, dtype=noise_pred.dtype)
                pred_original_sample = (
                    (latents_scaled[:, :, -64:] - sqrt_one_minus_alpha_bar_t * noise_pred)
                    / sqrt_alpha_bar_t).to(noise_pred)

                tgt_images = []
                for batch_pred_original_sample in pred_original_sample.split(diff_bs, dim=0):
                    tgt_images.append(self.vae.decode(
                        batch_pred_original_sample / self.vae.config.scaling_factor, return_dict=False)[0])
                tgt_images = torch.cat(tgt_images)
                tgt_images = (tgt_images / 2 + 0.5).clamp(min=0, max=1).permute(0, 2, 3, 1)

                tgt_images = tgt_images[None].to(torch.float32)

                if save_all_interval is not None and (i % save_all_interval == 0 or i == len(timesteps)):
                    self.save_all_viz(out_dir, i, num_keep_views, tgt_images, latents_scaled,
                                      ctrl_images, diff_bs=diff_bs)

            else:
                tgt_images = in_images.permute(0, 2, 3, 1)[None].to(torch.float32)
                if save_all_interval is not None:
                    self.save_all_viz(out_dir, i, num_keep_views, tgt_images, diff_bs=diff_bs)

            # ================= Texture optimization =================
            if i == 0:
                inverse_steps = init_inverse_steps
            else:
                inverse_steps = n_inverse_steps
            self.texture_optim(
                tgt_images,  # input images
                optimizer, lr_schedule(progress), inverse_steps, render_bs, patch_bs,  # optimization settings
                patch_rgb_weight(progress),  # loss weights
                nerf_code, in_mesh,  # mesh model
                render_size, intrinsics, intrinsics_size, camera_poses, cam_weights_dense, patch_size)

            # ================= Mesh rendering =================
            images = []
            for pose_batch, intrinsics_batch in zip(
                    camera_poses.split(render_bs, dim=0), intrinsics.split(render_bs, dim=0)):
                render_out = self.mesh_renderer(
                    [in_mesh],
                    pose_batch[None],
                    intrinsics_batch[None] * (render_size / intrinsics_size),
                    render_size, render_size,
                    self.make_nerf_albedo_shading_fun(nerf_code))
                images.append((render_out['rgba'][..., :3]
                               + self.bg_color * (1 - render_out['rgba'][..., 3:])).squeeze(0))

            images = torch.cat(images, dim=0).to(self.unet.dtype).permute(0, 3, 1, 2).clamp(min=0, max=1)
            ctrl_images = images

            if save_interval is not None and (i % save_interval == 0 or i == len(timesteps)):
                self.save_tiled_viz(out_dir, i, images, tgt_images)

            # ================= Solver step =================
            if ablation_nodiff:
                continue

            blend_weight_t = blend_weight(timesteps[0] if t is None else t)

            if blend_weight_t > 0:
                pred_nerf_sample = []
                for batch_images in images.split(diff_bs, dim=0):
                    pred_nerf_sample.append(self.vae.encode(
                        batch_images * 2 - 1, return_dict=False)[0].mean * self.vae.config.scaling_factor)
                pred_nerf_sample = torch.cat(pred_nerf_sample, dim=0)

            if t is not None:
                if blend_weight_t > 0:
                    pred_nerf_noise = (latents_scaled[:, :, -64:] - pred_nerf_sample * sqrt_alpha_bar_t
                                       ) / sqrt_one_minus_alpha_bar_t
                    merged_noise = pred_nerf_noise * blend_weight_t + noise_pred * (1 - blend_weight_t)
                else:
                    merged_noise = noise_pred
                if use_reference:
                    ref_noise = (latents_scaled[:, :, :64] - ref_latents * sqrt_alpha_bar_t
                                 ) / sqrt_one_minus_alpha_bar_t
                    merged_noise = torch.cat([ref_noise, merged_noise], dim=2)
                latents = self.scheduler.step(merged_noise, t, latents, return_dict=False)[0]

            else:
                if denoising_strength is None:
                    latents = torch.randn_like(init_latents[0]).expand(
                        init_latents.size(0), -1, -1, -1) * self.scheduler.init_noise_sigma
                    if use_reference:
                        ref_latents = torch.randn_like(init_latents[0]).expand(
                            init_latents.size(0), -1, -1, -1) * self.scheduler.init_noise_sigma
                        latents = torch.cat([ref_latents, latents], dim=2)
                else:
                    if blend_weight_t > 0:
                        latents = pred_nerf_sample * blend_weight_t + init_latents * (1 - blend_weight_t)
                    else:
                        latents = init_latents
                    if use_reference:
                        latents = torch.cat([ref_latents, latents], dim=2)
                    latents = self.scheduler.add_noise(latents, torch.randn_like(
                        latents[0]).expand(latents.size(0), -1, -1, -1), timesteps[0:1])

        # ================= Save results =================
        out_mesh = self.mesh_renderer.bake_xyz_shading_fun(
            [in_mesh.detach()], self.make_nerf_albedo_shading_fun(nerf_code),
            map_size=map_size, force_auto_uv=force_auto_uv
        )[0] if bake_texture else in_mesh.detach()

        output_state = deepcopy(self.nerf.decoder.state_dict())
        self.nerf.decoder.restore_state_dict()

        return out_mesh, output_state
