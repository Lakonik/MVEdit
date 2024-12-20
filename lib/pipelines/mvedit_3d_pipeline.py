import os.path as osp
import traceback
import math
import numpy as np
import PIL
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint
import torch.distributed
import torchvision.transforms.v2.functional as F_t
import open3d as o3d

from tqdm import tqdm
from typing import Dict, Optional, Union, List, Tuple, Callable
from PIL import Image
from transformers import CLIPTextModel, CLIPTokenizer
from diffusers import AutoencoderKL, UNet2DConditionModel, StableDiffusionControlNetPipeline, \
    DPMSolverMultistepScheduler, DPMSolverSDEScheduler
from diffusers.models import ControlNetModel
from diffusers.pipelines.controlnet import MultiControlNetModel
from diffusers.schedulers import KarrasDiffusionSchedulers
from copy import deepcopy
from mmgen.models.architectures.common import get_module_device

from lib.core import depth_to_normal, get_noise_scales, normalize_depth
from lib.core.utils.camera_utils import light_sampling
from lib.models import SRVGGNetCompact, TVLoss
from lib.models.autoencoders.base_nerf import BaseNeRF, get_rays, get_ray_directions
from lib.models.decoders.mesh_renderer.mesh_utils import Mesh
from lib.models.decoders.mesh_renderer.base_mesh_renderer import MeshRenderer, DMTet, laplacian_smooth_loss, normal_consistency
from lib.models.architecture.ip_adapter import IPAdapter
from lib.models.architecture.joint_attn import apply_cross_image_attn_proc, remove_cross_image_attn_proc
from lib.models.decoders.tonemapping import Tonemapping
from omnidata_modules.midas.dpt_depth import DPTDepthModel
from .utils import init_tet, highpass, do_segmentation, join_prompts, get_camera_dists, prune_cameras
from .adapter3d_mixin import Adapter3DMixin


def default_lr_multiplier(progress, progress_to_dmtet):
    return min((1 - progress) / (1 - progress_to_dmtet), 1)


def default_max_num_views(progress, progress_to_dmtet, start_num=32, mid_num=16, end_num=9, power=3):
    ratio = end_num / mid_num
    a = (start_num - mid_num) * (1 - progress) ** power + mid_num
    b = min((1 - progress) / (1 - progress_to_dmtet), 1) * (1 - ratio) + ratio
    return a * b


def default_render_size_p(progress):
    if progress <= 0.3:
        return 128
    if progress <= 0.6:
        return 256
    else:
        return 512


def default_lr_schedule(progress, start_lr=0.01, end_lr=0.005):
    return start_lr - (start_lr - end_lr) * progress


def default_patch_rgb_weight(progress, start_weight=0.3, end_weight=1.5):
    return start_weight + (end_weight - start_weight) * progress


def default_patch_normal_weight(progress, start_weight=0.0, end_weight=3.0):
    return start_weight + (end_weight - start_weight) * progress


def default_entropy_weight(progress, start_weight=0.0, end_weight=4.0):
    return start_weight - (start_weight - end_weight) * progress


def default_normal_reg_weight(progress, start_weight=4.0, end_weight=0.0):
    return start_weight - (start_weight - end_weight) * progress


class MVEdit3DPipeline(StableDiffusionControlNetPipeline, Adapter3DMixin):

    _optional_components = ['vae', 'text_encoder', 'tokenizer', 'unet', 'controlnet', 'scheduler',
                            'segmentation', 'normal_model', 'tonemapping']

    def __init__(
            self,
            vae: AutoencoderKL,
            text_encoder: CLIPTextModel,
            tokenizer: CLIPTokenizer,
            unet: UNet2DConditionModel,
            controlnet: Union[List[ControlNetModel], Tuple[ControlNetModel]],  # must be tile + depth
            scheduler: KarrasDiffusionSchedulers,
            nerf: BaseNeRF,
            mesh_renderer: MeshRenderer,
            image_enhancer: SRVGGNetCompact,
            segmentation: nn.Module,
            normal_model: DPTDepthModel,
            tonemapping: Tonemapping):
        super(StableDiffusionControlNetPipeline, self).__init__()
        if controlnet is not None:
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
            mesh_renderer=mesh_renderer,
            image_enhancer=image_enhancer,
            segmentation=segmentation,
            normal_model=normal_model,
            tonemapping=tonemapping)
        self.clip_img_size = 224
        self.clip_img_mean = [0.48145466, 0.4578275, 0.40821073]
        self.clip_img_std = [0.26862954, 0.26130258, 0.27577711]
        self.bg_color = self.nerf.bg_color
        self.normal_bg = [0.5, 0.5, 1.0]

    def to(self,
           torch_device: Optional[Union[str, torch.device]] = None,
           torch_dtype: Optional[torch.dtype] = None,
           silence_dtype_warnings: bool = False):
        if torch_device is None and torch_dtype is None:
            return self

        module_names, _ = self._get_signature_keys(self)
        modules = [getattr(self, n, None) for n in module_names]
        modules = [m for m in modules if isinstance(m, torch.nn.Module)]

        for module in modules:
            module.to(torch_device, torch_dtype)
        return self

    def load_init_nerf(self, nerf_code, density_bitfield, camera_poses, intrinsics, intrinsics_size,
                       lights, cam_lights, ambient_light, render_bs, testmode_dt_gamma_scale, diff_size=512):
        images = []
        alphas = []
        for pose_batch, intrinsics_batch, lights_batch, cam_lights_batch in zip(
                camera_poses.split(render_bs, dim=0),
                intrinsics.split(render_bs, dim=0),
                lights.split(render_bs, dim=0),
                cam_lights.split(render_bs, dim=0)):
            rgba_batch, depth_batch, normal_batch, normal_fg_batch = self.nerf.render(
                self.nerf.decoder,
                nerf_code,
                density_bitfield, diff_size, diff_size,
                intrinsics_batch[None] * (diff_size / intrinsics_size),
                pose_batch[None],
                cfg=dict(return_rgba=True, compute_normal=True, dt_gamma_scale=testmode_dt_gamma_scale),
                perturb=False,
                normal_bg=self.normal_bg)
            normal_fg_batch_opencv = torch.cat(
                [normal_fg_batch[..., :1] * 2 - 1,
                 -normal_fg_batch[..., 1:3] * 2 + 1], dim=-1)
            nerf_shading = ((cam_lights_batch[:, None, None, None, :] @ normal_fg_batch_opencv[..., :, None]
                             ).clamp(min=0) * (1 - ambient_light) + ambient_light).squeeze(-1)
            if self.tonemapping is None:
                image_batch = rgba_batch[..., :3] * nerf_shading \
                              + self.nerf.bg_color * (1 - rgba_batch[..., 3:])
            else:
                image_batch = self.tonemapping.lut(
                    self.tonemapping.inverse_lut(rgba_batch[..., :3] / rgba_batch[..., 3:].clamp(min=1e-6))
                    + nerf_shading.clamp(min=1e-6).log2()
                ) * rgba_batch[..., 3:] + self.nerf.bg_color * (1 - rgba_batch[..., 3:])
            images.append(image_batch.squeeze(0))
            alphas.append(rgba_batch[..., 3:].squeeze(0))
        images = torch.cat(images, dim=0).clamp(min=0, max=1)
        alphas = torch.cat(alphas, dim=0)
        return images, alphas

    def load_init_images(self, init_images, ret_masks=False, seg_padding=None, diff_size=512):
        """
        Args:
            init_images (List[Image.Image]): list of images to be loaded
            ret_masks (bool): whether to return masks
            seg_padding (int): padding for segmentation

        Returns:
            Tuple[torch.Tensor, torch.Tensor | None]:
                in_images: shape (N, 3, 512, 512), UNet dtype
                in_masks: shape (N, 1, 512, 512), UNet dtype
        """
        device = get_module_device(self.nerf)
        dtype = torch.float32 if self.unet is None else self.unet.dtype
        if ret_masks:
            init_images_ = []
            init_masks_ = []
            init_images_unmasked_idx_ = []
            for i, img in enumerate(init_images):
                img = np.asarray(img).astype(np.float32) / 255
                if img.shape[-1] == 4:
                    init_masks_.append(img[..., 3:])
                    img = img[..., :3] * img[..., 3:] + (1 - img[..., 3:]) * self.bg_color
                else:
                    init_masks_.append(None)
                    init_images_unmasked_idx_.append(i)
                init_images_.append(img)
            if len(init_images_unmasked_idx_) > 0:
                for idx, img in zip(init_images_unmasked_idx_,
                                    do_segmentation([init_images[i] for i in init_images_unmasked_idx_],
                                                    self.segmentation, padding=seg_padding)):
                    init_masks_[idx] = np.array(img)[..., 3:].astype(np.float32) / 255
            in_images = []
            in_masks = []
            for img, mask in zip(init_images_, init_masks_):
                img = torch.from_numpy(img).to(device=device, dtype=dtype).permute(2, 0, 1)[None]
                mask = torch.from_numpy(mask).to(device=device, dtype=dtype).permute(2, 0, 1)[None]
                if img.size(-1) < diff_size:
                    img = self.image_enhancer(img).clamp(min=0, max=1)
                in_images.append(F.interpolate(img, size=(diff_size, diff_size), mode='bilinear'))
                in_masks.append(F.interpolate(mask, size=(diff_size, diff_size), mode='bilinear'))
            in_images = torch.cat(in_images, dim=0)
            in_masks = torch.cat(in_masks, dim=0)
        else:
            in_images = []
            in_masks = None
            for i, img in enumerate(init_images):
                img = np.asarray(img).astype(np.float32) / 255
                if img.shape[-1] == 4:
                    img = img[..., :3] * img[..., 3:] + (1 - img[..., 3:]) * self.bg_color
                img = torch.from_numpy(img).to(device=device, dtype=dtype).permute(2, 0, 1)[None]
                if img.size(-1) < diff_size:
                    img = self.image_enhancer(img).clamp(min=0, max=1)
                in_images.append(F.interpolate(img, size=(diff_size, diff_size), mode='bilinear'))
            in_images = torch.cat(in_images, dim=0)
        return in_images, in_masks

    def load_depths(self, depths, diff_size=512):
        device = get_module_device(self.nerf)
        dtype = torch.float32 if self.unet is None else self.unet.dtype
        in_depths = []
        for depth in depths:
            if isinstance(depth, np.ndarray):
                depth = torch.from_numpy(depth).to(device=device, dtype=dtype)
            elif isinstance(depth, torch.Tensor):
                depth = depth.to(device=device, dtype=dtype)
            if depth.dim() == 3:
                depth = depth[..., 0]
            depth = depth[None, None]  # (1, 1, h, w)
            in_depths.append(F.interpolate(depth, size=(diff_size, diff_size), mode='bilinear'))
        in_depths = torch.cat(in_depths, dim=0)
        return in_depths

    def enable_normals(self, in_images, in_masks, cam_lights, ambient_light, normals=None):
        """
        Args:
            in_images (torch.Tensor): shape (N, 3, 512, 512)
            in_masks (torch.Tensor): shape (N, 1, 512, 512)
            cam_lights (torch.Tensor): shape (N, 3)
            ambient_light (float): ambient light
            normals (List[Image.Image | None]): list of normal images

        Returns:
            Tuple[torch.Tensor, torch.Tensor]:
                in_images: shape (N, 3, 512, 512)
                in_normals: shape (N, 3, 512, 512), opengl camera space, normalized to [0, 1]
        """
        dtype = torch.float32 if self.unet is None else self.unet.dtype
        diff_size = in_images.size(-1)
        no_normal_inds = [i for i, normal in enumerate(normals) if normal is None]
        if len(no_normal_inds) > 0:
            pred_normals = self.normal_model(
                F.interpolate(
                    in_images[no_normal_inds],
                    size=384, mode='bilinear', align_corners=False).to(dtype)
            ).clamp(min=0, max=1)  # opencv camera
            pred_normals[:, 1:] = 1 - pred_normals[:, 1:]  # to opengl convention
            pred_normals = F.interpolate(
                self.image_enhancer(pred_normals), size=diff_size, mode='bilinear', align_corners=False).clamp(min=0, max=1)
        in_normals = []
        ind = 0
        for normal in normals:
            if normal is None:
                in_normals.append(pred_normals[ind])
                ind += 1
            else:
                normal = np.asarray(normal).astype(np.float32) / 255
                normal = torch.from_numpy(normal).to(device=in_images.device, dtype=dtype).permute(2, 0, 1)
                if normal.size(-1) < diff_size:
                    normal = self.image_enhancer(normal[None]).squeeze(0)
                normal =  F.interpolate(
                    normal[None], size=diff_size, mode='bilinear', align_corners=False
                ).clamp(min=0, max=1).squeeze(0)
                in_normals.append(normal)
        in_normals = torch.stack(in_normals, dim=0)
        default_bg = in_normals.new_tensor([0.5, 0.5, 1.0])
        in_normals = in_normals * in_masks + (1 - in_masks) * default_bg[:, None, None]
        in_normals = F.normalize(in_normals * 2 - 1, dim=1) / 2 + 0.5
        in_normals += (1 - in_masks) * (in_normals.new_tensor(self.normal_bg) - default_bg)[:, None, None]

        in_normals_opencv = in_normals.permute(0, 2, 3, 1) * 2 - 1
        in_normals_opencv[..., 1:] = -in_normals_opencv[..., 1:]

        in_shading = ((cam_lights[:, None, None, None, :] @ in_normals_opencv[..., :, None].float()).clamp(min=0)
                      * (1 - ambient_light) + ambient_light - 1.0
                      ).squeeze(-1).permute(0, 3, 1, 2) * in_masks + 1
        if self.tonemapping is None:
            in_images = (in_images * in_shading).to(dtype)
        else:
            in_images = self.tonemapping.lut(self.tonemapping.inverse_lut(in_images)
                                             + in_shading.clamp(min=1e-6).log2()).to(dtype)
        return in_images, in_normals

    def load_cond_images(self, in_images, cond_images=None, extra_control_images=None):
        """
        Args:
            in_images (torch.Tensor): shape (N, 3, 512, 512)
            cond_images (List[Image.Image] | None): list of conditioning images
            extra_control_images (List[List[Image.Image | torch.Tensor] | Image.Image | torch.Tensor] | None): list of
                extra control images

        Returns:
            Tuple[List[torch.Tensor] | None, List[torch.Tensor] | list]:
                cond_images (List[torch.Tensor] | None): shape (N, (3, H, W))
                extra_control_images (List[torch.Tensor] | list): shape (num_control, (N, 3, 512, 512))
        """
        device = get_module_device(self.nerf)
        dtype = torch.float32 if self.unet is None else self.unet.dtype
        diff_size = in_images.size(-1)
        if cond_images is not None:
            cond_images_ = []
            for cond_image in cond_images:
                cond_image = np.array(cond_image).astype(np.float32) / 255
                if cond_image.shape[-1] == 4:
                    cond_image = cond_image[..., :3] * cond_image[..., -1:] + (1 - cond_image[..., -1:]) * self.bg_color
                cond_image = torch.from_numpy(cond_image).to(
                    device=device, dtype=dtype).permute(2, 0, 1)[None]
                cond_images_.append(cond_image)
            cond_images = cond_images_

        if isinstance(extra_control_images, list):
            if len(extra_control_images) > 0 and isinstance(extra_control_images[0], (PIL.Image.Image, torch.Tensor)):
                extra_control_images = [extra_control_images]
            extra_control_images_ = []
            for extra_control_images_group in extra_control_images:
                extra_control_images_group_ = []
                for extra_control_image in extra_control_images_group:
                    if isinstance(extra_control_image, PIL.Image.Image):
                        extra_control_image = np.array(extra_control_image).astype(np.float32) / 255
                        if extra_control_image.shape[-1] == 4:
                            extra_control_image = extra_control_image[..., :3] * extra_control_image[..., -1:] + (
                                1 - extra_control_image[..., -1:]) * self.bg_color
                        extra_control_image = torch.from_numpy(extra_control_image).to(
                            device=device, dtype=dtype).permute(2, 0, 1)[None]
                    else:
                        extra_control_image = extra_control_image.to(device=device, dtype=dtype)[None]
                    extra_control_images_group_.append(F.interpolate(
                        extra_control_image, size=(diff_size, diff_size), mode='bilinear'))
                extra_control_images_.append(torch.cat(extra_control_images_group_, dim=0))
            extra_control_images = extra_control_images_
        else:
            extra_control_images = []

        if len(extra_control_images) == 0 and self.controlnet is not None:
            for _ in range(len(self.controlnet.nets) - 2 - len(extra_control_images)):
                extra_control_images.append(in_images)

        return cond_images, extra_control_images

    def get_prompt_embeds(self, in_images, rgb_prompt, rgb_negative_prompt, ip_adapter=None, cond_images=None):
        device = self.unet.device
        dtype = self.unet.dtype
        if ip_adapter is None:
            prompt_embeds = self._encode_prompt(
                rgb_prompt,
                device,
                num_images_per_prompt=1,
                do_classifier_free_guidance=True,
                negative_prompt=rgb_negative_prompt)
        else:
            if cond_images is None:
                ipa_images = F.interpolate(
                    in_images, size=(self.clip_img_size, self.clip_img_size), mode='bilinear').to(dtype)
            else:
                ipa_images_ = []
                for cond_image in cond_images:
                    ipa_images_.append(F.interpolate(
                        cond_image, size=(self.clip_img_size, self.clip_img_size), mode='bilinear'))
                ipa_images = torch.cat(ipa_images_, dim=0)
            prompt_embeds = ip_adapter.get_prompt_embeds(
                (ipa_images - ipa_images.new_tensor(self.clip_img_mean)[:, None, None]
                 ) / ipa_images.new_tensor(self.clip_img_std)[:, None, None],
                prompt=rgb_prompt,
                negative_prompt=rgb_negative_prompt)
        return prompt_embeds

    @staticmethod
    def save_tiled_viz(out_dir, i, images, depths, normals, tgt_images, tgt_masks, tgt_normals):
        for img_id, (image, depth, normal) in enumerate(
                zip(images.permute(0, 2, 3, 1),
                    depths.permute(0, 2, 3, 1),
                    normals.permute(0, 2, 3, 1))):
            image = torch.cat([
                torch.cat([tgt_images[0, img_id],
                           tgt_masks[0, img_id].expand(-1, -1, 3),
                           tgt_normals[0, img_id] if tgt_normals is not None else torch.zeros_like(tgt_images[0, img_id])],
                          dim=1),
                torch.cat([image,
                           depth.expand(-1, -1, 3),
                           normal],
                          dim=1)],
                dim=0)
            plt.imsave(osp.join(out_dir, '{:03d}_{:03d}.png'.format(i, img_id)),
                       (image.clamp(min=0, max=1) * 255).to(torch.uint8).cpu().numpy())

    def make_shading_fun(self, worldspace_point_lights, ambient_light):
        """
        Simple point light Lambertian shading.
        """
        def shading_fun(world_pos=None, albedo=None, world_normal=None, fg_mask=None, **kwargs):
            fg_lights = worldspace_point_lights[fg_mask.squeeze(0)]
            shading = ((fg_lights[:, None, :] @ world_normal[:, :, None]).clamp(min=0)
                       * (1 - ambient_light) + ambient_light).squeeze(-1)
            if self.tonemapping is None:
                return albedo * shading
            else:
                return self.tonemapping.lut(
                    self.tonemapping.inverse_lut(albedo) + shading.clamp(min=1e-6).log2())
        return shading_fun

    def make_nerf_shading_fun(self, nerf_code, worldspace_point_lights, ambient_light):
        """
        Simple point light Lambertian shading.
        """
        def shading_fun(world_pos=None, albedo=None, world_normal=None, fg_mask=None, **kwargs):
            total_num_points = len(world_pos)
            if total_num_points == 0:
                return world_pos if albedo is None else albedo
            base_albedo = self.nerf.decoder.point_decode(world_pos[None], None, nerf_code)[1].squeeze(0)
            fg_lights = worldspace_point_lights[fg_mask.squeeze(0)]
            shading = ((fg_lights[:, None, :] @ world_normal[:, :, None]).clamp(min=0)
                       * (1 - ambient_light) + ambient_light).squeeze(-1)
            if self.tonemapping is None:
                return base_albedo * shading
            else:
                return self.tonemapping.lut(
                    self.tonemapping.inverse_lut(base_albedo) + shading.clamp(min=1e-6).log2())
        return shading_fun

    def make_nerf_albedo_shading_fun(self, nerf_code):
        def shading_fun(world_pos=None, albedo=None, **kwargs):
            total_num_points = len(world_pos)
            if total_num_points == 0:
                return world_pos if albedo is None else albedo
            return self.nerf.decoder.point_decode(world_pos[None], None, nerf_code)[1].squeeze(0)
        return shading_fun

    def nerf_optim(
            self, tgt_images, tgt_masks, tgt_normals,  # input images
            optimizer, lr, inverse_steps, n_inverse_rays,  # optimization settings
            patch_rgb_weight, patch_normal_weight, alpha_soften, normal_reg_weight, entropy_weight,  # loss weights
            nerf_code, density_grid, density_bitfield,  # nerf model
            render_size, intrinsics, intrinsics_size, camera_poses, cam_weights, cam_lights, patch_size,  # cameras
            is_init, bg_width, ambient_light, dt_gamma_scale, init_shaded,
            alpha_blur_std=1.5, debug=False, tgt_depths=None, depth_weight=0.0):
        device = get_module_device(self.nerf)
        loss_tv = TVLoss(loss_weight=1.0, power=1.5)
        use_normal = tgt_normals is not None
        use_depth = tgt_depths is not None and depth_weight > 0

        num_cameras = camera_poses.shape[0]

        cam_ids_dense = torch.arange(
            num_cameras, device=device)[None, :, None, None, None].expand(-1, -1, render_size, render_size, -1)
        cam_weights_mean = cam_weights.mean()

        if alpha_blur_std > 0:
            kernel_size = int((alpha_blur_std * 6) // 2 * 2 + 1)
            tgt_masks_blur = F_t.gaussian_blur(
                tgt_masks.square().squeeze(0).permute(0, 3, 1, 2),
                (kernel_size, kernel_size), (alpha_blur_std, alpha_blur_std)
            ).permute(0, 2, 3, 1)[None].clamp(min=alpha_soften ** 2, max=(1 - alpha_soften) ** 2).sqrt()
        else:
            tgt_masks_blur = tgt_masks.clamp(min=alpha_soften ** 2, max=(1 - alpha_soften) ** 2).sqrt()

        # (1, num_imgs, h, w, 3)
        directions = get_ray_directions(
            render_size, render_size,
            intrinsics[None] * (render_size / intrinsics_size),
            norm=False, device=intrinsics.device)

        cond_rays_o, cond_rays_d = get_rays(directions, camera_poses[None], norm=True)

        normal_bg = tgt_images.new_tensor(self.normal_bg)

        decoder_training_prev = self.nerf.decoder.training
        self.nerf.decoder.train(True)

        with torch.enable_grad():
            optimizer.param_groups[0]['lr'] = lr

            raybatch_inds, num_raybatch = self.nerf.get_raybatch_inds(tgt_images, n_inverse_rays)
            iter_density = 0

            if debug:
                pixel_rgb_loss_ = []
                patch_rgb_loss_ = []
                alphas_loss_ = []
                entropy_loss_ = []
                normal_reg_loss_ = []
                patch_normal_loss_ = []
                depth_loss_ = []
            for inverse_step_id in range(inverse_steps):
                if inverse_step_id % self.nerf.update_extra_interval == 0:
                    update_extra_state = self.nerf.update_extra_iters
                    extra_args = (density_grid, density_bitfield, iter_density)
                    extra_kwargs = dict(density_thresh=0.1)
                else:
                    update_extra_state = 0
                    extra_args = extra_kwargs = None

                inds = raybatch_inds[inverse_step_id % num_raybatch] if raybatch_inds is not None else None
                cond_extras = [tgt_masks_blur, directions, cam_ids_dense]
                if use_normal:
                    cond_extras.append(tgt_normals)
                if use_depth:
                    cond_extras.append(tgt_depths)
                ray_samples = self.nerf.ray_sample(
                    cond_rays_o, cond_rays_d, tgt_images, n_inverse_rays, sample_inds=inds, cond_extras=cond_extras)
                rays_o, rays_d, target_rgbs, target_m_blur, target_dir, target_cam_ids = ray_samples[:6]
                ray_samples = list(ray_samples[6:])
                if use_normal:
                    target_n = ray_samples.pop(0)
                if use_depth:
                    target_depth = ray_samples.pop(0)
                target_cam_ids = target_cam_ids[:, 0, 0, 0]
                target_w = cam_weights[target_cam_ids][:, None, None, None].expand(
                    -1, patch_size, patch_size, 1)
                target_lights = cam_lights[target_cam_ids][:, None, None, :].expand(
                    -1, patch_size, patch_size, 3)
                dt_gamma = dt_gamma_scale / (
                            intrinsics[target_cam_ids, :2].mean(dim=-1) * render_size / intrinsics_size)

                outputs = self.nerf.decoder(
                    rays_o, rays_d, nerf_code, density_bitfield, self.nerf.grid_size,
                    dt_gamma=dt_gamma, perturb=True,
                    update_extra_state=update_extra_state, extra_args=extra_args, extra_kwargs=extra_kwargs)
                out_rgbs = outputs['image'].reshape(target_rgbs.size())
                out_alphas = outputs['weights_sum'].reshape(target_m_blur.size())
                out_depth = outputs['depth'].reshape(-1, self.nerf.patch_size, self.nerf.patch_size)
                out_depth = out_depth * torch.linalg.norm(target_dir, dim=-1).reshape(
                    out_depth.size())  # from 1/r to 1/z

                out_depth_fg = out_depth / out_alphas.reshape(
                    -1, self.nerf.patch_size, self.nerf.patch_size).clamp(min=1e-6)
                # (num_patches, patch_size, patch_size, 3)
                out_normals_fg = depth_to_normal(out_depth_fg, target_dir)
                out_normals_fg_mask = out_alphas.reshape(-1, self.nerf.patch_size, self.nerf.patch_size, 1)
                out_normals = out_normals_fg * out_normals_fg_mask + normal_bg * (1 - out_normals_fg_mask)
                out_normals_fg_weight = -F.max_pool2d(
                    -out_normals_fg_mask.detach().squeeze(-1).unsqueeze(1), 3, stride=1, padding=1
                ).squeeze(1).unsqueeze(-1)

                if not is_init or init_shaded:
                    out_normals_fg_opencv = torch.cat(
                        [out_normals_fg[..., :1] * 2 - 1,
                         -out_normals_fg[..., 1:3] * 2 + 1], dim=-1)
                    nerf_shading = ((target_lights[..., None, :] @ out_normals_fg_opencv[..., :, None]
                                     ).clamp(min=0) * (1 - ambient_light) + ambient_light).squeeze(-1)
                    if self.tonemapping is None:
                        out_rgbs = out_rgbs * nerf_shading + self.nerf.bg_color * (1 - out_alphas)
                    else:
                        out_rgbs = self.tonemapping.lut(
                            self.tonemapping.inverse_lut(out_rgbs / out_alphas.clamp(min=1e-6))
                            + nerf_shading.clamp(min=1e-6).log2()
                        ) * out_alphas + self.nerf.bg_color * (1 - out_alphas)
                else:
                    out_rgbs = out_rgbs + self.nerf.bg_color * (1 - out_alphas)  # (1, num_rays, 3)

                loss = pixel_rgb_loss = self.nerf.pixel_loss(
                    out_rgbs.reshape(target_rgbs.size()), target_rgbs,
                    weight=target_w / cam_weights_mean) * 4.5

                alphas_loss = self.nerf.pixel_loss(
                    out_alphas.reshape(target_m_blur.size()), target_m_blur,
                    weight=target_w / cam_weights_mean) * (5.0 if is_init else 1.0)
                normal_reg_loss = loss_tv(
                    out_normals_fg.permute(0, 3, 1, 2),
                    target_n.permute(0, 3, 1, 2) if use_normal else None,
                    weight=out_normals_fg_weight.permute(0, 3, 1, 2)) * (normal_reg_weight * 10)
                loss = loss + alphas_loss + normal_reg_loss

                if use_depth:
                    depth_loss = self.nerf.pixel_loss(
                        out_depth.reshape(target_depth.size()), target_depth,
                        weight=target_w / cam_weights_mean) * depth_weight
                    loss = loss + depth_loss
                    if debug:
                        depth_loss_.append(depth_loss.item())

                bin_weights_sum = outputs['weights'].float()
                bin_width = outputs['ts'][0][:, 1].float()
                bg_weights_sum = 1 - outputs['weights_sum'].flatten()
                entropy_loss = -(torch.sum(
                    bin_weights_sum * (torch.log(bin_weights_sum.clamp(min=1e-6)) - torch.log(bin_width.clamp(min=1e-6)))
                ) + torch.sum(
                    bg_weights_sum * (torch.log(bg_weights_sum.clamp(min=1e-6)) - math.log(bg_width))
                )) * (entropy_weight / target_rgbs.shape[:-1].numel())
                loss = loss + entropy_loss

                if debug:
                    pixel_rgb_loss_.append(pixel_rgb_loss.item())
                    alphas_loss_.append(alphas_loss.item())
                    normal_reg_loss_.append(normal_reg_loss.item())
                    entropy_loss_.append(entropy_loss.item())

                if patch_rgb_weight > 0:
                    patch_rgb_loss = self.nerf.patch_loss(
                        out_rgbs.reshape(target_rgbs.size()).permute(0, 3, 1, 2),  # (num_patches, 3, h, w)
                        target_rgbs.permute(0, 3, 1, 2),  # (num_patches, 3, h, w)
                        weight=target_w[:, 0, 0, 0] / cam_weights_mean
                    ) * patch_rgb_weight
                    loss = loss + patch_rgb_loss
                    if debug:
                        patch_rgb_loss_.append(patch_rgb_loss.item())

                if use_normal and patch_normal_weight > 0:
                    patch_normal_loss = self.nerf.patch_loss(
                        highpass(out_normals.reshape(target_n.size()).permute(0, 3, 1, 2)),  # (num_patches, 3, h, w)
                        highpass(target_n.permute(0, 3, 1, 2)),  # (num_patches, 3, h, w)
                        weight=target_w[:, 0, 0, 0] / cam_weights_mean
                    ) * patch_normal_weight
                    loss = loss + patch_normal_loss
                    if debug:
                        patch_normal_loss_.append(patch_normal_loss.item())

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            if debug:
                print_str = []
                for name, loss in zip([
                        'pixel_rgb_loss',
                        'patch_rgb_loss',
                        'alphas_loss',
                        'entropy_loss',
                        'normal_reg_loss',
                        'patch_normal_loss',
                        'depth_loss'],
                        [pixel_rgb_loss_,
                         patch_rgb_loss_,
                         alphas_loss_,
                         entropy_loss_,
                         normal_reg_loss_,
                         patch_normal_loss_,
                         depth_loss_]):
                    if len(loss) > 0:
                        print_str.append(f'{name}: {np.mean(loss):.4f}')
                print('\n' + ', '.join(print_str))

        self.nerf.decoder.train(decoder_training_prev)

    def mesh_optim(
            self, tgt_images, tgt_masks, tgt_normals,  # input images
            optimizer, lr, lr_multiplier, inverse_steps, render_bs, patch_bs, mesh_simplify_texture_steps,  # optimization settings
            patch_rgb_weight, patch_normal_weight, alpha_soften, normal_reg_weight, mesh_normal_reg_weight,  # loss weights
            nerf_code, tet_verts, deform, tet_sdf, tet_indices, dmtet, in_mesh,  # mesh model
            render_size, intrinsics, intrinsics_size, camera_poses, cam_weights, lights, patch_size,  # cameras
            is_end, ambient_light, mesh_reduction, debug=False, perturb=True):
        device = get_module_device(self.nerf)
        loss_tv = TVLoss(loss_weight=1.0, power=1.5)
        use_normal = tgt_normals is not None

        cam_weights_mean = cam_weights.mean()

        tgt_masks_blur = F_t.gaussian_blur(
            tgt_masks.square().squeeze(0).permute(0, 3, 1, 2), (9, 9), (1.5, 1.5)
        ).permute(0, 2, 3, 1)[None].clamp(min=alpha_soften ** 2, max=(1 - alpha_soften) ** 2).sqrt()

        # (1, num_imgs, h, w, 3)
        directions = get_ray_directions(
            render_size, render_size,
            intrinsics[None] * (render_size / intrinsics_size),
            norm=False, device=intrinsics.device)

        normal_bg = tgt_images.new_tensor(self.normal_bg)

        decoder_training_prev = self.nerf.decoder.training
        self.nerf.decoder.train(True)

        with torch.enable_grad():
            optimizer.param_groups[0]['lr'] = lr
            optimizer.param_groups[1]['lr'] = lr * (0.04 if use_normal else 0.04 * lr_multiplier)

            camera_perm = torch.randperm(camera_poses.size(0), device=device)
            pose_batches = camera_poses[camera_perm].split(render_bs, dim=0)
            intrinsics_batches = intrinsics[camera_perm].split(render_bs, dim=0)
            tgt_image_batches = tgt_images.squeeze(0)[camera_perm].split(render_bs, dim=0)
            tgt_mask_batches = tgt_masks.squeeze(0)[camera_perm].split(render_bs, dim=0)
            tgt_mask_blur_batches = tgt_masks_blur.squeeze(0)[camera_perm].split(render_bs, dim=0)
            tgt_dir_batches = directions.squeeze(0)[camera_perm].split(render_bs, dim=0)
            num_pose_batches = len(pose_batches)
            cam_weights_batches = cam_weights[camera_perm].split(render_bs, dim=0)
            lights_batches = lights[camera_perm].split(render_bs, dim=0)
            if use_normal:
                tgt_normals_batches = tgt_normals.squeeze(0)[camera_perm].split(render_bs, dim=0)

            if debug:
                pixel_rgb_loss_ = []
                patch_rgb_loss_ = []
                alphas_loss_ = []
                lapsmth_loss_ = []
                norm_const_loss_ = []
                normal_reg_loss_ = []
                patch_normal_loss_ = []
            mesh_is_simplified = False

            if is_end:
                inverse_steps = max(inverse_steps, mesh_simplify_texture_steps)

            for inverse_step_id in range(inverse_steps):
                pose_batch = pose_batches[inverse_step_id % num_pose_batches]
                intrinsics_batch = intrinsics_batches[inverse_step_id % num_pose_batches]
                target_rgbs = tgt_image_batches[inverse_step_id % num_pose_batches]
                target_m = tgt_mask_batches[inverse_step_id % num_pose_batches]
                target_m_blur = tgt_mask_blur_batches[inverse_step_id % num_pose_batches]
                target_m_erode = -F.max_pool2d(
                    -target_m.permute(0, 3, 1, 2), 5, stride=1, padding=2).permute(0, 2, 3, 1)
                target_dir = tgt_dir_batches[inverse_step_id % num_pose_batches]
                target_w = cam_weights_batches[inverse_step_id % num_pose_batches][:, None, None, None].expand(
                    -1, render_size, render_size, 1)
                target_lights = lights_batches[inverse_step_id % num_pose_batches][:, None, None, :].expand(
                    -1, render_size, render_size, 3)
                if use_normal:
                    target_n = tgt_normals_batches[inverse_step_id % num_pose_batches]

                intrinsics_batch = intrinsics_batch * (render_size / intrinsics_size)
                if perturb:
                    intrinsics_batch[:, 2:] += (torch.rand_like(
                        intrinsics_batch[:, 2:]) - 0.5) / self.mesh_renderer.ssaa

                render_out = self.mesh_renderer(
                    [in_mesh],
                    pose_batch[None],
                    intrinsics_batch[None],
                    render_size, render_size,
                    self.make_nerf_shading_fun(
                        nerf_code, target_lights, ambient_light),
                    normal_bg=self.normal_bg)
                out_alphas = render_out['rgba'][..., 3:].squeeze(0)  # (num_imgs, h, w, 1)
                out_rgbs = (render_out['rgba'][..., :3]
                            / render_out['rgba'][..., 3:].clamp(min=1e-3)).squeeze(0)  # (num_imgs, h, w, 3)
                out_rgbs = out_rgbs * target_m_erode + target_rgbs * (1 - target_m_erode)
                out_normals = render_out['normal'].squeeze(0)  # (num_imgs, h, w, 3)
                out_normals_opencv = depth_to_normal(
                    render_out['depth'].squeeze(0).detach(), target_dir, format='opencv') * 2 - 1
                out_normals_cos = (out_normals_opencv[..., None, :]
                                   @ F.normalize(target_dir[..., :, None], dim=-2)
                                   ).squeeze(-1).neg().clamp(min=0)
                out_normals_cos = -F.max_pool2d(  # alleviate edge effect
                    -out_normals_cos.permute(0, 3, 1, 2), 5, stride=1, padding=2).permute(0, 2, 3, 1)
                out_normals = out_normals * out_normals_cos + out_normals.detach() * (1 - out_normals_cos)
                out_normals_fg = (out_normals - normal_bg * (1 - out_alphas)) / out_alphas.clamp(min=1e-3)
                out_normals_fg_weight = out_alphas.detach()

                loss = pixel_rgb_loss = self.nerf.pixel_loss(
                    out_rgbs.reshape(target_rgbs.size()), target_rgbs,
                    weight=target_w / cam_weights_mean) * 4.5
                if debug:
                    pixel_rgb_loss_.append(pixel_rgb_loss.item())

                if not mesh_is_simplified:
                    alphas_loss = self.nerf.pixel_loss(
                        out_alphas.reshape(target_m_blur.size()), target_m_blur,
                        weight=target_w / cam_weights_mean) * 2.0
                    normal_reg_loss = loss_tv(
                        out_normals_fg.permute(0, 3, 1, 2),
                        target_n.permute(0, 3, 1, 2) if use_normal else None,
                        weight=out_normals_fg_weight.permute(0, 3, 1, 2)) * (normal_reg_weight * 2)
                    lapsmth_loss = laplacian_smooth_loss(in_mesh.v, in_mesh.f) * mesh_normal_reg_weight
                    norm_const_loss = normal_consistency(in_mesh.face_normals, in_mesh.f) * mesh_normal_reg_weight
                    loss = loss + alphas_loss + normal_reg_loss + lapsmth_loss + norm_const_loss
                    if debug:
                        alphas_loss_.append(alphas_loss.item())
                        normal_reg_loss_.append(normal_reg_loss.item())
                        lapsmth_loss_.append(lapsmth_loss.item())
                        norm_const_loss_.append(norm_const_loss.item())

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
                target_w_patch_ = target_w_patch[patch_batch, 0, 0, 0]

                if patch_rgb_weight > 0:
                    patch_rgb_loss = self.nerf.patch_loss(
                        out_rgb_patch[patch_batch],  # (num_patches, 3, h, w)
                        tgt_rgb_patch[patch_batch],  # (num_patches, 3, h, w)
                        weight=target_w_patch_ / cam_weights_mean
                    ) * patch_rgb_weight
                    loss = loss + patch_rgb_loss
                    if debug:
                        patch_rgb_loss_.append(patch_rgb_loss.item())

                if use_normal and patch_normal_weight > 0 and not mesh_is_simplified:
                    out_normal_patch = out_normals.reshape(
                        -1, render_size // patch_size, patch_size, render_size // patch_size, patch_size, 3
                    ).permute(0, 1, 3, 5, 2, 4).reshape(-1, 3, patch_size, patch_size)
                    tgt_normal_patch = target_n.reshape(
                        -1, render_size // patch_size, patch_size, render_size // patch_size, patch_size, 3
                    ).permute(0, 1, 3, 5, 2, 4).reshape(-1, 3, patch_size, patch_size)
                    patch_batch = torch.randperm(out_normal_patch.size(0), device=device)[:patch_bs]
                    patch_normal_loss = self.nerf.patch_loss(
                        highpass(out_normal_patch[patch_batch]),  # (num_patches, 3, h, w)
                        highpass(tgt_normal_patch[patch_batch]),  # (num_patches, 3, h, w)
                        weight=target_w_patch_ / cam_weights_mean
                    ) * patch_normal_weight
                    loss = loss + patch_normal_loss
                    if debug:
                        patch_normal_loss_.append(patch_normal_loss.item())

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                if not mesh_is_simplified:
                    mesh_verts, mesh_faces = dmtet(tet_verts + deform, tet_sdf, tet_indices)
                    if mesh_reduction < 1 and is_end and (
                            inverse_steps - (inverse_step_id + 1)) <= mesh_simplify_texture_steps:
                        # mesh_verts_, mesh_faces_ = fast_simplification.simplify(
                        #     mesh_verts.detach().cpu().numpy(), mesh_faces.detach().cpu().numpy(),
                        #     target_reduction=1 - mesh_reduction)
                        o3d_mesh = o3d.geometry.TriangleMesh(
                            o3d.utility.Vector3dVector(mesh_verts.detach().cpu().numpy()),
                            o3d.utility.Vector3iVector(mesh_faces.detach().cpu().numpy()))
                        o3d_mesh = o3d_mesh.simplify_quadric_decimation(
                            round(len(o3d_mesh.triangles) * mesh_reduction), boundary_weight=0).compute_vertex_normals()
                        mesh_verts = torch.from_numpy(np.asarray(o3d_mesh.vertices)).to(mesh_verts)
                        mesh_faces = torch.from_numpy(np.asarray(o3d_mesh.triangles)).to(mesh_faces)
                        mesh_verts, indices = torch.unique(mesh_verts, dim=0, return_inverse=True, sorted=False)
                        mesh_faces = indices[mesh_faces]
                        optimizer = torch.optim.Adam(self.nerf.decoder.parameters(), lr=lr)
                        mesh_is_simplified = True
                    mesh_faces = mesh_faces.int()
                    in_mesh = Mesh(v=mesh_verts, f=mesh_faces, device=device)
                    in_mesh.auto_normal()

            if debug:
                print_str = []
                for name, loss in zip([
                        'pixel_rgb_loss',
                        'patch_rgb_loss',
                        'alphas_loss',
                        'lapsmth_loss',
                        'norm_const_loss',
                        'normal_reg_loss',
                        'patch_normal_loss'],
                        [pixel_rgb_loss_,
                         patch_rgb_loss_,
                         alphas_loss_,
                         lapsmth_loss_,
                         norm_const_loss_,
                         normal_reg_loss_,
                         patch_normal_loss_]):
                    if len(loss) > 0:
                        print_str.append(f'{name}: {np.mean(loss):.4f}')
                print('\n' + ', '.join(print_str))

        self.nerf.decoder.train(decoder_training_prev)

        return in_mesh

    @torch.no_grad()
    def __call__(
            self,
            prompt: Union[str, List[str]] = '',
            negative_prompt: Union[str, List[str]] = '',
            in_model: Optional[Union[str, Mesh]] = None,
            ingp_states: Optional[Dict] = None,
            init_images: Optional[List[PIL.Image.Image]] = None,
            cond_images: Optional[List[PIL.Image.Image]] = None,
            extra_control_images: Optional[List[List[PIL.Image.Image | torch.Tensor]]] = None,
            normals: Optional[List[PIL.Image.Image]] = None,
            depths: Optional[List[np.ndarray]] = None,
            nerf_code: Optional[torch.tensor] = None,
            density_grid: Optional[torch.tensor] = None,
            density_bitfield: Optional[torch.tensor] = None,
            camera_poses: List[np.ndarray] = None,
            intrinsics: torch.Tensor = None,  # (num_cameras, 4)
            intrinsics_size: Union[int, float] = 256,
            use_reference: bool = True,
            use_normal: bool = True,
            cam_weights: List[float] = None,
            keep_views: List[int] = None,
            guidance_scale: float = 7,
            num_inference_steps: int = 24,
            denoising_strength: Optional[float] = 0.6,
            progress_to_dmtet: float = 0.6,
            tet_resolution: int = 128,
            patch_size: int = 128,
            patch_bs: int = 8,
            diff_bs: int = 12,
            render_bs: int = 8,
            n_inverse_rays: int = 2 ** 14,
            n_inverse_steps: int = 64,
            init_inverse_steps: int = 256,
            tet_init_inverse_steps: int = 120,
            seg_padding: int = 0,
            ip_adapter: Optional[IPAdapter] = None,
            lr_schedule: Callable = default_lr_schedule,
            lr_multiplier: Callable = default_lr_multiplier,
            render_size_p: Callable = default_render_size_p,
            max_num_views: Callable = default_max_num_views,
            patch_rgb_weight: Callable = default_patch_rgb_weight,
            patch_normal_weight: Callable = default_patch_normal_weight,
            entropy_weight: Callable = default_entropy_weight,
            alpha_soften: float = 0.02,
            alpha_blur_std: float = 1.5,
            normal_reg_weight: Callable = default_normal_reg_weight,
            mesh_normal_reg_weight: float = 5.0,
            depth_weight=0.0,
            ambient_light: float = 0.2,
            mesh_reduction: float = 1.0,
            mesh_simplify_texture_steps: int = 24,
            dt_gamma_scale: float = 1.0,
            testmode_dt_gamma_scale: float = 0.25,
            bg_width: float = 0.015,
            optim_only: bool = False,
            debug: bool = False,
            out_dir: Optional[str] = None,
            save_interval: Optional[int] = None,
            save_all_interval: Optional[int] = None,
            default_prompt='best quality, sharp focus, photorealistic, extremely detailed',
            default_neg_prompt='worst quality, low quality, depth of field, blurry, out of focus, low-res, '
                               'illustration, painting, drawing',
            bake_texture=True,
            bake_texture_kwargs=None,
            mode='1-pass',
            blend_weight=0.0,
            prog_bar=tqdm):
        r"""
        Function invoked when calling the pipeline for generation.

        Args:
            prompt: The prompt or prompts to guide the image generation.
            negative_prompt: The prompt or prompts not to guide the image generation.
            in_model: The initial 3D model to start with. If not defined, one has to pass `init_images`.
            ingp_states: Optional initial InstantNGP state dict.
            init_images: The initial images to start with. If not defined, one has to pass `in_model`.
            cond_images: The conditioning images to guide the image generation via IP-Adapter and cross-image attention.
                If not defined, `init_images` or renderings of `in_model` will be used instead.
            extra_control_images: The extra control images to guide the image generation via ControlNets. If not
                defined, `init_images` or renderings of `in_model` will be used instead.
            nerf_code: Optional base NeRF code of the object.
            density_grid: Optional initial density grid of the NeRF.
            density_bitfield: Optional initial density bitfield of the NeRF.
            camera_poses: The camera poses to render the object from.
            intrinsics: The camera intrinsics.
            intrinsics_size: The image size corresponding to the intrinsics.
            use_reference: Whether to use cross-image attention.
            use_normal: Whether to use predicted normal maps.
            cam_weights: The camera weights for the views.
            keep_views: The views to keep regardless of `max_num_views`.
            guidance_scale: Classifier-Free Guidance scale, affecting the conditioning strength of `prompt` and
                `cond_images`.
            num_inference_steps: The number of denoising steps. More denoising steps usually lead to a higher quality
                image at the expense of slower inference. This parameter is modulated by `denoising_strength`.
            denoising_strength: Indicates extent to transform the inital object. Must be between 0 and 1. `in_model` or
                `init_images` is used as a starting point and more noise is added the higher the strength. The number of
                 denoising steps depends on the amount of noise initially added. When `denoising_strength` is 1, added
                 noise is maximum and the denoising process runs for the full number of iterations specified in
                 `num_inference_steps`. If None, `in_model` or `init_images` is completely ignored and sampling
                 starts from random Gaussian noise.
            progress_to_dmtet: The progress to switch from NeRF to DMTet. Must be between 0 and 1.
            tet_resolution: The resolution of the tetrahedral grid.
            patch_size: Rendering patch size.
            patch_bs: The batch size of the patches for computing the LPIPS loss.
            diff_bs: The batch size of the views for the diffusion model.
            render_bs: The batch size of the views for mesh rendering.
            n_inverse_rays: The number of rays to sample per NeRF optimization step. Must be a multiple of
                `patch_size`**2.
            n_inverse_steps: The number of optimization steps per denoising timestep.
            init_inverse_steps: The number of optimization steps for the initial NeRF optimization.
            tet_init_inverse_steps: The number of optimization steps for the initial DMTet optimization.
            seg_padding: The padding size for background segmentation. Proper padding can improve the accuracy of
                background segmentation.
            ip_adapter: The IP-Adapter text-image encoder.
            lr_schedule: The learning rate schedule as a function of the current progress.
            lr_multiplier: The learning rate multiplier for DMTet optimization as a function of the current progress.
            render_size_p: The rendering size as a function of the current progress.
            max_num_views: The maximum number of views to use for rendering as a function of the current progress.
            patch_rgb_weight: The patch RGB loss weight as a function of the current progress.
            patch_normal_weight: The patch normal loss weight as a function of the current progress.
            entropy_weight: The entropy loss weight as a function of the current progress.
            normal_reg_weight: The normal regularization loss weight as a function of the current progress.
            mesh_normal_reg_weight: The mesh normal regularization loss weight.
            ambient_light: The ratio of ambient light for shading.
            mesh_reduction: The reduction factor for the mesh simplification at the final denoising timestep. Must be
                between 0 and 1. When `mesh_reduction` is 1, no simplification is performed.
            mesh_simplify_texture_steps: The number of texture optimization steps after mesh simplification.
            dt_gamma_scale: Ray sampler hyperparameter.
            bg_width: Entropy loss hyperparameter.
            optim_only: Whether to ablate the diffusion model.
            debug: Whether to print loss values.
            out_dir: The output directory to save the visualizations.
            save_interval: The interval to save the tiled visualization of each view.
            save_all_interval: The interval to save the full visualization of each view.
            default_prompt: The quality control text appended to `prompt`.
            default_neg_prompt: The quality control text appended to `negative_prompt`.
        """
        if not optim_only:
            assert mode in ['1-pass', '2-pass']
            apply_cross_image_attn_proc(self.unet)

        device = get_module_device(self.nerf)
        nn_dtype = next(self.image_enhancer.parameters()).dtype
        torch.set_grad_enabled(False)

        # ================= Initialize NeRF and DMTet =================
        if self.nerf.decoder.state_dict_bak is None:  # 1st run
            self.nerf.decoder.backup_state_dict()

        try:
            if ingp_states is not None:
                self.nerf.decoder.load_state_dict(
                    ingp_states if isinstance(ingp_states, dict) else
                    torch.load(ingp_states, map_location='cpu'), strict=False)
            if density_grid is None:
                density_grid = self.nerf.get_init_density_grid(1, device)
            if density_bitfield is None:
                density_bitfield = self.nerf.get_init_density_bitfield(1, device)

            optimizer = torch.optim.Adam(self.nerf.decoder.parameters(), lr=0.01)
            if nerf_code is None:
                nerf_code = [None]

            dmtet = DMTet(device)
            tet_is_init = False

            # ================= Pre-process inputs =================
            if isinstance(camera_poses, torch.Tensor):
                camera_poses = camera_poses.to(device=device, dtype=torch.float32)
            else:
                camera_poses = torch.from_numpy(np.stack(camera_poses, axis=0)).to(device=device, dtype=torch.float32)
            num_cameras = len(camera_poses)

            if intrinsics.dim() == 1:
                intrinsics = intrinsics[None].expand(num_cameras, -1)

            lights, cam_lights = light_sampling(camera_poses)

            if init_images is None:
                if in_model is not None:
                    print('Initializing from mesh')
                    _, images, alphas, _ = self.load_init_mesh(
                        in_model, camera_poses, intrinsics, intrinsics_size, render_bs,
                        [self.make_shading_fun(
                            lights_batch[:, None, None, :].expand(-1, 1024, 1024, -1),
                            ambient_light) for lights_batch in lights.split(render_bs, dim=0)])
                else:
                    print('Initializing from NeRF')
                    images, alphas = self.load_init_nerf(
                        nerf_code, density_bitfield, camera_poses, intrinsics, intrinsics_size,
                        lights, cam_lights, ambient_light, render_bs, testmode_dt_gamma_scale)
                in_images = images.permute(0, 3, 1, 2).to(dtype=nn_dtype)
                in_masks = alphas.permute(0, 3, 1, 2).to(dtype=nn_dtype)
                init_shaded = True
            else:
                in_images, in_masks = self.load_init_images(init_images, ret_masks=True, seg_padding=seg_padding)
                init_shaded = False

            if cam_weights is None:
                cam_weights = [1.0] * num_cameras
            cam_weights = camera_poses.new_tensor(cam_weights)

            if not isinstance(prompt, list):
                prompt = [prompt] * num_cameras
            if not isinstance(negative_prompt, list):
                negative_prompt = [negative_prompt] * num_cameras

            if use_normal:
                if normals is None:
                    normals = [None] * num_cameras
                in_images, in_normals = self.enable_normals(in_images, in_masks, cam_lights, ambient_light, normals=normals)
                init_shaded = True

            use_depth = depths is not None and depth_weight > 0
            if use_depth:
                in_depths = self.load_depths(depths)

            cond_images, extra_control_images = self.load_cond_images(in_images, cond_images, extra_control_images)

            rgb_prompt = [join_prompts(prompt_single, default_prompt) for prompt_single in prompt]
            rgb_negative_prompt = [join_prompts(negative_prompt_single, default_neg_prompt)
                                   for negative_prompt_single in negative_prompt]

            if not optim_only:
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

            total_steps = num_inference_steps if optim_only else len(timesteps)

            # ================= Main sampling loop =================
            for i, t in enumerate(prog_bar([None] * (num_inference_steps + 1) if optim_only else [None] + list(timesteps))):
                progress = i / total_steps
                render_size = render_size_p(progress)

                do_save = save_interval is not None and (i % save_interval == 0 or i == total_steps)
                do_save_all = save_all_interval is not None and (i % save_all_interval == 0 or i == total_steps)

                # ================= Update cameras =================
                if i == 0:  # initialization
                    if keep_views is None:
                        keep_views = []
                    reorder_ids = keep_views
                    num_keep_views = len(keep_views)
                    for cam_id in range(num_cameras):
                        if cam_id not in keep_views:
                            reorder_ids.append(cam_id)
                    reorder_ids = torch.tensor(reorder_ids, device=device)
                    in_images = in_images[reorder_ids]
                    in_masks = in_masks[reorder_ids]
                    camera_poses = camera_poses[reorder_ids]
                    intrinsics = intrinsics[reorder_ids]
                    cam_weights = cam_weights[reorder_ids]
                    lights = lights[reorder_ids]
                    extra_control_images = [
                        extra_control_image[reorder_ids] for extra_control_image in extra_control_images]
                    if use_normal:
                        in_normals = in_normals[reorder_ids]
                    if use_depth:
                        in_depths = in_depths[reorder_ids]
                    if not optim_only:
                        init_latents = init_latents[reorder_ids]
                        ref_latents = ref_latents[reorder_ids] if use_reference else None
                        reorder_ids_ = torch.cat([reorder_ids, reorder_ids + num_cameras], dim=0)
                        prompt_embeds = prompt_embeds[reorder_ids_]
                    dists = get_camera_dists(camera_poses, cam_weights, device)
                    if isinstance(self.scheduler, DPMSolverSDEScheduler):
                        batch_scheduler = [deepcopy(self.scheduler) for _ in range(num_cameras)]

                else:
                    max_num_cameras = max(int(round(max_num_views(progress, progress_to_dmtet))), num_keep_views)
                    if max_num_cameras < num_cameras:
                        if optim_only:
                            pixel_dist = None
                        else:
                            pixel_dist = (ctrl_images - in_images).square().flatten(1).mean(dim=1) / (
                                in_masks.flatten(1).mean(dim=1) + 0.1)
                        keep_ids, dists = prune_cameras(
                            dists, num_keep_views, max_num_cameras, device, pixel_dist=pixel_dist)
                        in_images = in_images[keep_ids]
                        camera_poses = camera_poses[keep_ids]
                        intrinsics = intrinsics[keep_ids]
                        cam_weights = cam_weights[keep_ids]
                        lights = lights[keep_ids]
                        extra_control_images = [
                            extra_control_image[keep_ids] for extra_control_image in extra_control_images]
                        in_masks = in_masks[keep_ids]
                        if use_normal:
                            in_normals = in_normals[keep_ids]
                        if use_depth:
                            in_depths = in_depths[keep_ids]
                        if not optim_only:
                            if mode == '1-pass':
                                ctrl_images = ctrl_images[keep_ids]
                                ctrl_depths = ctrl_depths[keep_ids]
                            latents = latents[keep_ids]
                            ref_latents = ref_latents[keep_ids] if use_reference else None
                            keep_ids_ = torch.cat([keep_ids, keep_ids + num_cameras], dim=0)
                            prompt_embeds = prompt_embeds[keep_ids_]
                        if isinstance(self.scheduler, DPMSolverMultistepScheduler):
                            self.scheduler.model_outputs = [output[keep_ids] if output is not None else None for output in
                                                            self.scheduler.model_outputs]
                        elif isinstance(self.scheduler, DPMSolverSDEScheduler):
                            batch_scheduler = [batch_scheduler[keep_id] for keep_id in keep_ids]

                        num_cameras = max_num_cameras

                # ================= Denoising step P1 =================
                if not optim_only:
                    sqrt_alpha_bar_t, sqrt_one_minus_alpha_bar_t = get_noise_scales(
                        alphas_bar, timesteps[0] if t is None else t,
                        self.scheduler.num_train_timesteps, dtype=nn_dtype)
                    blend_weight_t = 1 - sqrt_alpha_bar_t if blend_weight == 'dynamic' else blend_weight

                if t is not None and not optim_only:
                    latents_scaled = self.scheduler.scale_model_input(latents, t)
                    if use_reference:
                        latent_batches = latents_scaled[:, :, -64:].split(diff_bs, dim=0) \
                                         + latents_scaled.split(diff_bs, dim=0)
                        prompt_embeds_batches = prompt_embeds[:num_cameras].split(diff_bs, dim=0) \
                                                + prompt_embeds[-num_cameras:].split(diff_bs, dim=0)
                        if mode == '1-pass':
                            ctrl_images_batches = ctrl_images.split(diff_bs, dim=0) * 2
                            ctrl_depths_batches = ctrl_depths.split(diff_bs, dim=0) * 2
                        extra_control_batches = [extra_control_image_group.split(diff_bs, dim=0) * 2
                                                 for extra_control_image_group in extra_control_images]
                    else:
                        latent_batches = torch.cat([latents_scaled] * 2, dim=0).split(diff_bs, dim=0)
                        prompt_embeds_batches = prompt_embeds.split(diff_bs, dim=0)
                        if mode == '1-pass':
                            ctrl_images_batches = torch.cat([ctrl_images] * 2, dim=0).split(diff_bs, dim=0)
                            ctrl_depths_batches = torch.cat([ctrl_depths] * 2, dim=0).split(diff_bs, dim=0)
                        extra_control_batches = [torch.cat([extra_control_image_group] * 2, dim=0).split(diff_bs, dim=0)
                                                 for extra_control_image_group in extra_control_images]

                    if mode == '1-pass':
                        noise_pred = self.get_noise_pred(
                            latent_batches, prompt_embeds_batches, ctrl_images_batches, ctrl_depths_batches,
                            t, sqrt_alpha_bar_t, sqrt_alpha_bar_t, guidance_scale,
                            extra_control_batches=extra_control_batches)
                    else:
                        noise_pred, dec_args, dec_kwargs = self.get_noise_pred_p1(
                            latent_batches, prompt_embeds_batches, t, guidance_scale,
                            extra_control_batches=extra_control_batches)
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
                    tgt_masks = self.get_tgt_masks(tgt_images, seg_padding)

                else:
                    tgt_images = in_images.permute(0, 2, 3, 1)[None].to(torch.float32)
                    tgt_masks = in_masks.permute(0, 2, 3, 1)[None].to(torch.float32)

                tgt_normals = in_normals.permute(0, 2, 3, 1)[None].to(torch.float32) if use_normal else None
                tgt_depths = in_depths.permute(0, 2, 3, 1)[None].to(torch.float32) if use_depth else None

                # ================= NeRF/mesh optimization =================
                if i == 0:
                    inverse_steps = init_inverse_steps
                else:
                    inverse_steps = n_inverse_steps
                cam_lights = (lights[:, None, :] @ camera_poses[:, :3, :3]).squeeze(-2)
                if render_size != 512:
                    tgt_images = F.interpolate(
                        tgt_images.squeeze(0).permute(0, 3, 1, 2), size=render_size, mode='bilinear'
                    ).permute(0, 2, 3, 1)[None]
                    tgt_masks = F.interpolate(
                        tgt_masks.squeeze(0).permute(0, 3, 1, 2), size=render_size, mode='bilinear'
                    ).permute(0, 2, 3, 1)[None]
                    if use_normal:
                        tgt_normals = F.interpolate(
                            tgt_normals.squeeze(0).permute(0, 3, 1, 2), size=render_size, mode='bilinear'
                        ).permute(0, 2, 3, 1)[None]
                    if use_depth:
                        tgt_depths = F.interpolate(
                            tgt_depths.squeeze(0).permute(0, 3, 1, 2), size=render_size, mode='bilinear'
                        ).permute(0, 2, 3, 1)[None]
                if progress <= progress_to_dmtet:
                    self.nerf_optim(
                        tgt_images, tgt_masks, tgt_normals,
                        optimizer, lr_schedule(progress), inverse_steps, n_inverse_rays,
                        patch_rgb_weight(progress), patch_normal_weight(progress),
                        alpha_soften, normal_reg_weight(progress), entropy_weight(progress),
                        nerf_code, density_grid, density_bitfield,
                        render_size, intrinsics, intrinsics_size, camera_poses, cam_weights, cam_lights, patch_size,
                        t is None, bg_width, ambient_light, dt_gamma_scale, init_shaded,
                        alpha_blur_std=alpha_blur_std, debug=debug, tgt_depths=tgt_depths, depth_weight=depth_weight)
                else:
                    if not tet_is_init:  # convert to dmtet
                        tet_verts, tet_indices, tet_sdf = init_tet(self.nerf, nerf_code, resolution=tet_resolution)
                        deform = torch.zeros_like(tet_verts)
                        tet_sdf.requires_grad_(True)
                        deform.requires_grad_(True)
                        tet_is_init = True
                        optimizer = torch.optim.Adam(
                            [{'params': self.nerf.decoder.parameters()},
                             {'params': [tet_sdf, deform], 'lr': 1e-3}], lr=0.01, weight_decay=0)
                        inverse_steps = tet_init_inverse_steps
                        with torch.enable_grad():
                            mesh_verts, mesh_faces = dmtet(tet_verts + deform, tet_sdf, tet_indices)
                            mesh_faces = mesh_faces.int()
                            in_mesh = Mesh(v=mesh_verts, f=mesh_faces, device=device)
                            in_mesh.auto_normal()
                        torch.cuda.empty_cache()

                    assert in_mesh.v.requires_grad
                    in_mesh = self.mesh_optim(
                        tgt_images, tgt_masks, tgt_normals,
                        optimizer, lr_schedule(progress), lr_multiplier(progress, progress_to_dmtet),
                        inverse_steps, render_bs, patch_bs, mesh_simplify_texture_steps,
                        patch_rgb_weight(progress), patch_normal_weight(progress),
                        alpha_soften, normal_reg_weight(progress), mesh_normal_reg_weight,
                        nerf_code, tet_verts, deform, tet_sdf, tet_indices, dmtet, in_mesh,
                        render_size, intrinsics, intrinsics_size, camera_poses, cam_weights, lights, patch_size,
                        progress == 1, ambient_light, mesh_reduction, debug=debug)

                # ================= NeRF/mesh rendering =================
                if (i < total_steps and not optim_only) or do_save or do_save_all:
                    images = []
                    alphas = []
                    depths = []
                    normals = []
                    for pose_batch, intrinsics_batch, lights_batch, cam_lights_batch in zip(
                            camera_poses.split(render_bs, dim=0),
                            intrinsics.split(render_bs, dim=0),
                            lights.split(render_bs, dim=0),
                            cam_lights.split(render_bs, dim=0)):
                        if tet_is_init:
                            render_out = self.mesh_renderer(
                                [in_mesh],
                                pose_batch[None],
                                intrinsics_batch[None] * (render_size / intrinsics_size),
                                render_size, render_size,
                                self.make_nerf_shading_fun(
                                    nerf_code, lights_batch[:, None, None, :].expand(-1, render_size, render_size, -1),
                                    ambient_light),
                                normal_bg=self.normal_bg)
                            images.append((render_out['rgba'][..., :3]
                                           + self.bg_color * (1 - render_out['rgba'][..., 3:])).squeeze(0))
                            alphas.append(render_out['rgba'][..., 3:].squeeze(0))
                            depths.append(render_out['depth'].squeeze(0))
                            normals.append(render_out['normal'].squeeze(0))

                        else:
                            rgba_batch, depth_batch, normal_batch, normal_fg_batch = self.nerf.render(
                                self.nerf.decoder,
                                nerf_code,
                                density_bitfield, render_size, render_size,
                                intrinsics_batch[None] * (render_size / intrinsics_size),
                                pose_batch[None],
                                cfg=dict(return_rgba=True, compute_normal=True, dt_gamma_scale=testmode_dt_gamma_scale),
                                perturb=False,
                                normal_bg=self.normal_bg)
                            normal_fg_batch_opencv = torch.cat(
                                [normal_fg_batch[..., :1] * 2 - 1,
                                 -normal_fg_batch[..., 1:3] * 2 + 1], dim=-1)
                            nerf_shading = ((cam_lights_batch[:, None, None, None, :] @ normal_fg_batch_opencv[..., :, None]
                                             ).clamp(min=0) * (1 - ambient_light) + ambient_light).squeeze(-1)
                            if self.tonemapping is None:
                                image_batch = rgba_batch[..., :3] * nerf_shading \
                                              + self.nerf.bg_color * (1 - rgba_batch[..., 3:])
                            else:
                                image_batch = self.tonemapping.lut(
                                    self.tonemapping.inverse_lut(rgba_batch[..., :3] / rgba_batch[..., 3:].clamp(min=1e-6))
                                    + nerf_shading.clamp(min=1e-6).log2()
                                ) * rgba_batch[..., 3:] + self.nerf.bg_color * (1 - rgba_batch[..., 3:])

                            images.append(image_batch.squeeze(0))
                            alphas.append(rgba_batch[..., 3:].squeeze(0))
                            depths.append(depth_batch.squeeze(0))
                            normals.append(normal_batch.squeeze(0))

                    images = torch.cat(images, dim=0).to(nn_dtype).permute(0, 3, 1, 2).clamp(min=0, max=1)
                    normals = torch.cat(normals, dim=0).to(nn_dtype).permute(0, 3, 1, 2).clamp(min=0, max=1)
                    alphas = torch.cat(alphas, dim=0)
                    depths = torch.cat(depths, dim=0)
                    depths = normalize_depth(depths, alphas).to(nn_dtype).unsqueeze(1).repeat(1, 3, 1, 1)
                    alphas = alphas.permute(0, 3, 1, 2).clamp(min=0, max=1).to(nn_dtype)

                    if render_size != 512:
                        images_ = F.interpolate(
                            self.image_enhancer(images), size=(512, 512), mode='bilinear').clamp(min=0, max=1)
                        depths_ = F.interpolate(depths, size=(512, 512), mode='bilinear')
                    else:
                        images_ = images
                        depths_ = depths

                    ctrl_images = images_
                    ctrl_depths = depths_

                    if do_save:
                        self.save_tiled_viz(out_dir, i, images, depths, normals, tgt_images, tgt_masks, tgt_normals)

                if i < total_steps:  # not the final step
                    # ================= Denoising step P2 =================
                    if mode == '2-pass':
                        if t is not None and not optim_only:
                            if use_reference:
                                ctrl_images_batches = ctrl_images.split(diff_bs, dim=0) * 2
                                ctrl_depths_batches = ctrl_depths.split(diff_bs, dim=0) * 2
                            else:
                                ctrl_images_batches = torch.cat([ctrl_images] * 2, dim=0).split(diff_bs, dim=0)
                                ctrl_depths_batches = torch.cat([ctrl_depths] * 2, dim=0).split(diff_bs, dim=0)

                            noise_pred = self.get_noise_pred_p2(
                                latent_batches, prompt_embeds_batches, dec_args, dec_kwargs, t, guidance_scale,
                                ctrl_images_batches, 1, ctrl_depths_batches,1)

                    if do_save_all:
                        if optim_only:
                            self.save_all_viz(out_dir, i, num_keep_views, tgt_images, diff_bs=diff_bs)
                        else:
                            self.save_all_viz(out_dir, i, num_keep_views, tgt_images, latents_scaled,
                                              ctrl_images, ctrl_depths, normals, alphas, diff_bs=diff_bs)

                    # ================= Solver step =================
                    if optim_only:
                        continue

                    if blend_weight_t > 0:
                        pred_nerf_sample = []
                        for batch_images in images_.split(diff_bs, dim=0):
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
                        if isinstance(self.scheduler, DPMSolverSDEScheduler):
                            latents = [batch_scheduler[i].step(merged_noise[i:i + 1], t, latents[i:i + 1], return_dict=False)[0]
                                       for i in range(num_cameras)]
                            latents = torch.cat(latents, dim=0)
                        else:
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
            if bake_texture_kwargs is None:
                bake_texture_kwargs = dict()
            out_mesh = self.mesh_renderer.bake_xyz_shading_fun(
                [in_mesh.detach()], self.make_nerf_albedo_shading_fun(nerf_code),
                **bake_texture_kwargs,
            )[0] if bake_texture else in_mesh.detach()

            output_state = deepcopy(self.nerf.decoder.state_dict())

        except Exception:
            print(traceback.format_exc())
            out_mesh = output_state = None

        self.nerf.decoder.restore_state_dict()

        if not optim_only:
            remove_cross_image_attn_proc(self.unet)

        return out_mesh, output_state
