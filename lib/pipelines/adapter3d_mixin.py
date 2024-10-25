import os.path as osp
import matplotlib.pyplot as plt
import torch
from copy import copy
from typing import Tuple
from diffusers.pipelines.controlnet import MultiControlNetModel
from lib.models.architecture.diffusers import unet_enc, unet_dec
from lib.models.decoders.mesh_renderer.mesh_utils import Mesh
from .utils import do_segmentation


class Adapter3DMixin:

    def get_tgt_masks(self, tgt_images, seg_padding):
        tgt_images = tgt_images.squeeze(0).clip(min=0, max=1).permute(0, 3, 1, 2)  # (num_cameras, 3, h, w)
        images_masked = do_segmentation(
            tgt_images, self.segmentation, padding=seg_padding, bg_color=self.bg_color)
        tgt_masks = images_masked[:, 3]
        return tgt_masks[None, ..., None]

    def load_init_mesh(self, in_model, camera_poses, intrinsics, intrinsics_size,
                       render_bs, shading_fun=None, diff_size=512):
        """
        Args:
            in_model (str | Mesh): input model
            camera_poses (torch.Tensor): camera poses, shape (N, 3, 4)
            intrinsics (torch.Tensor): intrinsics, shape (N, 4) in [fx, fy, cx, cy]
            intrinsics_size (int): image size corresponding to the intrinsics
            render_bs (int): batch size for rendering

        Returns:
            Tuple[Mesh, torch.Tensor, torch.Tensor, torch.Tensor]:
                in_mesh: input mesh
                images: rendered images, shape (N, 512, 512, 3), renderer dtype
                alphas: rendered alphas, shape (N, 512, 512, 1), renderer dtype
                depths: rendered depths, shape (N, 512, 512), renderer dtype
        """
        device = self.unet.device
        if isinstance(in_model, str):
            in_mesh = Mesh.load(in_model, flip_yz=in_model.endswith(('.obj', '.glb'))).to(device)
        else:
            in_mesh = in_model.detach().to(device)
        mesh_renderer = copy(self.mesh_renderer)
        mesh_renderer.ssaa = 2
        pose_batches = camera_poses.split(render_bs, dim=0)
        intrinsics_batches = intrinsics.split(render_bs, dim=0)
        shading_fun_batches = shading_fun if isinstance(shading_fun, list) else [shading_fun] * len(pose_batches)
        images = []
        alphas = []
        depths = []
        for pose_batch, intrinsics_batch, shading_fun_batch in zip(
                pose_batches, intrinsics_batches, shading_fun_batches):
            render_out = mesh_renderer(
                [in_mesh],
                pose_batch[None],
                intrinsics_batch[None] * (diff_size / intrinsics_size),
                diff_size, diff_size,
                shading_fun_batch)
            images.append((render_out['rgba'][..., :3]
                           + (1 - render_out['rgba'][..., 3:]) * self.bg_color).squeeze(0))
            alphas.append(render_out['rgba'][..., 3:].squeeze(0))
            depths.append(render_out['depth'].squeeze(0))
        images = torch.cat(images, dim=0).clamp(min=0, max=1)
        alphas = torch.cat(alphas, dim=0)
        depths = torch.cat(depths, dim=0)
        return in_mesh, images, alphas, depths

    def get_noise_pred(self, latent_batches, prompt_embeds_batches, ctrl_images_batches, ctrl_depths_batches,
                       t, tile_weight, depth_weight, guidance_scale, extra_control_batches=None,
                       added_cond_kwargs_batches=None, adapter_scale=None):
        latent_size = latent_batches[0].size(-1)
        if ctrl_depths_batches is None:
            ctrl_depths_batches = [None] * len(latent_batches)
        if extra_control_batches is None:
            extra_control_batches = []
        noise_pred = []
        for i, (batch_latent_model_input,
                batch_prompt_embeds,
                batch_ctrl_images,
                batch_ctrl_depths, *batch_extra_control) in enumerate(zip(
                    latent_batches,
                    prompt_embeds_batches,
                    ctrl_images_batches,
                    ctrl_depths_batches, *extra_control_batches)):
            latent_shape = batch_latent_model_input.size()
            if latent_shape[2] == 2 * latent_shape[3]:
                cross_attention_kwargs = dict(num_cross_attn_imgs=2)
                unet_model_input = batch_latent_model_input.reshape(
                    *latent_shape[:2], 2, latent_shape[3], latent_shape[3]
                ).permute(0, 2, 1, 3, 4).reshape(latent_shape[0] * 2, latent_shape[1], latent_shape[3], latent_shape[3])
                controlnet_model_input = batch_latent_model_input[:, :, -latent_size:]
                unet_prompt_embeds = batch_prompt_embeds.unsqueeze(1).expand(-1, 2, -1, -1).reshape(
                    -1, *batch_prompt_embeds.shape[1:])
                controlnet_prompt_embeds = batch_prompt_embeds
            else:
                cross_attention_kwargs = None
                unet_model_input = controlnet_model_input = batch_latent_model_input
                unet_prompt_embeds = controlnet_prompt_embeds = batch_prompt_embeds
            added_cond_kwargs = None if added_cond_kwargs_batches is None else {
                k: v[i] for k, v in added_cond_kwargs_batches.items()}
            down_block_res_samples, mid_block_res_sample = self.controlnet(
                controlnet_model_input,
                t,
                encoder_hidden_states=controlnet_prompt_embeds,
                controlnet_cond=[batch_ctrl_images, batch_ctrl_depths] + list(batch_extra_control),
                conditioning_scale=[tile_weight, depth_weight] + [1.0] * len(batch_extra_control),
                guess_mode=False,
                added_cond_kwargs=added_cond_kwargs,
                return_dict=False)
            if latent_shape[2] == 2 * latent_shape[3]:
                down_block_res_samples = [
                    torch.stack([torch.zeros_like(res_sample), res_sample], dim=1).view(-1, *res_sample.shape[1:])
                    for res_sample in down_block_res_samples]
                mid_block_res_sample = torch.stack(
                    [torch.zeros_like(mid_block_res_sample), mid_block_res_sample], dim=1
                ).view(-1, *mid_block_res_sample.shape[1:])
            unet_out = self.unet(
                unet_model_input,
                t,
                encoder_hidden_states=unet_prompt_embeds,
                cross_attention_kwargs=cross_attention_kwargs,
                down_block_additional_residuals=down_block_res_samples,
                mid_block_additional_residual=mid_block_res_sample,
                added_cond_kwargs=added_cond_kwargs,
                return_dict=False)[0]
            if latent_shape[2] == 2 * latent_shape[3]:
                unet_out = unet_out.view(latent_shape[0], 2, latent_shape[1], latent_shape[3], latent_shape[3])[:, 1]
            noise_pred.append(unet_out)
        noise_pred = torch.cat(noise_pred, dim=0)
        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
        if adapter_scale is not None:
            noise_pred = adapter_scale * (noise_pred_text - noise_pred_uncond)
        else:
            noise_pred = guidance_scale * noise_pred_text + (1 - guidance_scale) * noise_pred_uncond
        return noise_pred

    def get_noise_pred_p1(
            self, latent_batches, prompt_embeds_batches, t, guidance_scale,
            ctrl_depths_batches=None, depth_weight=None, extra_control_batches=None,
            cond_noisy_latent_batches=None, added_cond_kwargs_batches=None):
        latent_size = latent_batches[0].size(-1)
        if extra_control_batches is None:
            extra_control_batches = []
        noise_pred = []
        dec_args = []
        dec_kwargs = []
        for i, (batch_latent_model_input,
                batch_prompt_embeds,
                batch_ctrl_depths,
                *batch_extra_control) in enumerate(zip(
                    latent_batches,
                    prompt_embeds_batches,
                    [None] * len(latent_batches) if ctrl_depths_batches is None else ctrl_depths_batches,
                    *extra_control_batches)):
            latent_shape = batch_latent_model_input.size()
            if latent_shape[2] == 2 * latent_shape[3]:
                cross_attention_kwargs = dict(num_cross_attn_imgs=2)
                unet_model_input = batch_latent_model_input.reshape(
                    *latent_shape[:2], 2, latent_shape[3], latent_shape[3]
                ).permute(0, 2, 1, 3, 4).reshape(latent_shape[0] * 2, latent_shape[1], latent_shape[3], latent_shape[3])
                controlnet_model_input = batch_latent_model_input[:, :, -latent_size:]
                unet_prompt_embeds = batch_prompt_embeds.unsqueeze(1).expand(-1, 2, -1, -1).reshape(
                    -1, *batch_prompt_embeds.shape[1:])
                controlnet_prompt_embeds = batch_prompt_embeds
            else:
                cross_attention_kwargs = None
                unet_model_input = controlnet_model_input = batch_latent_model_input
                unet_prompt_embeds = controlnet_prompt_embeds = batch_prompt_embeds
            added_cond_kwargs = None if added_cond_kwargs_batches is None else {
                k: v[i] for k, v in added_cond_kwargs_batches.items()}
            controlnet_skip = 2 if batch_ctrl_depths is None else 1
            if len(self.controlnet.nets) > controlnet_skip:
                controlnet = MultiControlNetModel(self.controlnet.nets[controlnet_skip:])
                down_block_additional_res_samples, mid_block_additional_res_sample = controlnet(
                    controlnet_model_input,
                    t,
                    encoder_hidden_states=controlnet_prompt_embeds,
                    controlnet_cond=list(batch_extra_control) if batch_ctrl_depths is None
                    else [batch_ctrl_depths] + list(batch_extra_control),
                    conditioning_scale=[1.0] * len(batch_extra_control) if batch_ctrl_depths is None
                    else [depth_weight] + [1.0] * len(batch_extra_control),
                    guess_mode=False,
                    added_cond_kwargs=added_cond_kwargs,
                    return_dict=False)
                if latent_shape[2] == 2 * latent_shape[3]:
                    down_block_additional_res_samples = [
                        torch.stack([torch.zeros_like(res_sample), res_sample], dim=1).view(-1, *res_sample.shape[1:])
                        for res_sample in down_block_additional_res_samples]
                    mid_block_additional_res_sample = torch.stack(
                        [torch.zeros_like(mid_block_additional_res_sample), mid_block_additional_res_sample], dim=1
                    ).view(-1, *mid_block_additional_res_sample.shape[1:])
            else:
                down_block_additional_res_samples = mid_block_additional_res_sample = None
            if cond_noisy_latent_batches is not None:
                assert cross_attention_kwargs is None
                ref_dict_enc = dict()
                ref_dict_dec = dict()
                emb_cond, down_block_res_samples_cond, sample_cond = unet_enc(
                    self.unet,
                    cond_noisy_latent_batches[i],
                    t,
                    encoder_hidden_states=unet_prompt_embeds,
                    added_cond_kwargs=added_cond_kwargs,
                    cross_attention_kwargs=dict(mode='w', ref_dict=ref_dict_enc))
                unet_dec(
                    self.unet,
                    emb_cond,
                    down_block_res_samples_cond,
                    sample_cond,
                    encoder_hidden_states=unet_prompt_embeds,
                    cross_attention_kwargs=dict(mode='w', ref_dict=ref_dict_dec))
            emb, down_block_res_samples, sample = unet_enc(
                self.unet,
                unet_model_input,
                t,
                encoder_hidden_states=unet_prompt_embeds,
                cross_attention_kwargs=cross_attention_kwargs if cond_noisy_latent_batches is None else dict(
                    mode='r', ref_dict=ref_dict_enc),
                added_cond_kwargs=added_cond_kwargs)
            dec_args.append((emb, down_block_res_samples, sample))
            dec_kwargs.append(dict(
                encoder_hidden_states=unet_prompt_embeds,
                cross_attention_kwargs=cross_attention_kwargs if cond_noisy_latent_batches is None else dict(
                    mode='m', ref_dict=ref_dict_dec),
                down_block_additional_residuals=down_block_additional_res_samples,
                mid_block_additional_residual=mid_block_additional_res_sample))
            unet_out = unet_dec(
                self.unet,
                *dec_args[-1],
                **dec_kwargs[-1])
            if latent_shape[2] == 2 * latent_shape[3]:
                unet_out = unet_out.view(latent_shape[0], 2, latent_shape[1], latent_shape[3], latent_shape[3])[:, 1]
            noise_pred.append(unet_out)
        noise_pred = torch.cat(noise_pred, dim=0)
        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
        noise_pred = guidance_scale * noise_pred_text + (1 - guidance_scale) * noise_pred_uncond
        return noise_pred, dec_args, dec_kwargs

    def get_noise_pred_p2(
            self, latent_batches, prompt_embeds_batches, dec_args, dec_kwargs, t, guidance_scale,
            ctrl_images_batches, tile_weight, ctrl_depths_batches=None, depth_weight=None,
            added_cond_kwargs_batches=None, guess_mode=False, adapter_scale=None, ctrl_text_embedding=True):
        latent_size = latent_batches[0].size(-1)
        noise_pred = []
        for i, (batch_latent_model_input,
                batch_prompt_embeds,
                dec_arg,
                dec_kwarg,
                batch_ctrl_images,
                batch_ctrl_depths) in enumerate(zip(
                    latent_batches,
                    prompt_embeds_batches,
                    dec_args,
                    dec_kwargs,
                    ctrl_images_batches,
                    [None] * len(latent_batches) if ctrl_depths_batches is None else ctrl_depths_batches)):
            latent_shape = batch_latent_model_input.size()
            if latent_shape[2] == 2 * latent_shape[3]:
                controlnet_model_input = batch_latent_model_input[:, :, -latent_size:]
            else:
                controlnet_model_input = batch_latent_model_input
            added_cond_kwargs = None if added_cond_kwargs_batches is None else {
                k: v[i] for k, v in added_cond_kwargs_batches.items()}
            controlnet = MultiControlNetModel(self.controlnet.nets[:1 if batch_ctrl_depths is None else 2])
            if ctrl_text_embedding:
                ctrl_prompt_embeds = batch_prompt_embeds
                ctrl_added_cond_kwargs = added_cond_kwargs
            else:
                ctrl_prompt_embeds = self.negative_prompt_embeds.to(controlnet_model_input).expand(
                    latent_shape[0], -1, -1)
                if hasattr(self, 'negative_added_cond_kwargs') and self.negative_added_cond_kwargs is not None:
                    ctrl_added_cond_kwargs = {
                        k: v.to(controlnet_model_input).expand(latent_shape[0], *[-1] * (v.dim() - 1))
                        for k, v in self.negative_added_cond_kwargs.items()}
                else:
                    ctrl_added_cond_kwargs = None
            down_block_additional_res_samples, mid_block_additional_res_sample = controlnet(
                controlnet_model_input,
                t,
                encoder_hidden_states=ctrl_prompt_embeds,
                controlnet_cond=[batch_ctrl_images] if batch_ctrl_depths is None
                    else [batch_ctrl_images, batch_ctrl_depths],
                conditioning_scale=[tile_weight] if batch_ctrl_depths is None else [tile_weight, depth_weight],
                guess_mode=guess_mode,
                added_cond_kwargs=ctrl_added_cond_kwargs,
                return_dict=False)
            if latent_shape[2] == 2 * latent_shape[3]:
                down_block_additional_res_samples = [
                    torch.stack([torch.zeros_like(res_sample), res_sample], dim=1).view(-1, *res_sample.shape[1:])
                    for res_sample in down_block_additional_res_samples]
                mid_block_additional_res_sample = torch.stack(
                    [torch.zeros_like(mid_block_additional_res_sample), mid_block_additional_res_sample], dim=1
                ).view(-1, *mid_block_additional_res_sample.shape[1:])
            down_block_additional_res_samples = [
                a + b for a, b in zip(down_block_additional_res_samples, dec_kwarg['down_block_additional_residuals'])
            ] if dec_kwarg['down_block_additional_residuals'] is not None else down_block_additional_res_samples
            mid_block_additional_res_sample = (
                mid_block_additional_res_sample if dec_kwarg['mid_block_additional_residual'] is None
                else mid_block_additional_res_sample + dec_kwarg['mid_block_additional_residual'])
            dec_kwarg_ = copy(dec_kwarg)
            dec_kwarg_.update(dict(
                down_block_additional_residuals=down_block_additional_res_samples,
                mid_block_additional_residual=mid_block_additional_res_sample))
            unet_out = unet_dec(
                self.unet,
                *dec_arg,
                **dec_kwarg_)
            if latent_shape[2] == 2 * latent_shape[3]:
                unet_out = unet_out.view(latent_shape[0], 2, latent_shape[1], latent_shape[3], latent_shape[3])[:, 1]
            noise_pred.append(unet_out)
        noise_pred = torch.cat(noise_pred, dim=0)
        noise_pred_uncond, noise_pred_cond = noise_pred.chunk(2)
        if adapter_scale is not None:
            noise_pred = adapter_scale * (noise_pred_cond - noise_pred_uncond)
        else:
            noise_pred = guidance_scale * noise_pred_cond + (1 - guidance_scale) * noise_pred_uncond
        return noise_pred

    def save_all_viz(self, out_dir, i, num_keep_views, tgt_images=None, latents_scaled=None, ctrl_images=None,
                     ctrl_depths=None, normals=None, alphas=None, pred_original_sample=None, diff_bs=12):
        latent_size = latents_scaled.size(-1)
        if num_keep_views == 0:
            num_keep_views = tgt_images.size(1)
        if latents_scaled is not None:
            noisy_images = []
            for batch_latents_scaled in latents_scaled[:num_keep_views, :, -latent_size:].split(diff_bs, dim=0):
                noisy_images.append(self.vae.decode(
                    batch_latents_scaled / self.vae.config.scaling_factor, return_dict=False)[0])
            noisy_images = torch.cat(noisy_images, dim=0) / 2 + 0.5
            for img_id, image in enumerate(noisy_images[:num_keep_views].permute(0, 2, 3, 1)):
                plt.imsave(osp.join(out_dir, '{:03d}_{:03d}_noisy.png'.format(i, img_id)),
                           (image.clamp(min=0, max=1) * 255).to(torch.uint8).cpu().numpy())
            if pred_original_sample is not None:
                pred_images = []
                for batch_latents_scaled in (
                        pred_original_sample[:num_keep_views, :, -latent_size:].split(diff_bs, dim=0)):
                    pred_images.append(self.vae.decode(
                        batch_latents_scaled / self.vae.config.scaling_factor, return_dict=False)[0])
                pred_images = torch.cat(pred_images, dim=0) / 2 + 0.5
                for img_id, image in enumerate(pred_images[:num_keep_views].permute(0, 2, 3, 1)):
                    plt.imsave(osp.join(out_dir, '{:03d}_{:03d}_pred.png'.format(i, img_id)),
                               (image.clamp(min=0, max=1) * 255).to(torch.uint8).cpu().numpy())
        if ctrl_images is not None:
            for img_id, image in enumerate(ctrl_images[:num_keep_views].permute(0, 2, 3, 1)):
                plt.imsave(osp.join(out_dir, '{:03d}_{:03d}_cond.png'.format(i, img_id)),
                           (image.clamp(min=0, max=1) * 255).to(torch.uint8).cpu().numpy())
        if ctrl_depths is not None:
            for img_id, image in enumerate(ctrl_depths[:num_keep_views].permute(0, 2, 3, 1)):
                plt.imsave(osp.join(out_dir, '{:03d}_{:03d}_ctrl_depth.png'.format(i, img_id)),
                           (image.clamp(min=0, max=1) * 255).to(torch.uint8).cpu().numpy())
        if normals is not None and alphas is not None:
            viz_normals = torch.cat([
                (normals - (1 - alphas) * alphas.new_tensor(self.normal_bg)[..., None, None]
                 ) / alphas.clamp(min=1e-4),
                alphas], dim=1).clamp(min=0, max=1)
            for img_id, image in enumerate(viz_normals[:num_keep_views].permute(0, 2, 3, 1)):
                plt.imsave(osp.join(out_dir, '{:03d}_{:03d}_cond_normal.png'.format(i, img_id)),
                           (image.clamp(min=0, max=1) * 255).to(torch.uint8).cpu().numpy().copy(order='C'))
        if tgt_images is not None:
            for img_id, image in enumerate(tgt_images[:num_keep_views].squeeze(0)):
                plt.imsave(osp.join(out_dir, '{:03d}_{:03d}_tgt.png'.format(i, img_id)),
                           (image.clamp(min=0, max=1) * 255).to(torch.uint8).cpu().numpy())
