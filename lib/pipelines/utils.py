import os
import numpy as np
import cv2
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as F_t
from PIL import Image
from pymatting.alpha.estimate_alpha_cf import estimate_alpha_cf
from pymatting.foreground.estimate_foreground_ml import estimate_foreground_ml
from pymatting.util.util import stack_images
from scipy.ndimage import binary_erosion
from typing import Tuple
from huggingface_hub import hf_hub_download
from transformers import CLIPTextModel, CLIPTextModelWithProjection, CLIPTokenizer, CLIPVisionModelWithProjection
from diffusers import AutoencoderKL, ControlNetModel, UNet2DConditionModel
from lib.models.decoders.mesh_renderer.base_mesh_renderer import MeshRenderer
from lib.models import SRVGGNetCompact
from lib.models.autoencoders.base_nerf import BaseNeRF
from lib.models.decoders.tonemapping import Tonemapping
from lib.models.segmentors.tracer_b7 import TracerUniversalB7
from lib.ops.rotation_conversions import matrix_to_quaternion


blender2opencv = np.array(
    [[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]],
    dtype=np.float32)


def pad_rgba_image(in_img, ratio=0.95, shift=[0, 0]):
    in_mask = np.where(in_img[:, :, 3] > 127)

    x1 = np.min(in_mask[1])
    x2 = np.max(in_mask[1]) + 1
    y1 = np.min(in_mask[0])
    y2 = np.max(in_mask[0]) + 1
    w = x2 - x1
    h = y2 - y1
    center_x = int(round((x1 + x2) / 2))
    center_y = int(round((y1 + y2) / 2))

    padded_size = int(round(max(w, h) / ratio))
    padded_center = int(round(padded_size / 2))
    padded_img = np.zeros((padded_size, padded_size, 4), dtype=np.uint8)
    padded_img[..., :3] = 255

    paste_x1 = padded_center - center_x + shift[0]
    paste_x2 = paste_x1 + in_img.shape[1]
    paste_y1 = padded_center - center_y + shift[1]
    paste_y2 = paste_y1 + in_img.shape[0]

    crop_l = max(0, -paste_x1)
    crop_r = max(0, paste_x2 - padded_size)
    crop_t = max(0, -paste_y1)
    crop_b = max(0, paste_y2 - padded_size)

    paste_x1 = max(0, paste_x1)
    paste_x2 = min(padded_size, paste_x2)
    paste_y1 = max(0, paste_y1)
    paste_y2 = min(padded_size, paste_y2)

    padded_img[paste_y1:paste_y2, paste_x1:paste_x2] = in_img[crop_t:in_img.shape[0]-crop_b, crop_l:in_img.shape[1]-crop_r]

    return Image.fromarray(padded_img)


def rgba_to_rgb(in_img, bg_color=(255, 255, 255)):
    in_img = np.array(in_img).astype(np.float32)
    alpha = in_img[..., 3:] / 255
    out_img = in_img[..., :3] * alpha + np.array(bg_color, dtype=np.float32) * (1 - alpha)
    return out_img.astype(np.uint8)


def do_segmentation(in_imgs, seg_model, sam_predictor=None, padding=0, to_np=False,
                    bg_color=None, color_threshold=0.25, sam_erosion=0):
    """
    Args:
        in_imgs (np.ndarray | torch.Tensor): input images, shape (N, H, W, 3) dtype uint8 for np.ndarray, or
            (N, 3, H, W) dtype float for torch.Tensor
    """
    if isinstance(in_imgs, np.ndarray):
        assert in_imgs.dtype == np.uint8
        in_imgs_torch = torch.from_numpy(in_imgs.transpose(0, 3, 1, 2).astype(np.float32) / 255)
    else:
        in_imgs_torch = in_imgs
    assert in_imgs_torch.size(1) == 3
    in_imgs_torch = in_imgs_torch.to(next(seg_model.parameters()).device)
    if padding > 0:  # padding helps to detect foreground objects
        masks = seg_model(F.pad(
            in_imgs_torch, (padding, padding, padding, padding), mode='replicate'
        ))[:, :, padding:-padding, padding:-padding]
    else:
        masks = seg_model(in_imgs_torch)
    if bg_color is not None:
        bg_color_torch = in_imgs_torch.new_tensor(bg_color)[..., None, None]
        non_fg_mask = torch.all(bg_color_torch - color_threshold <= in_imgs_torch, dim=1) \
                      & torch.all(in_imgs_torch <= bg_color_torch + color_threshold, dim=1)
        masks[~non_fg_mask.unsqueeze(1)] = 1
    if sam_predictor is None:
        if to_np:
            if isinstance(in_imgs, np.ndarray):
                in_imgs_np = in_imgs
            else:
                in_imgs_np = (in_imgs.permute(0, 2, 3, 1) * 255).round().to(torch.uint8).cpu().numpy()
            masks_np = (masks.permute(0, 2, 3, 1) * 255).round().to(torch.uint8).cpu().numpy()
            out_imgs = np.concatenate([in_imgs_np, masks_np], axis=-1)
        else:
            out_imgs = torch.cat([in_imgs_torch, masks], dim=1)
    else:
        masks_sam = []
        if isinstance(in_imgs, np.ndarray):
            in_imgs_np = in_imgs
        else:
            in_imgs_np = (in_imgs.permute(0, 2, 3, 1) * 255).round().to(torch.uint8).cpu().numpy()
        for i, in_img in enumerate(in_imgs_np):
            mask_binary = masks[i, 0] > 0.5
            alpha_x_nonzero = mask_binary.any(dim=0).nonzero()
            alpha_y_nonzero = mask_binary.any(dim=1).nonzero()
            sam_predictor.set_image(in_img)
            x1 = torch.amin(alpha_x_nonzero).item()
            x2 = torch.amax(alpha_x_nonzero).item() + 1
            y1 = torch.amin(alpha_y_nonzero).item()
            y2 = torch.amax(alpha_y_nonzero).item() + 1
            bbox = np.array([x1, y1, x2, y2])
            pred, _, _ = sam_predictor.predict(
                box=bbox,
                multimask_output=True)
            mask = pred[-1].astype(np.uint8)
            if sam_erosion > 0:
                kernel = np.ones((sam_erosion * 2 + 1, sam_erosion * 2 + 1), dtype=np.uint8)
                mask = cv2.erode(mask, kernel, iterations=1)
            masks_sam.append(mask[..., None])
        masks_sam = np.stack(masks_sam, axis=0)
        if bg_color is not None:
            bg_color_255 = np.array(bg_color, dtype=np.float32) * 255
            color_threshold_255 = color_threshold * 255
            in_imgs_np_float = in_imgs_np.astype(np.float32)
            non_fg_mask = np.all(bg_color_255 - color_threshold_255 <= in_imgs_np_float, axis=-1) \
                          & np.all(in_imgs_np_float <= bg_color_255 + color_threshold_255, axis=-1)
            masks_sam[~non_fg_mask] = 1
        if to_np:
            masks_sam = masks_sam * 255
            out_imgs = np.concatenate([in_imgs_np, masks_sam], axis=-1)
        else:
            masks_sam = torch.from_numpy(masks_sam).to(in_imgs_torch)
            out_imgs = torch.cat([in_imgs_torch, masks_sam.squeeze(-1).unsqueeze(1)], dim=1)
    return out_imgs


def do_segmentation_pil(in_imgs, *args, **kwargs):
    in_imgs_np = np.stack([np.asarray(in_img) for in_img in in_imgs], axis=0)
    kwargs['to_np'] = True
    out_imgs_np = do_segmentation(in_imgs_np, *args, **kwargs)
    return [Image.fromarray(out_img) for out_img in out_imgs_np]


def init_tet(nerf_model, nerf_code=None, density_thresh=5.0, resolution=128):
    local_path = os.path.abspath(os.path.join(__file__, f'../../../demo/tets/{resolution}_tets.npz'))
    if not os.path.exists(local_path):
        hf_hub_download(
            'Lakonik/mvedit_tets',
            f'{resolution}_tets.npz',
            local_dir=os.path.dirname(local_path))
    tets = np.load(local_path)
    verts = -torch.tensor(tets['vertices'], dtype=torch.float32, device='cuda') * 2  # covers [-1, 1]
    indices = torch.tensor(tets['indices'], dtype=torch.long, device='cuda')
    if nerf_code is None:
        nerf_code = [None]

    # init scale
    sigma = nerf_model.decoder.point_density_decode([verts], nerf_code)[0]  # tet_verts covers [-1, 1] now
    mask = sigma > density_thresh
    valid_verts = verts[mask]
    valid_verts_max = valid_verts.amax(dim=0) + 0.1
    valid_verts_min = valid_verts.amin(dim=0) - 0.1
    valid_verts_center = (valid_verts_max + valid_verts_min) / 2
    valid_verts_size = (valid_verts_max - valid_verts_min).max()
    verts = verts * (valid_verts_size / 2) + valid_verts_center

    # init sigma
    sigma = nerf_model.decoder.point_density_decode([verts], nerf_code)[0]  # new tet_verts
    sdf = (sigma - density_thresh).clamp(-1, 1)
    mask = (verts < -1).any(dim=-1) | (verts > 1).any(dim=-1)
    sdf[mask] = -1
    return verts, indices, sdf


def highpass(x, std=5, offset=0.5):
    return offset + x - F_t.gaussian_blur(x, int(round(std)) * 6 + 1, std)


def init_base_modules(device):
    mesh_renderer = MeshRenderer(
        near=0.01,
        far=100,
        ssaa=1,
        texture_filter='linear-mipmap-linear').to(device)
    tonemapping = Tonemapping()
    tonemapping.to(device=device)
    return mesh_renderer, tonemapping


def init_mvedit(device,
                image_enhancer=True,
                segmentation=True,
                nerf=True,
                controlnets=True,
                max_resolution=320,
                n_levels=12,
                dtype=torch.bfloat16):
    out_dict = dict()
    if image_enhancer:
        out_dict['image_enhancer'] = SRVGGNetCompact(
            num_in_ch=3, num_out_ch=3, num_feat=64, num_conv=32, upscale=4, act_type='prelu',
            pretrained='https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-general-x4v3.pth'
        ).eval().requires_grad_(False).to(dtype=dtype, device=device)
    if segmentation:
        out_dict['segmentation'] = TracerUniversalB7(torch_dtype=dtype).to(device)
    if nerf:
        out_dict['nerf'] = BaseNeRF(
            code_size=(3, 16, 160, 160),
            code_activation=dict(type='IdentityCode'),
            grid_size=128,
            encoder=None,
            bg_color=1.0,
            decoder=dict(
                type='iNGPDecoder',
                max_resolution=max_resolution,
                n_levels=n_levels,
                max_steps=1024,
                weight_culling_th=0.001),
            pixel_loss=dict(type='L1LossMod', loss_weight=1.2),
            patch_loss=dict(type='LPIPSLoss', loss_weight=1.2, net='vgg'),
            patch_size=128).to(device)
    if controlnets:
        controlnet_path = 'lllyasviel/control_v11f1e_sd15_tile'
        controlnet_depth_path = 'lllyasviel/control_v11f1p_sd15_depth'
        out_dict['controlnet'] = ControlNetModel.from_pretrained(
            controlnet_path, torch_dtype=dtype).requires_grad_(False).to(device)
        out_dict['controlnet_depth'] = ControlNetModel.from_pretrained(
            controlnet_depth_path, torch_dtype=dtype).requires_grad_(False).to(device)
    return out_dict


def init_instant3d(device, ckpt, variant='bf16', dtype=torch.bfloat16, local_files_only=False):
    unet = UNet2DConditionModel.from_pretrained(
        ckpt,
        subfolder='unet',
        variant=variant,
        local_files_only=local_files_only,
        torch_dtype=dtype).requires_grad_(False).to(device)
    vae = AutoencoderKL.from_pretrained(
        ckpt,
        subfolder='vae',
        variant=variant,
        local_files_only=local_files_only,
        torch_dtype=dtype).requires_grad_(False).to(device)
    text_encoder = CLIPTextModel.from_pretrained(
        ckpt,
        subfolder='text_encoder',
        variant=variant,
        local_files_only=local_files_only,
        torch_dtype=dtype).requires_grad_(False).to(device)
    text_encoder_2 = CLIPTextModelWithProjection.from_pretrained(
        ckpt,
        subfolder='text_encoder_2',
        variant=variant,
        local_files_only=local_files_only,
        torch_dtype=dtype).requires_grad_(False).to(device)
    tokenizer = CLIPTokenizer.from_pretrained(
        ckpt,
        local_files_only=local_files_only,
        subfolder='tokenizer')
    tokenizer_2 = CLIPTokenizer.from_pretrained(
        ckpt,
        local_files_only=local_files_only,
        subfolder='tokenizer_2')
    return dict(
        unet=unet,
        vae=vae,
        text_encoder=text_encoder,
        text_encoder_2=text_encoder_2,
        tokenizer=tokenizer,
        tokenizer_2=tokenizer_2)


def init_zero123plus(device, ckpt, dtype=torch.bfloat16, local_files_only=False):
    unet = UNet2DConditionModel.from_pretrained(
        ckpt,
        subfolder='unet',
        local_files_only=local_files_only,
        torch_dtype=dtype).requires_grad_(False).to(device)
    vae = AutoencoderKL.from_pretrained(
        ckpt,
        subfolder='vae',
        local_files_only=local_files_only,
        torch_dtype=dtype).requires_grad_(False).to(device)
    vision_encoder = CLIPVisionModelWithProjection.from_pretrained(
        ckpt,
        subfolder='vision_encoder',
        local_files_only=local_files_only,
        torch_dtype=dtype).requires_grad_(False).to(device)
    return dict(
        unet=unet,
        vae=vae,
        vision_encoder=vision_encoder)


def join_prompts(prompt_1, prompt_2, separator=', '):
    if prompt_1 and prompt_2:
        return f'{prompt_1}{separator}{prompt_2}'
    else:
        return prompt_1 or prompt_2


def zero123plus_postprocess(rgb_img: Image.Image, normal_img: Image.Image) -> Tuple[Image.Image, Image.Image]:
    normal_vecs_pred = np.array(normal_img, dtype=np.float64) / 255.0 * 2 - 1
    alpha_pred = np.linalg.norm(normal_vecs_pred, axis=-1)

    is_foreground = alpha_pred > 0.6
    is_background = alpha_pred < 0.2
    structure = np.ones(
        (4, 4), dtype=np.uint8
    )

    is_foreground = binary_erosion(is_foreground, structure=structure)
    is_background = binary_erosion(is_background, structure=structure, border_value=1)

    trimap = np.full(alpha_pred.shape, dtype=np.uint8, fill_value=128)
    trimap[is_foreground] = 255
    trimap[is_background] = 0

    img_normalized = np.array(rgb_img, dtype=np.float64) / 255.0
    trimap_normalized = trimap.astype(np.float64) / 255.0

    alpha = estimate_alpha_cf(img_normalized, trimap_normalized)
    foreground = estimate_foreground_ml(img_normalized, alpha)
    cutout = stack_images(foreground, alpha)

    cutout = np.clip(cutout * 255, 0, 255).astype(np.uint8)
    cutout = Image.fromarray(cutout)

    normal_vecs_pred = normal_vecs_pred / (np.linalg.norm(normal_vecs_pred, axis=-1, keepdims=True) + 1e-8)
    normal_vecs_pred = normal_vecs_pred * 0.5 + 0.5
    normal_vecs_pred = normal_vecs_pred * alpha[..., None] + 0.5 * (1 - alpha[..., None])
    normal_image_normalized = np.clip(normal_vecs_pred * 255, 0, 255).astype(np.uint8)

    return cutout, Image.fromarray(normal_image_normalized)


def get_camera_dists(camera_poses, cam_weights, device):
    num_cameras = camera_poses.size(0)
    cam_positions = camera_poses[:, :3, 3]  # (num_cameras, 3)
    cam_quaternions = matrix_to_quaternion(camera_poses[:, :3, :3])  # (num_cameras, 4)
    dot_quat = (cam_quaternions[:, None, None, :] @ cam_quaternions[None, :, :, None]).reshape(num_cameras, num_cameras)
    dists_half_theta = torch.acos(dot_quat.abs().clamp(max=1))
    # (num_cameras, num_cameras)
    dists = (torch.cdist(
        cam_positions[None], cam_positions[None], compute_mode='donot_use_mm_for_euclid_dist'
    ).squeeze(0) + dists_half_theta * 4)
    if cam_weights is not None:
        dists = dists * cam_weights[:, None]
    dists = dists + 999999 * torch.eye(len(dists), dtype=dists.dtype, device=device)
    return dists


def prune_cameras(dists, num_keep_views, max_num_cameras, device, pixel_dist=None):
    num_cameras = dists.size(0)
    keep_ids = torch.arange(num_cameras, device=device)
    for _ in range(num_cameras - max_num_cameras):
        view_importance = dists[num_keep_views:].amin(dim=1)
        if pixel_dist is not None:
            view_importance -= pixel_dist[num_keep_views:] * 0.05
        remove_id = view_importance.argmin() + num_keep_views
        keep_mask = torch.arange(len(keep_ids), device=device) != remove_id
        keep_ids = keep_ids[keep_mask]
        dists = dists[keep_mask][:, keep_mask]
        if pixel_dist is not None:
            pixel_dist = pixel_dist[keep_mask]
    return keep_ids, dists
