# modified from torch-ngp

import os
import random
import math
import copy
import json
import numpy as np
import trimesh
import torch
import torch.nn.functional as F
import cv2
import mmcv
import dearpygui.dearpygui as dpg
from scipy.spatial.transform import Rotation as R

from mmgen.models.builder import build_module
from mmgen.models.architectures.common import get_module_device
from mmgen.apis import set_random_seed  # isort:skip  # noqa
from .utils import extract_geometry, surround_views, vdb_utils, rgetattr, rsetattr
from .utils.nerf_utils import extract_fields
from lib.datasets.shapenet_srn import load_pose, load_intrinsics
from videoio import VideoWriter
import matplotlib.pyplot as plotlib


def load_img(path, background=[1., 1., 1.]):
    bgra = mmcv.imread(
        path, flag='unchanged', channel_order='bgr'
    ).astype(np.float32) / 255
    bgr = bgra[:, :, :3]
    rgb = bgr[:, :, ::-1]
    if bgra.shape[2] == 4:
        alpha = bgra[:, :, 3:4]
        rgb = rgb * alpha + np.array(background, dtype=np.float32) * (1 - alpha)
    return np.ascontiguousarray(rgb)


class OrbitCamera:
    def __init__(self, name, W, H, r=2., fovy=60., euler=[0, 0, 0]):
        self.name = name
        self.W = W
        self.H = H
        self.radius = r  # camera distance from center
        self.fovy = fovy  # in degree
        self.center = np.array([0, 0, 0], dtype=np.float32)  # look at this point
        self.default_rot = R.from_quat([0.5, -0.5, 0.5, -0.5])
        self.rot = copy.deepcopy(self.default_rot)
        self.up = np.array([0, 0, 1], dtype=np.float32)  # need to be normalized!

        self.set_euler(euler)

    # pose
    @property
    def pose(self):
        # first move camera to radius
        res = np.eye(4, dtype=np.float32)
        res[2, 3] -= self.radius
        # rotate
        rot = np.eye(4, dtype=np.float32)
        rot[:3, :3] = self.rot.as_matrix()
        res = rot @ res
        # translate
        res[:3, 3] -= self.center
        return res

    def set_pose(self, pose):
        self.rot = R.from_matrix(pose[:3, :3])
        self.center = -pose[:3, 3] - self.rot.as_matrix()[:3, 2] * self.radius

    @property
    def intrinsics(self):
        focal = self.H / (2 * np.tan(np.radians(self.fovy) / 2))
        return np.array([focal, focal, self.W / 2, self.H / 2])

    @property
    def euler(self):
        return (self.rot * self.default_rot.inv()).as_euler('xyz', degrees=True)

    def set_euler(self, euler):
        self.rot = R.from_euler('xyz', euler, degrees=True) * self.default_rot

    def orbit(self, dx, dy):
        # rotate along camera up/side axis!
        side = self.rot.as_matrix()[:3, 0]  # why this is side --> ? # already normalized.
        rotvec_x = self.up * np.radians(-0.1 * dx)
        rotvec_y = side * np.radians(-0.1 * dy)
        self.rot = R.from_rotvec(rotvec_x) * R.from_rotvec(rotvec_y) * self.rot

    def scale(self, delta):
        self.radius *= 1.1 ** (-delta)

    def pan(self, dx, dy, dz=0):
        # pan in camera coordinate system (careful on the sensitivity!)
        self.center += 0.0005 * self.rot.as_matrix()[:3, :3] @ np.array([dx, dy, dz])

    def pose2str(self):
        with np.printoptions(precision=3, suppress=True):
            return str(self.pose)


class SSDNeRFGUI:

    default_cam_fovy = 52.0
    default_cam_radius = 2.6
    default_cam_euler = [0.0, 23.0, -47.4]

    def __init__(self, model, W=512, H=512, max_spp=1, debug=True):
        self.W = W
        self.H = H
        self.max_spp = max_spp
        self.default_cam = OrbitCamera(
            'default', W, H, r=self.default_cam_radius, fovy=self.default_cam_fovy, euler=self.default_cam_euler)
        self.guide_cam = OrbitCamera(
            'guide', W, H, r=self.default_cam_radius, fovy=self.default_cam_fovy, euler=self.default_cam_euler)
        self.active_cam = self.default_cam
        self.debug = debug
        self.bg_color = torch.ones(3, dtype=torch.float32)  # default white bg
        self.step = 0  # training step

        self.model = model
        self.model_decoder = model.decoder_ema if model.decoder_use_ema else model.decoder
        self.model_diffusion = model.diffusion_ema if model.diffusion_use_ema else model.diffusion

        self.video_sec = 4
        self.video_fps = 30
        self.video_res = 256

        self.render_buffer = np.zeros((self.H, self.W, 3), dtype=np.float32)
        self.need_update = True  # camera moved, should reset accumulation
        self.spp = 1  # sample per pixel
        self.dt_gamma_scale = 0.0
        self.density_thresh = 0.1
        self.mode = 'image'  # choose from ['image', 'depth']

        self.mesh_resolution = 256
        self.mesh_threshold = 10
        self.scene_name = 'model_default'

        self.sampling_mode = 'text'
        self.pos_prompt = ''
        self.neg_prompt = ''
        self.diffusion_seed = -1
        self.diffusion_steps = model.test_cfg.get('num_timesteps', 20)
        self.diffusion_sampler = 'DDIM'
        self.cfg_scale = 1.0
        self.embed_guidance_scale = 0.0
        self.clip_denoised = True

        dtype = next(self.model_decoder.parameters()).dtype
        if self.model.init_code is None:
            self.code_buffer = torch.zeros(
                self.model.code_size, device=get_module_device(self.model), dtype=dtype)
        else:
            self.code_buffer = self.model.init_code.clone().to(dtype)
        _, self.density_bitfield = self.model.get_density(
            self.model_decoder, self.code_buffer[None],
            cfg=dict(density_thresh=self.density_thresh, density_step=16))

        self.dynamic_resolution = False
        self.downscale = 1

        self.image_enhancer = build_module(dict(
            type='SRVGGNetCompact',
            # num_in_ch=3, num_out_ch=3, num_feat=64, num_conv=16, upscale=4, act_type='prelu',
            num_in_ch=3, num_out_ch=3, num_feat=64, num_conv=32, upscale=4, act_type='prelu',
            # pretrained='https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-animevideov3.pth'
            pretrained='https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-general-x4v3.pth'
        )).half().eval().requires_grad_(False)
        if torch.cuda.is_available():
            self.image_enhancer.cuda()
        self.use_image_enhancer = False

        self.guide_image = None
        self.guide_image_overlay = None
        if 'guidance_gain' in model.test_cfg and 'n_inverse_rays' in model.test_cfg:
            self.guide_gain = model.test_cfg['guidance_gain'] / model.test_cfg['n_inverse_rays']
        else:
            self.guide_gain = 1.0
        self.overlay_opacity = 0.3

        self.code_viz_range = model.test_cfg.get('clip_range', [-1, 1])

        self.ddpm_loss_key = 'diffusion_ema.ddpm_loss.weight_scale' if model.diffusion_use_ema else 'diffusion.ddpm_loss.weight_scale'
        self.train_ddpm_weight = model.train_cfg_backup.get(
            self.ddpm_loss_key, rgetattr(model, self.ddpm_loss_key))
        self.loss_coef = 0.1  # ignore model's test cfg
        self.ft_optimizer = model.test_cfg.get(
            'optimizer', dict(type='Adam', lr=model.train_cfg['optimizer']['lr'] / 2, weight_decay=0.))
        self.ft_lr_scheduler = model.test_cfg.get(
            'lr_scheduler', dict(type='ExponentialLR', gamma=0.998))

        self.extrinsic_ndc_scale = 2.0  # default shapenet dataset value

        dpg.create_context()
        if self.debug:
            dpg.configure_app(manual_callback_management=True)
        self.register_dpg()
        self.test_step()

    def __del__(self):
        dpg.destroy_context()

    def prepare_buffer(self, outputs):
        if self.mode == 'image':
            return outputs['image']
        else:
            return np.expand_dims(outputs['depth'], -1).repeat(3, -1)

    def test_gui(self, pose, intrinsics, W, H, bg_color, spp, dt_gamma_scale, downscale):
        with torch.no_grad():
            self.model.bg_color = bg_color.to(self.code_buffer.device)
            if self.use_image_enhancer and self.mode == 'image':
                rH, rW = H // 2, W // 2
                intrinsics = intrinsics / 2
            else:
                rH, rW = H, W
            image, depth = self.model.render(
                self.model_decoder,
                self.code_buffer[None],
                self.density_bitfield[None], rH, rW,
                self.code_buffer.new_tensor(intrinsics * downscale, dtype=torch.float32)[None, None],
                self.code_buffer.new_tensor(pose, dtype=torch.float32)[None, None],
                cfg=dict(dt_gamma_scale=dt_gamma_scale))
            if self.use_image_enhancer and self.mode == 'image':
                image = self.image_enhancer(image[0].half().permute(0, 3, 1, 2))
                image = F.interpolate(image, size=(H, W), mode='area').permute(0, 2, 3, 1)[None].float()
            results = dict(
                image=image[0, 0],
                depth=depth[0, 0])
            if downscale != 1:
                results['image'] = F.interpolate(
                    results['image'].permute(2, 0, 1)[None], size=(H, W), mode='nearest'
                ).permute(0, 2, 3, 1).reshape(H, W, 3)
                results['depth'] = F.interpolate(results['depth'][None, None], size=(H, W), mode='nearest').reshape(H, W)
            if self.overlay_opacity > 0.003 and self.guide_image is not None and self.active_cam.name == 'guide':
                results['image'] = self.guide_image_overlay * self.overlay_opacity + results['image'] * (1 - self.overlay_opacity)
            results['image'] = results['image'].cpu().numpy()
            results['depth'] = results['depth'].cpu().numpy()
        return results

    def update_params(self):
        with torch.no_grad():
            self.density_bitfield = self.model.get_density(
                self.model_decoder, self.code_buffer[None],
                cfg=dict(density_thresh=self.density_thresh, density_step=16))[1].squeeze(0)

    def test_step(self):
        # TODO: seems we have to move data from GPU --> CPU --> GPU?

        if self.need_update or self.spp < self.max_spp:

            starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            starter.record()

            outputs = self.test_gui(
                self.active_cam.pose, self.active_cam.intrinsics,
                self.W, self.H, self.bg_color, self.spp, self.dt_gamma_scale, self.downscale)

            ender.record()
            torch.cuda.synchronize()
            t = starter.elapsed_time(ender)

            # update dynamic resolution
            if self.dynamic_resolution:
                # max allowed infer time per-frame is 200 ms
                full_t = t / (self.downscale ** 2)
                downscale = min(1, max(1 / 4, math.sqrt(200 / full_t)))
                if downscale > self.downscale * 1.2 or downscale < self.downscale * 0.8:
                    self.downscale = downscale

            if self.need_update:
                self.render_buffer = np.ascontiguousarray(self.prepare_buffer(outputs))
                self.spp = 1
                self.need_update = False
            else:
                self.render_buffer = (self.render_buffer * self.spp + self.prepare_buffer(outputs)) / (self.spp + 1)
                self.spp += 1

            dpg.set_value('_log_infer_time', f'{t:.4f}ms ({int(1000 / t)} FPS)')
            dpg.set_value('_log_resolution', f'{int(self.downscale * self.W)}x{int(self.downscale * self.H)}')
            dpg.set_value('_log_spp', self.spp)
            dpg.set_value('_log_scene_name', self.scene_name)
            dpg.set_value('_texture', self.render_buffer)

    def register_dpg(self):

        ### register texture

        with dpg.texture_registry(show=False):
            dpg.add_raw_texture(self.W, self.H, self.render_buffer, format=dpg.mvFormat_Float_rgb, tag='_texture')

        ### register window

        # the rendered image, as the primary window
        with dpg.window(tag='_primary_window', width=self.W, height=self.H):

            # add the texture
            dpg.add_image('_texture')

        dpg.set_primary_window('_primary_window', True)

        def update_camera_status():
            if self.debug:
                dpg.set_value('_log_pose', self.active_cam.pose2str())
            dpg.set_value('fov', self.active_cam.fovy)
            dpg.set_value('radius', self.active_cam.radius)
            euler = self.active_cam.euler
            dpg.set_value('roll', euler[0])
            dpg.set_value('elevation', euler[1])
            dpg.set_value('azimuth', euler[2])
            center = self.active_cam.center
            dpg.set_value('center_x', center[0])
            dpg.set_value('center_y', center[1])
            dpg.set_value('center_z', center[2])

        # control window
        with dpg.window(label='Control', tag='_control_window', width=380, height=self.H, pos=[self.W, 0]):

            # button theme
            with dpg.theme() as theme_button:
                with dpg.theme_component(dpg.mvButton):
                    dpg.add_theme_color(dpg.mvThemeCol_Button, (23, 3, 18))
                    dpg.add_theme_color(dpg.mvThemeCol_ButtonHovered, (51, 3, 47))
                    dpg.add_theme_color(dpg.mvThemeCol_ButtonActive, (83, 18, 83))
                    dpg.add_theme_style(dpg.mvStyleVar_FrameRounding, 5)
                    dpg.add_theme_style(dpg.mvStyleVar_FramePadding, 3, 3)

            # time
            with dpg.group(horizontal=True):
                dpg.add_text('Infer time: ')
                dpg.add_text('no data', tag='_log_infer_time')

            with dpg.group(horizontal=True):
                dpg.add_text('SPP: ')
                dpg.add_text('1', tag='_log_spp')

            with dpg.collapsing_header(label='SSDNeRF', default_open=True):

                def callback_diffusion_generate(sender, app_data):
                    diffusion_seed = random.randint(0, 2**31) if self.diffusion_seed == -1 else self.diffusion_seed
                    set_random_seed(diffusion_seed, deterministic=True)
                    noise = torch.randn((1,) + self.model.code_size)
                    self.model_diffusion.test_cfg['num_timesteps'] = self.diffusion_steps
                    self.model_diffusion.sample_method = self.diffusion_sampler
                    self.model_diffusion.test_cfg['cfg_scale'] = self.cfg_scale
                    self.model_diffusion.test_cfg['embed_guidance_scale'] = self.embed_guidance_scale
                    self.model_diffusion.test_cfg['clip_denoised'] = self.clip_denoised
                    device = get_module_device(self.model)
                    data = dict(
                        noise=noise.to(device),
                        scene_id=[0],
                        scene_name=['seed_{}'.format(diffusion_seed)],
                        prompts=[self.pos_prompt],
                        neg_prompts=[self.neg_prompt])
                    if self.sampling_mode == 'image_text':
                        data['extra_cond_img'] = self.extra_cond_image
                        data['extra_pose_cond'] = torch.tensor(self.guide_cam.pose[:3].reshape(1, 12)).to(device).float()
                    if self.sampling_mode in ['guide', 'optim']:
                        scale = max(self.guide_image.shape[1] / self.W, self.guide_image.shape[0] / self.H)
                        data['cond_imgs'] = self.guide_image[None, None]
                        data['cond_intrinsics'] = torch.tensor(
                            self.guide_cam.intrinsics[None, None] * np.array([
                                scale, scale,
                                self.guide_image.size(1) / self.W, self.guide_image.size(0) / self.H])
                        ).to(device).float()
                        data['cond_poses'] = torch.tensor(self.guide_cam.pose[None, None]).to(device).float()
                        self.model_diffusion.test_cfg['n_inverse_rays'] = self.guide_image.numel()
                        self.model.test_cfg['loss_coef'] = self.loss_coef / self.guide_image.numel()
                    if self.sampling_mode == 'guide':
                        self.model_diffusion.test_cfg['guidance_gain'] = self.guide_gain * self.guide_image.numel()
                    if self.sampling_mode == 'optim':
                        self.model.test_cfg['optimizer'] = self.ft_optimizer
                        self.model.test_cfg['lr_scheduler'] = self.ft_lr_scheduler
                        optim_kwargs = dict(
                            code_=self.model.code_activation.inverse(self.code_buffer[None]))
                    else:
                        optim_kwargs = dict()

                    with torch.no_grad():
                        sample_fun = getattr(self.model, 'val_' + self.sampling_mode)
                        code, density_grid, density_bitfield = sample_fun(
                            data, show_pbar=True, **optim_kwargs)
                    self.code_buffer = code[0].to(self.code_buffer)
                    self.density_bitfield = density_bitfield[0]
                    self.scene_name = 'seed_{}'.format(diffusion_seed)
                    self.need_update = True
                    print("Peak VRAM usage:", int(torch.cuda.max_memory_allocated() / 1024 ** 2 + 1), "(M)")

                def callback_change_mode(sender, app_data):
                    self.sampling_mode = app_data

                def callback_change_sampler(sender, app_data):
                    self.diffusion_sampler = app_data

                with dpg.group(horizontal=True):
                    dpg.add_combo(
                        ('text', 'image_text', 'uncond', 'guide', 'optim'), label='mode', default_value=self.sampling_mode,
                        width=75, callback=callback_change_mode)
                    dpg.add_combo(
                        self.model_diffusion.available_samplers, label='sampler', default_value=self.diffusion_sampler,
                        width=190, callback=callback_change_sampler)

                def callback_set_pos_prompt(sender, app_data):
                    self.pos_prompt = app_data

                dpg.add_input_text(
                    label='prompt', width=290, default_value=self.pos_prompt, callback=callback_set_pos_prompt)

                def callback_set_neg_prompt(sender, app_data):
                    self.neg_prompt = app_data

                dpg.add_input_text(
                    label='neg prompt', width=290, default_value=self.neg_prompt, callback=callback_set_neg_prompt)

                def callback_set_cfg_scale(sender, app_data):
                    self.cfg_scale = app_data

                dpg.add_input_float(
                    label='prompt scale', width=100, default_value=self.cfg_scale, callback=callback_set_cfg_scale)

                def callback_set_embed_guidance_scale(sender, app_data):
                    self.embed_guidance_scale = app_data
                
                dpg.add_input_float(
                    label='embed guidance', width=100, default_value=self.embed_guidance_scale, callback=callback_set_embed_guidance_scale)

                def callback_set_diffusion_seed(sender, app_data):
                    self.diffusion_seed = app_data

                def callback_set_diffusion_steps(sender, app_data):
                    self.diffusion_steps = app_data

                def callback_set_clip_denoised(sender, app_data):
                    self.clip_denoised = app_data

                dpg.add_checkbox(label='clip denoised', callback=callback_set_clip_denoised,
                                 default_value=self.clip_denoised)

                with dpg.group(horizontal=True):
                    dpg.add_button(label='Generate', callback=callback_diffusion_generate)
                    dpg.add_input_int(
                        label='seed', width=130, min_value=-1, max_value=2**31 - 1, min_clamped=True, max_clamped=True,
                        default_value=self.diffusion_seed, callback=callback_set_diffusion_seed, tag='seed')
                    dpg.add_input_int(
                        label='steps', width=80, min_value=1, max_value=1000, min_clamped=True, max_clamped=True,
                        default_value=self.diffusion_steps, callback=callback_set_diffusion_steps)

                def callback_save_scene(sender, app_data):
                    path = app_data['file_path_name']
                    out = dict(
                        param=dict(
                            code=self.code_buffer.cpu(),
                            density_bitfield=self.density_bitfield.cpu()))
                    torch.save(out, path)

                with dpg.file_dialog(directory_selector=False, show=False, width=450, height=400,
                                     callback=callback_save_scene, tag='save_scene_dialog'):
                    dpg.add_file_extension('.pth')

                with dpg.group(horizontal=True):
                    dpg.add_button(label='Save scene', callback=lambda: dpg.show_item('save_scene_dialog'))

                # scene selector
                def callback_load_scene(sender, app_data):
                    self.scene_name = os.path.splitext(app_data['file_name'])[0]
                    scene = torch.load(app_data['file_path_name'], map_location='cpu')
                    self.code_buffer = (
                        scene['param']['code'] if 'code' in scene['param']
                        else self.model.code_activation(scene['param']['code_'])).to(self.code_buffer)
                    self.update_params()
                    print('Loaded scene: ' + self.scene_name)
                    self.need_update = True

                def callback_recover_seed(sender, app_data):
                    if self.scene_name.startswith('seed_'):
                        seed = int(self.scene_name[5:])
                        self.diffusion_seed = seed
                        dpg.set_value('seed', seed)
                        print('Recovered seed: ' + str(seed))
                    else:
                        print('Failed to recover seed: ' + self.scene_name)

                with dpg.file_dialog(directory_selector=False, show=False, width=450, height=400,
                                     callback=callback_load_scene, tag='scene_selector_dialog'):
                    dpg.add_file_extension('.pth')

                with dpg.group(horizontal=True):
                    dpg.add_button(label='Load scene', callback=lambda: dpg.show_item('scene_selector_dialog'))
                    dpg.add_text(tag='_log_scene_name')
                    dpg.add_button(label='Recover seed', callback=callback_recover_seed)

                # save geometry
                def callback_export_mesh(sender, app_data):
                    self.export_mesh(app_data['file_path_name'])

                def callback_export_vdb(sender, app_data):
                    self.export_vdb(app_data['file_path_name'])

                def callback_save_code(sender, app_data):
                    dir_path = app_data['file_path_name']
                    assert os.path.isdir(dir_path), dir_path + ' is not a directory'
                    self.model_decoder.visualize(
                        self.code_buffer[None], [self.scene_name], dir_path, code_range=self.code_viz_range)
                
                def callback_set_vmin(sender, app_data):
                    self.code_viz_range[0] = app_data
                    
                def callback_set_vmax(sender, app_data):
                    self.code_viz_range[1] = app_data

                def callback_set_mesh_resolution(sender, app_data):
                    self.mesh_resolution = app_data

                def callback_set_mesh_threshold(sender, app_data):
                    self.mesh_threshold = app_data

                def callback_set_video_resolution(sender, app_data):
                    self.video_res = app_data

                def callback_set_video_sec(sender, app_data):
                    self.video_sec = app_data

                def callback_export_screenshot(sender, app_data):
                    path = app_data['file_path_name']
                    cv2.imwrite(path, np.round(self.render_buffer[..., ::-1] * 255).astype(np.uint8))

                def callback_export_multi_view(sender, app_data):
                    dir_path = app_data['file_path_name']
                    assert os.path.isdir(dir_path), dir_path + ' is not a directory'
                    self.export_multi_view_data(dir_path)

                def callback_export_video(sender, app_data):
                    path = app_data['file_path_name']
                    num_frames = int(round(self.video_fps * self.video_sec))
                    tmp_cam = OrbitCamera(
                        'tmp', self.video_res, self.video_res,
                        r=self.default_cam_radius, fovy=self.default_cam_fovy, euler=self.default_cam_euler)
                    camera_poses = surround_views(
                        self.code_buffer.new_tensor(tmp_cam.pose, dtype=torch.float32), num_frames=num_frames)
                    writer = VideoWriter(
                        path,
                        resolution=(self.video_res, self.video_res),
                        lossless=False,
                        fps=self.video_fps)
                    bs = 4
                    device = self.code_buffer.device
                    with torch.no_grad():
                        prog = mmcv.ProgressBar(num_frames)
                        prog.start()
                        for pose_batch in camera_poses.split(bs, dim=0):
                            intrinsics = self.code_buffer.new_tensor(
                                tmp_cam.intrinsics[None], dtype=torch.float32).expand(pose_batch.size(0), -1)[None]
                            res = self.video_res
                            if self.use_image_enhancer:
                                res = res // 2
                                intrinsics = intrinsics * (res / self.video_res)
                            image_batch, depth = self.model.render(
                                self.model_decoder,
                                self.code_buffer[None],
                                self.density_bitfield[None], res, res,
                                intrinsics,
                                pose_batch.to(device)[None])
                            if self.use_image_enhancer:
                                image_batch = self.image_enhancer(image_batch[0].half().permute(0, 3, 1, 2).clamp(min=0, max=1))
                                image_batch = F.interpolate(
                                    image_batch, size=(self.video_res, self.video_res), mode='area'
                                ).permute(0, 2, 3, 1)[None]
                            for image in torch.round(image_batch[0].clamp(min=0, max=1) * 255).to(torch.uint8).cpu().numpy():
                                writer.write(image)
                            prog.update(bs)
                    writer.close()

                with dpg.file_dialog(directory_selector=False, show=False, width=450, height=400,
                                     callback=callback_export_mesh, tag='export_mesh_dialog'):
                    dpg.add_file_extension('.stl')
                    dpg.add_file_extension('.dict')
                    dpg.add_file_extension('.json')
                    dpg.add_file_extension('.glb')
                    dpg.add_file_extension('.obj')
                    dpg.add_file_extension('.gltf')
                    dpg.add_file_extension('.dict64')
                    dpg.add_file_extension('.msgpack')
                    dpg.add_file_extension('.stl_ascii')

                with dpg.file_dialog(directory_selector=False, show=False, width=450, height=400,
                                     callback=callback_export_vdb, tag='export_vdb_dialog'):
                    dpg.add_file_extension('.vdb')

                with dpg.file_dialog(directory_selector=False, show=False, width=450, height=400,
                                     callback=callback_save_code, tag='save_code_dialog'):
                    dpg.add_file_extension('.')

                with dpg.file_dialog(directory_selector=False, show=False, width=450, height=400,
                                     callback=callback_export_screenshot, tag='export_screenshot_dialog'):
                    dpg.add_file_extension('.png')

                with dpg.file_dialog(directory_selector=False, show=False, width=450, height=400,
                                     callback=callback_export_multi_view, tag='export_multi_view_dialog'):
                    dpg.add_file_extension('.')

                with dpg.file_dialog(directory_selector=False, show=False, width=450, height=400,
                                     callback=callback_export_video, tag='export_video_dialog'):
                    dpg.add_file_extension('.mp4')

                with dpg.group(horizontal=True):
                    dpg.add_button(label='Export screenshot', callback=lambda: dpg.show_item('export_screenshot_dialog'))
                    dpg.add_button(label='Export multi-view', callback=lambda: dpg.show_item('export_multi_view_dialog'))

                with dpg.group(horizontal=True):
                    dpg.add_button(label='Export video', callback=lambda: dpg.show_item('export_video_dialog'))
                    dpg.add_input_int(
                        label='res', width=90, min_value=4, max_value=1024, min_clamped=True, max_clamped=True,
                        default_value=self.video_res, callback=callback_set_video_resolution)
                    dpg.add_input_float(
                        label='len', width=100, min_value=0, max_value=10, min_clamped=True, max_clamped=True,
                        default_value=self.video_sec, callback=callback_set_video_sec, format='%.1f sec')

                with dpg.group(horizontal=True):
                    dpg.add_button(label='Export mesh', callback=lambda: dpg.show_item('export_mesh_dialog'))
                    dpg.add_input_int(
                        label='res', width=90, min_value=4, max_value=1024, min_clamped=True, max_clamped=True,
                        default_value=self.mesh_resolution, callback=callback_set_mesh_resolution)
                    dpg.add_input_float(
                        label='thr', width=100, min_value=0, max_value=1000, min_clamped=True, max_clamped=True,
                        format='%.2f', default_value=self.mesh_threshold, callback=callback_set_mesh_threshold)

                dpg.add_button(label='Export volume', callback=lambda: dpg.show_item('export_vdb_dialog'))

                with dpg.group(horizontal=True):
                    dpg.add_button(label='Export code viz', callback=lambda: dpg.show_item('save_code_dialog'))
                    dpg.add_input_float(
                        label='vmin', width=85, format='%.1f',
                        default_value=self.code_viz_range[0], callback=callback_set_vmin)
                    dpg.add_input_float(
                        label='vmax', width=85, format='%.1f',
                        default_value=self.code_viz_range[1], callback=callback_set_vmax)

                with dpg.collapsing_header(label='Guidance/finetuning options', default_open=False):

                    def callback_load_guide_image(sender, app_data):
                        img = load_img(app_data['file_path_name'], [0.5, 0.5, 0.5])
                        img = (img - 0.5) * 1.2
                        self.extra_cond_image = torch.tensor(
                            cv2.resize(img, [384, 384], interpolation=cv2.INTER_LINEAR)
                        )[None].float().to(self.code_buffer.device)
                        self.guide_image = torch.tensor(
                            load_img(app_data['file_path_name'])).float().to(self.code_buffer.device)
                        bg = self.bg_color.to(self.guide_image.device)[:, None, None]
                        scale = min(self.W / self.guide_image.shape[1], self.H / self.guide_image.shape[0])
                        grid = F.affine_grid(
                            torch.tensor(
                                [[self.W / (self.guide_image.shape[1] * scale), 0, 0],
                                 [0, self.H / (self.guide_image.shape[0] * scale), 0]],
                                dtype=self.guide_image.dtype, device=self.guide_image.device)[None],
                            [1, 3, self.H, self.W], align_corners=False)
                        self.guide_image_overlay = (F.grid_sample(
                            self.guide_image.permute(2, 0, 1)[None] - bg,
                            grid, mode='nearest', padding_mode='zeros', align_corners=False,
                        ) + bg).squeeze(0).permute(1, 2, 0)
                        self.active_cam = self.guide_cam
                        update_camera_status()
                        dpg.set_value('cam_combo', 'guide')
                        self.need_update = True

                    with dpg.file_dialog(directory_selector=False, show=False, width=450, height=400,
                                         callback=callback_load_guide_image, tag='guide_image_dialog'):
                        dpg.add_file_extension('.png')

                    def callback_set_guide_gain(sender, app_data):
                        self.guide_gain = app_data

                    def callback_set_guide_overlay(sender, app_data):
                        self.overlay_opacity = app_data
                        self.need_update = True

                    def callback_set_snr_power(sender, app_data):
                        self.model_diffusion.test_cfg['snr_weight_power'] = app_data

                    def callback_set_langevin_steps(sender, app_data):
                        self.model_diffusion.test_cfg['langevin_steps'] = app_data

                    def callback_set_langevin_delta(sender, app_data):
                        self.model_diffusion.test_cfg['langevin_delta'] = app_data

                    def callback_set_ddpm_loss_gain(sender, app_data):
                        rsetattr(self.model, self.ddpm_loss_key, app_data * self.train_ddpm_weight)

                    def callback_set_learning_rate(sender, app_data):
                        self.ft_optimizer['lr'] = app_data

                    def callback_set_outer_loop_steps(sender, app_data):
                        self.model.test_cfg['n_inverse_steps'] = app_data

                    def callback_set_inner_loop_steps(sender, app_data):
                        self.model.test_cfg['extra_scene_step'] = app_data - 1

                    with dpg.group(horizontal=True):
                        dpg.add_button(label='load input img', callback=lambda: dpg.show_item('guide_image_dialog'))
                        dpg.add_slider_float(
                            label='overlay', min_value=0.0, max_value=1.0, width=170,
                            default_value=self.overlay_opacity, callback=callback_set_guide_overlay)
                    dpg.add_text('Guidance params:')
                    dpg.add_input_float(
                        label='guidance gain', width=130, default_value=self.guide_gain, callback=callback_set_guide_gain)
                    dpg.add_input_float(
                        label='SNR power', width=100,
                        default_value=self.model_diffusion.test_cfg.get(
                            'snr_weight_power', self.model_diffusion.timestep_sampler.power),
                        format='%.3f', callback=callback_set_snr_power)
                    with dpg.group(horizontal=True):
                        dpg.add_input_int(
                            label='langevin steps', width=90, default_value=self.model_diffusion.test_cfg.get('langevin_steps', 0),
                            min_value=0, max_value=100, min_clamped=True, callback=callback_set_langevin_steps)
                        dpg.add_input_float(
                            label='delta', width=100, default_value=self.model_diffusion.test_cfg.get('langevin_delta', 0.4),
                            format='%.2f', callback=callback_set_langevin_delta)

                    dpg.add_text('Finetuning optim params:')
                    dpg.add_input_float(
                        label='ddpm loss gain', width=130,
                        default_value=rgetattr(self.model, self.ddpm_loss_key) / self.train_ddpm_weight,
                        callback=callback_set_ddpm_loss_gain)
                    dpg.add_input_float(
                        label='learning rate', width=130, default_value=self.ft_optimizer['lr'], format='%.2e',
                        callback=callback_set_learning_rate)
                    with dpg.group(horizontal=True):
                        dpg.add_input_int(
                            label='Outer steps', width=90, default_value=self.model.test_cfg.get('n_inverse_steps', 25),
                            min_value=0, max_value=1000, min_clamped=True, callback=callback_set_outer_loop_steps)
                        dpg.add_input_int(
                            label='Inner steps', width=90, default_value=self.model.test_cfg.get('extra_scene_step', 3) + 1,
                            min_value=1, max_value=100, min_clamped=True, callback=callback_set_inner_loop_steps)

            with dpg.collapsing_header(label='Camera options', default_open=True):

                def callback_set_cam(sender, app_data):
                    self.active_cam = getattr(self, app_data + '_cam')
                    update_camera_status()
                    self.need_update = True

                def callback_reset_camera(sender, app_data):
                    self.active_cam.fovy = self.default_cam_fovy
                    self.active_cam.radius = self.default_cam_radius
                    self.active_cam.set_euler(self.default_cam_euler)
                    self.active_cam.center = np.array([0, 0, 0], dtype=np.float32)
                    update_camera_status()
                    self.need_update = True

                with dpg.group(horizontal=True):
                    dpg.add_combo(
                        ['default', 'guide'], label='camera', width=150,
                        default_value=self.active_cam.name, callback=callback_set_cam, tag='cam_combo')
                    dpg.add_button(label='Reset camera', callback=callback_reset_camera)

                def callback_set_fovy(sender, app_data):
                    self.active_cam.fovy = app_data
                    update_camera_status()
                    self.need_update = True

                def callback_set_cam_r(sender, app_data):
                    self.active_cam.radius = app_data
                    update_camera_status()
                    self.need_update = True

                def callback_set_euler(sender, app_data, axis):
                    euler = self.active_cam.euler
                    euler[axis] = app_data
                    self.active_cam.set_euler(euler)
                    update_camera_status()
                    self.need_update = True

                def callback_set_center(sender, app_data, axis):
                    self.active_cam.center[axis] = app_data
                    update_camera_status()
                    self.need_update = True

                dpg.add_slider_float(
                    label='FoV (vertical)', min_value=1, max_value=120, clamped=True, format='%.1f deg',
                    default_value=self.active_cam.fovy, callback=callback_set_fovy, tag='fov')
                dpg.add_slider_float(
                    label='radius', min_value=1.0, max_value=5.0, format='%.2f',
                    default_value=self.active_cam.radius, callback=callback_set_cam_r, tag='radius')
                dpg.add_slider_float(
                    label='azimuth', min_value=-180, max_value=180, clamped=True, format='%.1f deg',
                    default_value=self.active_cam.euler[2],
                    callback=lambda x, y: callback_set_euler(x, y, 2), tag='azimuth')
                dpg.add_slider_float(
                    label='elevation', min_value=-89, max_value=89, clamped=True, format='%.1f deg',
                    default_value=self.active_cam.euler[1],
                    callback=lambda x, y: callback_set_euler(x, y, 1), tag='elevation')
                dpg.add_slider_float(
                    label='roll', min_value=-180, max_value=180, clamped=True, format='%.1f deg',
                    default_value=self.active_cam.euler[0],
                    callback=lambda x, y: callback_set_euler(x, y, 0), tag='roll')
                dpg.add_text('Orbit center:')
                with dpg.group(horizontal=True):
                    dpg.add_input_float(
                        width=110, format='x: %.2f', tag='center_x',
                        default_value=self.active_cam.center[0], callback=lambda x, y: callback_set_center(x, y, 0))
                    dpg.add_input_float(
                        width=110, format='y: %.2f', tag='center_y',
                        default_value=self.active_cam.center[1], callback=lambda x, y: callback_set_center(x, y, 1))
                    dpg.add_input_float(
                        width=110, format='z: %.2f', tag='center_z',
                        default_value=self.active_cam.center[2], callback=lambda x, y: callback_set_center(x, y, 2))

                def callback_load_intrinsic(sender, app_data):
                    fx, fy, cx, cy, h, w = load_intrinsics(app_data['file_path_name'])
                    assert fx == fy and cx == w / 2 and cy == h / 2, 'GUI supports only rectified images'
                    self.active_cam.fovy = np.rad2deg(2 * np.arctan2(h / 2, fy))
                    update_camera_status()
                    self.need_update = True

                def callback_load_extrinsic(sender, app_data):
                    c2w = load_pose(app_data['file_path_name'])
                    cam_to_ndc = np.concatenate([c2w[:3, :3], c2w[:3, 3:] * self.extrinsic_ndc_scale], axis=-1)
                    self.active_cam.set_pose(cam_to_ndc)
                    update_camera_status()
                    self.need_update = True

                def callback_set_ndc_scale(sender, app_data):
                    self.extrinsic_ndc_scale = app_data

                with dpg.file_dialog(directory_selector=False, show=False, width=450, height=400,
                                     callback=callback_load_intrinsic, tag='load_intrinsic_dialog'):
                    dpg.add_file_extension('.txt')

                with dpg.file_dialog(directory_selector=False, show=False, width=450, height=400,
                                     callback=callback_load_extrinsic, tag='load_extrinsic_dialog'):
                    dpg.add_file_extension('.txt')

                with dpg.group(horizontal=True):
                    dpg.add_button(label='Load intrinsic', callback=lambda: dpg.show_item('load_intrinsic_dialog'))
                    dpg.add_button(label='Load pose', callback=lambda: dpg.show_item('load_extrinsic_dialog'))
                    dpg.add_input_float(
                        label='NDC scale', width=80, format='%.1f',
                        default_value=self.extrinsic_ndc_scale, callback=callback_set_ndc_scale)

            with dpg.collapsing_header(label='Render options', default_open=True):

                # dynamic rendering resolution
                with dpg.group(horizontal=True):

                    def callback_set_dynamic_resolution(sender, app_data):
                        if self.dynamic_resolution:
                            self.dynamic_resolution = False
                            self.downscale = 1
                        else:
                            self.dynamic_resolution = True
                        self.need_update = True

                    dpg.add_checkbox(label='dynamic resolution', default_value=self.dynamic_resolution,
                                     callback=callback_set_dynamic_resolution)
                    dpg.add_text(f'{self.W}x{self.H}', tag='_log_resolution')

                def callback_set_image_ss(sender, app_data):
                    self.use_image_enhancer = app_data
                    self.need_update = True

                dpg.add_checkbox(label='image enhancer', default_value=self.use_image_enhancer,
                                 callback=callback_set_image_ss)

                # mode combo
                def callback_change_mode(sender, app_data):
                    self.mode = app_data
                    self.need_update = True

                dpg.add_combo(('image', 'depth'), label='mode', default_value=self.mode, callback=callback_change_mode)

                # bg_color picker
                def callback_change_bg(sender, app_data):
                    self.bg_color = torch.tensor(app_data[:3], dtype=torch.float32)  # only need RGB in [0, 1]
                    self.need_update = True

                dpg.add_color_edit((255, 255, 255), label='Background Color', width=200, tag='_color_editor',
                                   no_alpha=True, callback=callback_change_bg)

                # dt_gamma_scale slider
                def callback_set_dt_gamma_scale(sender, app_data):
                    self.dt_gamma_scale = app_data
                    self.need_update = True

                dpg.add_slider_float(
                    label='dt_gamma_scale', min_value=0, max_value=1.0, clamped=True,
                    format='%.2f', default_value=self.dt_gamma_scale, callback=callback_set_dt_gamma_scale)

                # max_steps slider
                def callback_set_max_steps(sender, app_data):
                    self.model_decoder.max_steps = app_data
                    self.need_update = True

                dpg.add_slider_int(
                    label='max steps', min_value=1, max_value=1024, clamped=True,
                    format='%d', default_value=self.model_decoder.max_steps, callback=callback_set_max_steps)

                # aabb slider
                def callback_set_aabb(sender, app_data, user_data):
                    # user_data is the dimension for aabb (xmin, ymin, zmin, xmax, ymax, zmax)
                    self.model_decoder.aabb[user_data] = app_data
                    self.need_update = True

                dpg.add_separator()
                dpg.add_text('Axis-aligned bounding box:')

                with dpg.group(horizontal=True):
                    dpg.add_slider_float(label='x', width=150, min_value=-self.model_decoder.bound, max_value=0, format='%.2f',
                                         default_value=-self.model_decoder.bound, callback=callback_set_aabb, user_data=0)
                    dpg.add_slider_float(label='', width=150, min_value=0, max_value=self.model_decoder.bound, format='%.2f',
                                         default_value=self.model_decoder.bound, callback=callback_set_aabb, user_data=3)

                with dpg.group(horizontal=True):
                    dpg.add_slider_float(label='y', width=150, min_value=-self.model_decoder.bound, max_value=0, format='%.2f',
                                         default_value=-self.model_decoder.bound, callback=callback_set_aabb, user_data=1)
                    dpg.add_slider_float(label='', width=150, min_value=0, max_value=self.model_decoder.bound, format='%.2f',
                                         default_value=self.model_decoder.bound, callback=callback_set_aabb, user_data=4)

                with dpg.group(horizontal=True):
                    dpg.add_slider_float(label='z', width=150, min_value=-self.model_decoder.bound, max_value=0, format='%.2f',
                                         default_value=-self.model_decoder.bound, callback=callback_set_aabb, user_data=2)
                    dpg.add_slider_float(label='', width=150, min_value=0, max_value=self.model_decoder.bound, format='%.2f',
                                         default_value=self.model_decoder.bound, callback=callback_set_aabb, user_data=5)

            # debug info
            if self.debug:
                with dpg.collapsing_header(label='Debug'):
                    # pose
                    dpg.add_separator()
                    dpg.add_text('Camera Pose:')
                    dpg.add_text(self.active_cam.pose2str(), tag='_log_pose')

        def callback_camera_drag_rotate(sender, app_data):
            if not dpg.is_item_focused('_primary_window'):
                return

            dx = app_data[1]
            dy = app_data[2]

            self.active_cam.orbit(dx, dy)
            self.need_update = True

            update_camera_status()

        def callback_camera_wheel_scale(sender, app_data):

            if not dpg.is_item_focused('_primary_window'):
                return

            delta = app_data

            self.active_cam.scale(delta)
            self.need_update = True

            update_camera_status()

        def callback_camera_drag_pan(sender, app_data):

            if not dpg.is_item_focused('_primary_window'):
                return

            dx = app_data[1]
            dy = app_data[2]

            self.active_cam.pan(dx, dy)
            self.need_update = True

            update_camera_status()

        with dpg.handler_registry():
            dpg.add_mouse_drag_handler(button=dpg.mvMouseButton_Left, callback=callback_camera_drag_rotate)
            dpg.add_mouse_wheel_handler(callback=callback_camera_wheel_scale)
            dpg.add_mouse_drag_handler(button=dpg.mvMouseButton_Middle, callback=callback_camera_drag_pan)

        dpg.create_viewport(
            title='SSDNeRF GUI',
            width=self.W + 400, height=self.H + 50,
            resizable=False)

        ### global theme
        with dpg.theme() as theme_no_padding:
            with dpg.theme_component(dpg.mvAll):
                # set all padding to 0 to avoid scroll bar
                dpg.add_theme_style(dpg.mvStyleVar_WindowPadding, 0, 0, category=dpg.mvThemeCat_Core)
                dpg.add_theme_style(dpg.mvStyleVar_FramePadding, 0, 0, category=dpg.mvThemeCat_Core)
                dpg.add_theme_style(dpg.mvStyleVar_CellPadding, 0, 0, category=dpg.mvThemeCat_Core)

        dpg.bind_item_theme('_primary_window', theme_no_padding)
        dpg.setup_dearpygui()
        dpg.show_viewport()

    def render(self):
        while dpg.is_dearpygui_running():
            if self.debug:
                jobs = dpg.get_callback_queue()  # retrieves and clears queue
                dpg.run_callbacks(jobs)
            # update texture every frame
            self.test_step()
            dpg.render_dearpygui_frame()
    
    def fib_sphere_images(self, n: int, cam: OrbitCamera, root: str):
        images = []
        intrinsics = []
        poses = []
        BLENDER_TO_OPENCV_MATRIX = np.array([
            [1,  0,  0,  0],
            [0, -1,  0,  0],
            [0,  0, -1,  0],
            [0,  0,  0,  1]
        ], dtype=np.float32)
        meta = {
            "camera_angle_x": np.radians(cam.fovy),
            "camera_angle_y": np.radians(cam.fovy),
            "fl_x": cam.intrinsics[0],
            "fl_y": cam.intrinsics[1],
            "cx": cam.intrinsics[2],
            "cy": cam.intrinsics[3],
            "w": cam.W,
            "h": cam.H,
            "aabb_scale": 4,
            "frames": []
        }
        pbar = mmcv.ProgressBar(n)
        for i in range(n):
            k = i + .5
            phi = np.arccos(1 - 2 * k / n)
            theta = np.pi * (1 + np.sqrt(5)) * k
            cam.set_euler([0, np.degrees(theta), np.degrees(phi)])
            image, depth = self.model.render(
                self.model_decoder,
                self.code_buffer[None],
                self.density_bitfield[None], cam.H, cam.W,
                self.code_buffer.new_tensor(cam.intrinsics, dtype=torch.float32)[None, None],
                self.code_buffer.new_tensor(cam.pose, dtype=torch.float32)[None, None],
                cfg=dict(dt_gamma_scale=0.0, return_rgba=True))
            images.append(image[0, 0].clamp(min=0, max=1).cpu())
            intrinsics.append(cam.intrinsics)
            poses.append(cam.pose)
            meta['frames'].append({
                "file_path": "images/%04d.png" % i,
                "transform_matrix": (np.array(cam.pose) @ BLENDER_TO_OPENCV_MATRIX).tolist()
            })
            pbar.update(1)
        print(flush=True)
        os.makedirs(os.path.join(root, "images"), exist_ok=True)
        with open(os.path.join(root, "transforms.json"), "w") as fo:
            json.dump(meta, fo, indent=2)
        for i, image in enumerate(images):
            plotlib.imsave(os.path.join(root, "images", "%04d.png" % i), image.numpy())
        return images, intrinsics, poses

    @torch.no_grad()
    def export_multi_view_data(self, root):
        cam = OrbitCamera('export', 192, 192, 2.8, 45)
        self.fib_sphere_images(12, cam, os.path.join(root, "low"))
        self.fib_sphere_images(100, cam, os.path.join(root, "basic"))

    @torch.no_grad()
    def export_vdb(self, save_path):
        print(f'==> Saving volume to {save_path}')
        decoder = self.model_decoder
        code_single = self.code_buffer
        code_single = decoder.preproc(code_single[None]).squeeze(0)

        def query_func(pts):
            with torch.no_grad():
                pts = pts.to(code_single.device)[None]
                sigma = decoder.point_density_decode(
                    pts,
                    code_single[None])[0].flatten()
                out_mask = (pts.squeeze(0) < decoder.aabb[:3]).any(dim=-1) | (pts.squeeze(0) > decoder.aabb[3:]).any(dim=-1)
                sigma.masked_fill_(out_mask, 0)
            return sigma.float()

        aabb = decoder.aabb.float()
        u = extract_fields(aabb[:3], aabb[3:], self.mesh_resolution, query_func)
        with open(save_path, "wb") as fo:
            fo.write(vdb_utils.dumps(u, self.model.test_cfg['density_thresh']))
        print(f'==> Finished saving volume.')

    @torch.no_grad()
    def export_mesh(self, save_path):
        print(f'==> Saving mesh to {save_path}')
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        vertices, triangles = extract_geometry(
            self.model_decoder,
            self.code_buffer,
            resolution=self.mesh_resolution,
            threshold=self.mesh_threshold)
        mesh = trimesh.Trimesh(vertices, triangles, process=False)  # important, process=True leads to seg fault...
        code_pre = self.model_decoder.preproc(self.code_buffer[None])
        colors = self.model_decoder.point_decode(
            torch.tensor(mesh.vertices)[None].to(
                device=code_pre.device, dtype=torch.float32),
            -torch.tensor(mesh.vertex_normals)[None].to(
                device=code_pre.device, dtype=torch.float32),
            code_pre
        )[1].clamp(min=0, max=1).cpu().float().numpy() * 255
        mesh = trimesh.Trimesh(mesh.vertices, mesh.faces, vertex_colors=colors, process=False)
        mesh.export(save_path)
        print(f'==> Finished saving mesh.')
