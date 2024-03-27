# modified from torch-ngp

import copy
import numpy as np
import torch
import torch.nn.functional as F
import dearpygui.dearpygui as dpg
from scipy.spatial.transform import Rotation as R

from mmgen.models.builder import build_module
from mmgen.apis import set_random_seed  # isort:skip  # noqa
from lib.datasets.shapenet_srn import load_pose, load_intrinsics


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


class MeshGUI:

    default_cam_fovy = 52.0
    default_cam_radius = 2.6
    default_cam_euler = [0.0, 23.0, -47.4]

    def __init__(self, mesh, renderer, W=512, H=512, debug=True):
        self.W = W
        self.H = H
        self.default_cam = OrbitCamera(
            'default', W, H, r=self.default_cam_radius, fovy=self.default_cam_fovy, euler=self.default_cam_euler)
        self.active_cam = self.default_cam
        self.debug = debug
        self.bg_color = torch.ones(3, dtype=torch.float32)  # default white bg
        self.step = 0  # training step

        self.mesh = mesh
        self.renderer = renderer

        self.video_sec = 4
        self.video_fps = 30
        self.video_res = 256

        self.render_buffer = np.zeros((self.H, self.W, 3), dtype=np.float32)
        self.need_update = True  # camera moved, should reset accumulation
        self.mode = 'image'  # choose from ['image', 'depth']

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
        elif self.mode == 'depth':
            return np.expand_dims(outputs['depth'], -1).repeat(3, -1)
        elif self.mode == 'alpha':
            return np.expand_dims(outputs['alpha'], -1).repeat(3, -1)
        elif self.mode == 'normal':
            return outputs['normal']
        else:
            raise ValueError(f'Unknown mode {self.mode}')

    def test_gui(self, pose, intrinsics, W, H):
        with torch.no_grad():
            if self.use_image_enhancer and self.mode == 'image':
                rH, rW = H // 2, W // 2
                intrinsics = intrinsics / 2
            else:
                rH, rW = H, W
            results = self.renderer(
                [self.mesh],
                torch.tensor(pose, dtype=torch.float32, device=self.mesh.device)[None, None],
                torch.tensor(intrinsics, dtype=torch.float32, device=self.mesh.device)[None, None],
                rH, rW)
            image = results['rgba'][..., :3] + self.bg_color.to(results['rgba']) * (1 - results['rgba'][..., 3:])
            if self.use_image_enhancer and self.mode == 'image':
                image = self.image_enhancer(image[0].half().permute(0, 3, 1, 2))
                image = F.interpolate(image, size=(H, W), mode='area').permute(0, 2, 3, 1)[None].float()
            results = dict(
                image=image[0, 0].cpu().numpy(),
                alpha=results['rgba'][0, 0, :, :, 3].cpu().numpy(),
                depth=results['depth'][0, 0].cpu().numpy(),
                normal=results['normal'][0, 0].cpu().numpy())
        return results

    def test_step(self):
        if self.need_update:

            starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            starter.record()

            outputs = self.test_gui(
                self.active_cam.pose, self.active_cam.intrinsics,
                self.W, self.H)

            ender.record()
            torch.cuda.synchronize()
            t = starter.elapsed_time(ender)

            self.render_buffer = np.ascontiguousarray(self.prepare_buffer(outputs))
            self.need_update = False

            dpg.set_value('_log_infer_time', f'{t:.4f}ms ({int(1000 / t)} FPS)')
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
                        ['default'], label='camera', width=150,
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

                def callback_set_image_ss(sender, app_data):
                    self.use_image_enhancer = app_data
                    self.need_update = True

                dpg.add_checkbox(label='image enhancer', default_value=self.use_image_enhancer,
                                 callback=callback_set_image_ss)

                # mode combo
                def callback_change_mode(sender, app_data):
                    self.mode = app_data
                    self.need_update = True

                dpg.add_combo(('image', 'depth', 'alpha', 'normal'), label='mode', default_value=self.mode, callback=callback_change_mode)

                # bg_color picker
                def callback_change_bg(sender, app_data):
                    self.bg_color = torch.tensor(app_data[:3], dtype=torch.float32)  # only need RGB in [0, 1]
                    self.need_update = True

                dpg.add_color_edit((255, 255, 255), label='Background Color', width=200, tag='_color_editor',
                                   no_alpha=True, callback=callback_change_bg)

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
