import os
import numpy as np
import cv2
import torch
import torch.nn.functional as F
from scipy.optimize import least_squares
from copy import deepcopy
from PIL import Image
from loftr import LoFTR, default_cfg
from .camera_utils import get_pose_from_angles_np


def init_matcher():
    # initialize feature matcher for elevation estimation
    _default_cfg = deepcopy(default_cfg)
    _default_cfg['coarse']['temp_bug_fix'] = True  # set to False when using the old ckpt
    matcher = LoFTR(config=_default_cfg)
    current_dir = os.path.dirname(os.path.abspath(__file__))
    ckpt_path = os.path.join(current_dir, '../../../checkpoints/indoor_ds_new.ckpt')
    if not os.path.exists(ckpt_path):
        os.makedirs(os.path.join(current_dir, '../../../checkpoints'), exist_ok=True)
        import gdown
        gdown.cached_download(url="https://drive.google.com/uc?id=19s3QvcCWQ6g-N1PrYlDCg-2mOJZ3kkgS",
                              path=ckpt_path)
    matcher.load_state_dict(torch.load(ckpt_path)['state_dict'])
    matcher = matcher.eval()
    return matcher


def elev_estimation(in_img, ref_imgs, ref_poses, intrinsics, intrinsics_size, matcher):
    with torch.inference_mode():
        device = ref_poses.device
        in_img = torch.from_numpy(
            cv2.resize(np.asarray(in_img)[..., :3].mean(axis=-1), (480, 480))
        ).to(device).float() / 255
        intrinsics = intrinsics * 480 / intrinsics_size
        in_imgs_mdirs = []
        ref_imgs_mdirs_world = []
        ref_imgs_mpos_world = []
        mconfs = []
        for ref_img, ref_pose in zip(ref_imgs, ref_poses):
            ref_img = torch.from_numpy(
                cv2.resize(np.asarray(ref_img)[..., :3].mean(axis=-1), (480, 480))
            ).to(device).float() / 255
            batch = {'image0': in_img[None, None], 'image1': ref_img[None, None]}
            matcher(batch)
            mkpts0 = batch['mkpts0_f']  # [N, 2] in [x, y] format
            mkpts1 = batch['mkpts1_f']
            mconf = batch['mconf']
            mdirs0 = F.normalize(F.pad((mkpts0 - intrinsics[2:]) / intrinsics[:2], [0, 1], value=1), dim=-1)
            mdirs1 = F.normalize(F.pad((mkpts1 - intrinsics[2:]) / intrinsics[:2], [0, 1], value=1), dim=-1)
            mdirs1_world = (ref_pose[:3, :3] @ mdirs1[..., None]).squeeze(-1)
            mpos1_world = ref_pose[None, :3, 3].expand_as(mdirs1_world)
            in_imgs_mdirs.append(mdirs0.cpu().numpy())
            ref_imgs_mdirs_world.append(mdirs1_world.cpu().numpy())
            ref_imgs_mpos_world.append(mpos1_world.cpu().numpy())
            mconfs.append(mconf.cpu().numpy())

        distance = torch.linalg.norm(ref_poses[:, :3, 3], dim=-1).mean().item()
        in_imgs_mdirs = np.concatenate(in_imgs_mdirs, axis=0)
        ref_imgs_mdirs_world = np.concatenate(ref_imgs_mdirs_world, axis=0)
        ref_imgs_mpos_world = np.concatenate(ref_imgs_mpos_world, axis=0)
        sqrt_mconfs = np.sqrt(np.concatenate(mconfs, axis=0))

        def residual_fun(elev):
            opt_pose = get_pose_from_angles_np(
                np.array([0], dtype=np.float64),
                elev,
                np.array([distance], dtype=np.float64)).squeeze(0)
            in_imgs_mdirs_world = (opt_pose[:3, :3] @ in_imgs_mdirs[..., None]).squeeze(-1)
            in_imgs_mpos_world = opt_pose[None, :3, 3]
            normals = np.cross(in_imgs_mdirs_world, ref_imgs_mdirs_world, axis=-1)
            normals = normals / np.linalg.norm(normals, axis=-1, keepdims=True).clip(min=1e-8)
            distances = ((in_imgs_mpos_world - ref_imgs_mpos_world)[:, None, :] @ normals[:, :, None]).reshape(-1)
            return distances * sqrt_mconfs * 100

        res_lsq = least_squares(
            residual_fun, 0.0, method='dogbox', loss='huber', bounds=[-np.pi / 2, np.pi / 2],
            # verbose=2
        )
        elev = res_lsq.x
        in_pose = get_pose_from_angles_np(
            np.array([0], dtype=np.float32),
            elev,
            np.array([distance], dtype=np.float32)).squeeze(0)

    return elev[0], torch.from_numpy(in_pose).float().to(device)


def pose5dof_estimation(in_img, ref_imgs, ref_poses, intrinsics, intrinsics_size, matcher):
    """
    Estimate elevation, distance, focal, cx, cy, designed specifically for Zero123++ v1.2
    """
    with torch.inference_mode():
        device = ref_poses.device
        in_img = torch.from_numpy(
            cv2.resize(np.asarray(in_img)[..., :3].mean(axis=-1), (480, 480))
        ).to(device).float() / 255
        intrinsics_ = intrinsics * 480 / intrinsics_size
        in_imgs_mkpts = []
        ref_imgs_mdirs_world = []
        ref_imgs_mpos_world = []
        mconfs = []
        for ref_img, ref_pose in zip(ref_imgs, ref_poses):
            ref_img = torch.from_numpy(
                cv2.resize(np.asarray(ref_img)[..., :3].mean(axis=-1), (480, 480))
            ).to(device).float() / 255
            batch = {'image0': in_img[None, None], 'image1': ref_img[None, None]}
            matcher(batch)
            mkpts0 = batch['mkpts0_f']  # [N, 2] in [x, y] format
            mkpts1 = batch['mkpts1_f']
            mconf = batch['mconf']
            mdirs1 = F.normalize(F.pad((mkpts1 - intrinsics_[2:]) / intrinsics_[:2], [0, 1], value=1), dim=-1)
            mdirs1_world = (ref_pose[:3, :3] @ mdirs1[..., None]).squeeze(-1)
            mpos1_world = ref_pose[None, :3, 3].expand_as(mdirs1_world)
            in_imgs_mkpts.append(mkpts0.cpu().numpy())
            ref_imgs_mdirs_world.append(mdirs1_world.cpu().numpy())
            ref_imgs_mpos_world.append(mpos1_world.cpu().numpy())
            mconfs.append(mconf.cpu().numpy())

        init_distance = torch.linalg.norm(ref_poses[:, :3, 3], dim=-1).mean().item()
        init_focal = intrinsics[0].item()
        in_imgs_mkpts = np.concatenate(in_imgs_mkpts, axis=0)
        ref_imgs_mdirs_world = np.concatenate(ref_imgs_mdirs_world, axis=0)
        ref_imgs_mpos_world = np.concatenate(ref_imgs_mpos_world, axis=0)
        sqrt_mconfs = np.sqrt(np.concatenate(mconfs, axis=0))

        def residual_fun(params):
            elev, distance = params[:2]
            focal, cx, cy = params[2:] * 480 / intrinsics_size
            in_imgs_mdirs = np.pad(
                (in_imgs_mkpts - np.array([cx, cy], dtype=np.float64)) / focal, [[0, 0], [0, 1]], constant_values=1)
            in_imgs_mdirs = in_imgs_mdirs / np.linalg.norm(in_imgs_mdirs, axis=-1, keepdims=True).clip(min=1e-8)
            opt_pose = get_pose_from_angles_np(
                np.array([0], dtype=np.float64),
                np.array([elev], dtype=np.float64),
                np.array([distance], dtype=np.float64)).squeeze(0)
            in_imgs_mdirs_world = (opt_pose[:3, :3] @ in_imgs_mdirs[..., None]).squeeze(-1)
            in_imgs_mpos_world = opt_pose[None, :3, 3]
            normals = np.cross(in_imgs_mdirs_world, ref_imgs_mdirs_world, axis=-1)
            normals = normals / np.linalg.norm(normals, axis=-1, keepdims=True).clip(min=1e-8)
            distances = ((in_imgs_mpos_world - ref_imgs_mpos_world)[:, None, :] @ normals[:, :, None]).reshape(-1)
            return distances * sqrt_mconfs * 100

        res_lsq = least_squares(
            residual_fun,
            [0.0, init_distance, init_focal, intrinsics_size / 2., intrinsics_size / 2.],
            method='dogbox',
            loss='huber',
            bounds=[
                [-np.pi / 2, 1.5, init_focal / 2, intrinsics_size / 2. - 50, intrinsics_size / 2. - 50],
                [np.pi / 2, 10, init_focal * 2, intrinsics_size / 2. + 50, intrinsics_size / 2. + 50]],
            x_scale=[1, 3, 200, 10, 10])
        elev, distance, focal, cx, cy = res_lsq.x
        in_pose = get_pose_from_angles_np(
            np.array([0], dtype=np.float32),
            np.array([elev], dtype=np.float32),
            np.array([distance], dtype=np.float32)).squeeze(0)

    return torch.from_numpy(in_pose).float().to(device), elev, distance, focal, cx, cy
