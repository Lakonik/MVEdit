import math
import numpy as np
import torch
import torch.nn.functional as F


def look_at(center, target, up):
    f = F.normalize(target - center, dim=-1)
    s = F.normalize(torch.cross(f, up, dim=-1), dim=-1)
    u = F.normalize(torch.cross(s, f, dim=-1), dim=-1)
    m = torch.stack([s, -u, f], dim=-1)
    return m


def look_at_np(center, target, up):
    f = target - center
    f = f / np.linalg.norm(f, axis=-1, keepdims=True).clip(min=1e-8)
    s = np.cross(f, up, axis=-1)
    s = s / np.linalg.norm(s, axis=-1, keepdims=True).clip(min=1e-8)
    u = np.cross(s, f, axis=-1)
    u = u / np.linalg.norm(u, axis=-1, keepdims=True).clip(min=1e-8)
    m = np.stack([s, -u, f], axis=-1)
    return m


def surround_views(initial_pose, angle_amp=1.0, num_frames=60):
    rad = torch.from_numpy(
        np.linspace(0, 2 * np.pi, num=num_frames, endpoint=False)).to(initial_pose)

    initial_pos = initial_pose[:3, -1]
    initial_pos_dist = torch.linalg.norm(initial_pos)
    initial_pos_norm = initial_pos / initial_pos_dist
    initial_angle = torch.asin(initial_pos_norm[-1])

    angles = initial_angle * (rad.sin() * angle_amp + 1)
    pos_xy = F.normalize(initial_pos_norm[:2], dim=0) @ torch.stack(
        [rad.cos(), -rad.sin(),
         rad.sin(), rad.cos()], dim=-1).reshape(-1, 2, 2)
    pos = torch.cat(
        [pos_xy * angles.cos().unsqueeze(-1), angles.sin().unsqueeze(-1)],
        dim=-1) * initial_pos_dist
    rot = look_at(pos, torch.zeros_like(pos), pos.new_tensor([0, 0, 1]).expand(pos.size()))
    poses = torch.cat(
        [torch.cat([rot, pos.unsqueeze(-1)], dim=-1),
         rot.new_tensor([0, 0, 0, 1]).expand(num_frames, 1, -1)], dim=-2)

    return poses


def get_pose_from_angles(azi, elev, distance):
    """
    Get pose from azimuth, elevation and distance

    Args:
        azi (torch.Tensor): (B, )
        elev (torch.Tensor): (B, )
        distance (float)

    Returns:
        torch.Tensor: (B, 4, 4)
    """
    pos_xy = torch.stack([azi.cos(), azi.sin()], dim=-1)
    pos = torch.cat(
        [pos_xy * elev.cos().unsqueeze(-1), elev.sin().unsqueeze(-1)], dim=-1) * distance
    rot = look_at(pos, torch.zeros_like(pos), pos.new_tensor([0, 0, 1]).expand(pos.size()))
    poses = torch.cat(
        [torch.cat([rot, pos.unsqueeze(-1)], dim=-1),
         rot.new_tensor([0, 0, 0, 1]).expand(azi.size(0), 1, -1)], dim=-2)
    return poses


def get_pose_from_angles_np(azi, elev, distance):
    pos_xy = np.stack([np.cos(azi), np.sin(azi)], axis=-1)
    pos = np.concatenate(
        [pos_xy * np.cos(elev)[:, None], np.sin(elev)[:, None]], axis=-1) * distance
    rot = look_at_np(pos, np.zeros_like(pos), np.array([0, 0, 1])[None].repeat(pos.shape[0], axis=0))
    poses = np.concatenate(
        [np.concatenate([rot, pos[:, :, None]], axis=-1),
         np.array([0, 0, 0, 1])[None].repeat(azi.shape[0], axis=0)[:, None]], axis=-2)
    return poses


def surround_views_v2(
        input_elev_deg, num_frames, max_elev_deg=30, min_elev_deg=-10, elev_cycles=3, radius=2.4,
        dtype=torch.float32, device=None):
    assert min_elev_deg <= input_elev_deg <= max_elev_deg
    elev_rescaled = (input_elev_deg - min_elev_deg) * 2 / max(max_elev_deg - min_elev_deg, 1e-6) - 1
    input_phase = math.asin(elev_rescaled)
    azi = torch.from_numpy(
        np.linspace(0, 2 * np.pi, num=num_frames, endpoint=False)).to(dtype=dtype, device=device)
    elev_deg = (torch.sin(azi * elev_cycles + input_phase) + 1) / 2 * (max_elev_deg - min_elev_deg) + min_elev_deg
    elev = elev_deg * (np.pi / 180)
    poses = get_pose_from_angles(azi, elev, radius)
    return poses, azi * (180 / np.pi), elev_deg


def random_surround_views(
        camera_distance, num_cameras, min_angle=0.1, max_angle=0.4,
        use_linspace=False, begin_rad=0, uniform=True):
    if use_linspace:
        rad = torch.from_numpy(
            np.linspace(0 + np.pi / num_cameras, 2 * np.pi - np.pi / num_cameras, num=num_cameras, dtype=np.float32))
    else:
        rad = torch.rand(num_cameras) * (2 * np.pi)
    rad += begin_rad - rad[0]
    if uniform:
        angles = torch.asin(torch.rand(num_cameras) * (math.sin(max_angle) - math.sin(min_angle)) + math.sin(min_angle))
    else:
        angles = torch.rand(num_cameras) * (max_angle - min_angle) + min_angle
    pos_xy = torch.stack([rad.cos(), rad.sin()], dim=-1)
    pos = torch.cat([pos_xy * angles.cos().unsqueeze(-1), angles.sin().unsqueeze(-1)], dim=-1) * camera_distance
    rot = look_at(pos, torch.zeros_like(pos), pos.new_tensor([0, 0, 1]).expand(pos.size()))
    poses = torch.cat(
        [torch.cat([rot, pos.unsqueeze(-1)], dim=-1),
         rot.new_tensor([0, 0, 0, 1]).expand(num_cameras, 1, -1)], dim=-2)
    return poses


def random_surround_views_v2(
        num_cameras, min_angle_deg, max_angle_deg, min_distance, max_distance, center_std, use_linspace=False, begin_rad=0):
    min_angle = min_angle_deg * (np.pi / 180)
    max_angle = max_angle_deg * (np.pi / 180)
    camera_distance = torch.rand(num_cameras) * (max_distance - min_distance) + min_distance
    centers = torch.randn(num_cameras, 3) * center_std
    if use_linspace:
        rad = torch.from_numpy(
            np.linspace(0 + np.pi / num_cameras, 2 * np.pi - np.pi / num_cameras, num=num_cameras, dtype=np.float32))
    else:
        rad = torch.rand(num_cameras) * (2 * np.pi)
    rad += begin_rad - rad[0]
    angles = torch.rand(num_cameras) * (max_angle - min_angle) + min_angle
    pos_xy = torch.stack([rad.cos(), rad.sin()], dim=-1)
    pos = torch.cat([pos_xy * angles.cos().unsqueeze(-1), angles.sin().unsqueeze(-1)], dim=-1) * camera_distance[:, None]
    rot = look_at(pos, centers, pos.new_tensor([0, 0, 1]).expand(pos.size()))
    poses = torch.cat(
        [torch.cat([rot, pos.unsqueeze(-1)], dim=-1),
         rot.new_tensor([0, 0, 0, 1]).expand(num_cameras, 1, -1)], dim=-2)
    return poses


def sample_within_circle(num_samples, spread=0.5, device=None):
    r = torch.sqrt(torch.rand(num_samples, device=device) * spread)
    theta = torch.rand(num_samples, device=device) * 2 * math.pi
    x = r * torch.cos(theta)
    y = r * torch.sin(theta)
    return torch.stack([x, y], dim=-1)


def light_sampling(camera_poses, elev_range=[10, 90], centered_light_views=None):
    """
    Args:
        camera_poses (torch.Tensor): (N, 3, 4), camera to world transformation matrix
        elev_range (list): [min, max] in degrees

    Returns:
        Tuple[torch.Tensor, torch.Tensor]:
            world_light_dir: (N, 3), light direction in world space
            cam_light_dir: (N, 3), light direction in camera space
    """
    camera_pos = F.normalize(camera_poses[:, :3, 3], dim=-1)
    cam_space_xy = sample_within_circle(camera_pos.size(0), device=camera_pos.device)
    # get camera elevation
    camera_elev = torch.asin(camera_pos[:, 2])
    delta_elev_max = (elev_range[1] * math.pi / 180 - camera_elev).clamp(min=-math.pi / 2, max=math.pi / 2)
    delta_elev_min = (elev_range[0] * math.pi / 180 - camera_elev).clamp(min=-math.pi / 2, max=math.pi / 2)
    # opencv camera convention
    cam_space_y_min = -torch.sin(delta_elev_max)
    cam_space_y_max = -torch.sin(delta_elev_min)
    mul = torch.sqrt(1 - cam_space_xy[:, 0] * cam_space_xy[:, 0])
    cam_space_y_max *= mul
    cam_space_y_min *= mul
    cam_space_xy[:, 1] = cam_space_xy[:, 1] * ((cam_space_y_max - cam_space_y_min) / 2) \
        + (cam_space_y_max + cam_space_y_min) / 2
    cam_space_z = -torch.sqrt(1 - (cam_space_xy[:, None, :] @ cam_space_xy[:, :, None]).squeeze(-1))
    cam_light_dir = torch.cat([cam_space_xy, cam_space_z], dim=-1)
    if centered_light_views is not None:
        cam_light_dir[centered_light_views] = cam_light_dir.new_tensor([0, 0, -1])
    world_light_dir = (camera_poses[:, :3, :3] @ cam_light_dir[:, :, None]).squeeze(-1)
    return world_light_dir, cam_light_dir


def view_prompts(camera_poses, front_azi, camera_azi=None):
    if camera_poses is not None:
        camera_azi = torch.atan2(camera_poses[:, 1, 3], camera_poses[:, 0, 3])
    delta_azi = (camera_azi - front_azi) % (2 * math.pi)
    out_prompts = []
    for delta_azi_single in delta_azi:
        if delta_azi_single < math.pi / 6:  # 30 degrees
            out_prompts.append('')
        elif delta_azi_single < 2 * math.pi / 3:  # 120 degrees
            out_prompts.append('side view')
        elif delta_azi_single <= 4 * math.pi / 3:  # 240 degrees
            out_prompts.append('view from behind')
        elif delta_azi_single <= 11 * math.pi / 6:  # 330 degrees
            out_prompts.append('side view')
        else:
            out_prompts.append('')
    return out_prompts
