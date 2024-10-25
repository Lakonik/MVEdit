import numpy as np
import torch
import torch.nn.functional as F
import mcubes

from skimage import morphology
from packaging import version as pver


def custom_meshgrid(*args):
    # ref: https://pytorch.org/docs/stable/generated/torch.meshgrid.html?highlight=meshgrid#torch.meshgrid
    if pver.parse(torch.__version__) < pver.parse('1.10'):
        return torch.meshgrid(*args)
    else:
        return torch.meshgrid(*args, indexing='ij')


def get_ray_directions(h, w, intrinsics, norm=False, device=None):
    """
    Args:
        h (int)
        w (int)
        intrinsics: (*, 4), in [fx, fy, cx, cy]

    Returns:
        directions: (*, h, w, 3), the direction of the rays in camera coordinate
    """
    batch_size = intrinsics.shape[:-1]
    x = torch.linspace(0.5, w - 0.5, w, device=device)
    y = torch.linspace(0.5, h - 0.5, h, device=device)
    # (*, h, w, 2)
    directions_xy = torch.stack(
        [((x - intrinsics[..., 2:3]) / intrinsics[..., 0:1])[..., None, :].expand(*batch_size, h, w),
         ((y - intrinsics[..., 3:4]) / intrinsics[..., 1:2])[..., :, None].expand(*batch_size, h, w)], dim=-1)
    # (*, h, w, 3)
    directions = F.pad(directions_xy, [0, 1], mode='constant', value=1.0)
    if norm:
        directions = F.normalize(directions, dim=-1)
    return directions


def get_rays(directions, c2w, norm=False):
    """
    Args:
        directions: (*, h, w, 3) precomputed ray directions in camera coordinate
        c2w: (*, 3, 4) transformation matrix from camera coordinate to world coordinate
    Returns:
        rays_o: (*, h, w, 3), the origin of the rays in world coordinate
        rays_d: (*, h, w, 3), the normalized direction of the rays in world coordinate
    """
    rays_d = directions @ c2w[..., None, :3, :3].transpose(-1, -2)  # (*, h, w, 3)
    rays_o = c2w[..., None, None, :3, 3].expand(rays_d.shape)  # (*, h, w, 3)
    if norm:
        rays_d = F.normalize(rays_d, dim=-1)
    return rays_o, rays_d


def get_cam_rays(c2w, intrinsics, h, w):
    directions = get_ray_directions(
        h, w, intrinsics, norm=False, device=intrinsics.device)  # (num_scenes, num_imgs, h, w, 3)
    rays_o, rays_d = get_rays(directions, c2w, norm=True)
    return rays_o, rays_d


def extract_fields(bound_min, bound_max, resolution, query_func, S=128):
    X = torch.linspace(bound_min[0], bound_max[0], resolution).split(S)
    Y = torch.linspace(bound_min[1], bound_max[1], resolution).split(S)
    Z = torch.linspace(bound_min[2], bound_max[2], resolution).split(S)

    u = np.zeros([resolution, resolution, resolution], dtype=np.float32)
    with torch.no_grad():
        for xi, xs in enumerate(X):
            for yi, ys in enumerate(Y):
                for zi, zs in enumerate(Z):
                    xx, yy, zz = custom_meshgrid(xs, ys, zs)
                    pts = torch.cat([xx.reshape(-1, 1), yy.reshape(-1, 1), zz.reshape(-1, 1)], dim=-1)  # [S, 3]
                    val = query_func(pts).reshape(len(xs), len(ys),
                                                  len(zs)).detach().cpu().numpy()  # [S, 1] --> [x, y, z]
                    u[xi * S: xi * S + len(xs), yi * S: yi * S + len(ys), zi * S: zi * S + len(zs)] = val
    return u


def _extract_geometry(bound_min, bound_max, resolution, threshold, query_func):
    # print('threshold: {}'.format(threshold))
    u = extract_fields(bound_min, bound_max, resolution, query_func)

    # print(u.shape, u.max(), u.min(), np.percentile(u, 50))

    vertices, triangles = mcubes.marching_cubes(u, threshold)

    b_max_np = bound_max.detach().cpu().numpy()
    b_min_np = bound_min.detach().cpu().numpy()

    vertices = vertices / (resolution - 1.0) * (b_max_np - b_min_np)[None, :] + b_min_np[None, :]
    return vertices, triangles


def extract_geometry(decoder, code_single, resolution=256, threshold=10):

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
    vertices, triangles = _extract_geometry(
        aabb[:3] - 0.01, aabb[3:] + 0.01,
        resolution=resolution, threshold=threshold, query_func=query_func)
    return vertices, triangles


def depth_to_normal(depth, directions, format='opengl'):
    """
    Args:
        depth: shape (*, h, w), inverse depth defined as 1 / z
        directions: shape (*, h, w, 3), unnormalized ray directions, under OpenCV coordinate system

    Returns:
        out_normal: shape (*, h, w, 3), in range [0, 1]
    """
    out_xyz = directions / depth.unsqueeze(-1).clamp(min=1e-6)
    dx = out_xyz[..., :, 1:, :] - out_xyz[..., :, :-1, :]
    dy = out_xyz[..., 1:, :, :] - out_xyz[..., :-1, :, :]
    right = F.pad(dx, (0, 0, 0, 1, 0, 0), mode='replicate')
    up = F.pad(-dy, (0, 0, 0, 0, 1, 0), mode='replicate')
    left = F.pad(-dx, (0, 0, 1, 0, 0, 0), mode='replicate')
    down = F.pad(dy, (0, 0, 0, 0, 0, 1), mode='replicate')
    out_normal = F.normalize(
        F.normalize(torch.cross(right, up, dim=-1), dim=-1)
        + F.normalize(torch.cross(up, left, dim=-1), dim=-1)
        + F.normalize(torch.cross(left, down, dim=-1), dim=-1)
        + F.normalize(torch.cross(down, right, dim=-1), dim=-1),
        dim=-1)
    if format == 'opengl':
        out_normal[..., 1:3] = -out_normal[..., 1:3]  # to opengl coord
    elif format == 'opencv':
        out_normal = out_normal
    else:
        raise ValueError('format should be opengl or opencv')
    out_normal = out_normal / 2 + 0.5
    return out_normal


def normalize_depth(depths, alphas, far_depth=0.25, alpha_clip=0.5, eps=1e-5):
    """
    Args:
        depths (torch.Tensor): (N, H, W)
        alphas (torch.Tensor): (N, H, W, 1)
        far_depth (float)

    Returns:
        depths (torch.Tensor): (N, H, W)
    """
    depths_max = depths.flatten(1).amax(dim=1)[:, None, None]
    depths_fg = depths / alphas.clamp(min=eps).squeeze(-1)
    depths_fg_min = depths_fg.masked_fill(
        alphas.squeeze(-1) < alpha_clip, 1 / eps).flatten(1).amin(dim=1)[:, None, None]
    depths_fg = (depths_fg - depths_fg_min) / (depths_max - depths_fg_min).clamp(min=eps)
    depths_fg = depths_fg * (1 - far_depth) + far_depth
    depths = (depths_fg * alphas.squeeze(-1)).clamp(min=0, max=1)
    return depths


def fill_holes(image):
    """
    Args:
        image (np.ndarray): (H, W), grayscale image
    """
    seed = np.copy(image)
    seed[1:-1, 1:-1] = image.max()
    mask = image
    reconstructed = morphology.reconstruction(seed, mask, method='erosion')
    return reconstructed
