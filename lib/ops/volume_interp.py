import torch
import torch.nn.functional as F
try:
    import spconv.pytorch as spconv
except:
    pass


def _encode_coords(coords, spatial_shape):
    """
    Args:
        coords (torch.Tensor): (*, 4) in [batch_idx, x_D, y_H, z_W]
        spatial_shape (tuple | list): (3, ) in [D, H, W]

    Returns:
        torch.Tensor: (*, )
    """
    assert coords.size(-1) == 4
    assert len(spatial_shape) == 3
    coords = coords.long()
    shape_bitlength = [(s - 1).bit_length() for s in spatial_shape]
    shift = (shape_bitlength[0] + shape_bitlength[1] + shape_bitlength[2],
             shape_bitlength[1] + shape_bitlength[2],
             shape_bitlength[2])
    return (coords[..., 0] << shift[0]) | (coords[..., 1] << shift[1]) | (coords[..., 2] << shift[2]) | coords[..., 3]


def _coord_to_feat_idx_search(
        query, indices, spatial_shape, key, sort_idx, check_valid=True, return_masked=True):
    query_enc = _encode_coords(query, spatial_shape)

    if check_valid:
        idx = torch.bucketize(query_enc, key).clamp(max=key.size(0) - 1)
        feat_idx = sort_idx[idx] if sort_idx is not None else idx
        valid_mask = torch.all(indices[feat_idx] == query, dim=-1)
        if return_masked:
            return feat_idx[valid_mask], valid_mask
        else:
            feat_idx.masked_fill_(~valid_mask, -1)
            return feat_idx

    else:
        idx = torch.bucketize(query_enc, key)
        feat_idx = sort_idx[idx] if sort_idx is not None else idx
        return feat_idx


def _coord_to_feat_idx_maskedsearch(query, indices, spatial_shape, batch_size, mask, key, sort_idx):
    valid_mask = torch.all((query >= 0) & (query < query.new_tensor([batch_size, *spatial_shape])), dim=-1)
    query.masked_fill_(~valid_mask.unsqueeze(-1), 0)
    valid_mask &= mask[query[..., 0], query[..., 1], query[..., 2], query[..., 3]]
    valid_feat_idx = _coord_to_feat_idx_search(query[valid_mask], indices, spatial_shape, key, sort_idx, check_valid=False)
    return valid_feat_idx, valid_mask


def _prepare_search_vars(x):
    if not hasattr(x, 'encoded_indices'):
        x.encoded_indices = _encode_coords(x.indices, x.spatial_shape)
    if not hasattr(x, 'indices_sorted'):
        x.indices_sorted = torch.all(x.encoded_indices[1:] > x.encoded_indices[:-1])
    if x.indices_sorted:
        key = x.encoded_indices
        sort_idx = None
    else:
        key, sort_idx = x.encoded_indices.sort()
    return key, sort_idx


def _prepare_mask(x):
    if not hasattr(x, 'mask'):
        indices = x.indices.long()
        x.mask = torch.zeros(
            (x.batch_size, *x.spatial_shape), dtype=torch.bool, device=indices.device)
        x.mask[indices[:, 0], indices[:, 1], indices[:, 2], indices[:, 3]] = True


def coord_to_feat_idx_search(x, query, check_valid=True, return_masked=True):
    """
    Inspired by OCNN's implementation (https://github.com/octree-nn/ocnn-pytorch).

    Args:
        x (spconv.SparseConvTensor): Spatial shape (D, H, W)
        query (torch.Tensor): (*, 4) in [batch_idx, x_D, y_H, z_W]

    Returns:
        Tuple[Tensor]:
            valid_feat_idx: Shape (num_valid, ) or (*, )
            valid_mask: Shape (*, ), optional
    """
    key, sort_idx = _prepare_search_vars(x)
    return _coord_to_feat_idx_search(
        query, x.indices, x.spatial_shape, key, sort_idx, check_valid, return_masked)


def coord_to_feat_idx_maskedsearch(x, query):
    """
    Faster but higher VRAM footprint for large matrices.

    Args:
        x (spconv.SparseConvTensor): Spatial shape (D, H, W)
        query (torch.Tensor): (*, 4) in [batch_idx, x_D, y_H, z_W]

    Returns:
        Tuple[Tensor]:
            valid_feat_idx: Shape (num_valid, )
            valid_mask: Shape (*, )
    """
    _prepare_mask(x)
    key, sort_idx = _prepare_search_vars(x)
    return _coord_to_feat_idx_maskedsearch(query, x.indices, x.spatial_shape, x.batch_size, x.mask, key, sort_idx)


class NeighborData(object):
    """
    Args:
        floor_inds_plus_one_mask (torch.Tensor): Shape (batch_size, *spatial_shape + 1)
        floor_inds_plus_one (torch.Tensor): Shape (num_floor, 3)
        enc_floor_inds_plus_one (torch.Tensor): Shape (num_floor, )
        neighbor_feat_idx (torch.Tensor): Shape (num_floor, 8)
    """
    def __init__(self, floor_inds_plus_one_mask, floor_inds_plus_one, enc_floor_inds_plus_one, neighbor_feat_idx,
                 spatial_shape_plus_one, batch_size, grid):
        self.floor_inds_plus_one_mask = floor_inds_plus_one_mask
        self.floor_inds_plus_one = floor_inds_plus_one
        self.enc_floor_inds_plus_one = enc_floor_inds_plus_one
        self.neighbor_feat_idx = neighbor_feat_idx
        self.spatial_shape_plus_one = spatial_shape_plus_one
        self.batch_size = batch_size
        self.grid = grid


def build_neighbor(x):
    floor_inds_plus_one_mask = F.max_pool3d(x.mask.half(), 2, stride=1, padding=1, ceil_mode=False).bool()
    floor_inds_plus_one = floor_inds_plus_one_mask.nonzero()

    spatial_shape_plus_one = [s + 1 for s in x.spatial_shape]
    enc_floor_inds_plus_one = _encode_coords(floor_inds_plus_one, spatial_shape_plus_one)

    grid = torch.tensor(
        [[0, 0, 0], [0, 0, 1], [0, 1, 0], [0, 1, 1],
         [1, 0, 0], [1, 0, 1], [1, 1, 0], [1, 1, 1]],
        device=x.indices.device)

    neighbor_inds = floor_inds_plus_one[:, None, 1:] + (grid - 1)  # (num_floor, 8, 3)
    neighbor_feat_idx = coord_to_feat_idx_search(
        x, torch.cat([floor_inds_plus_one[:, None, :1].expand(-1, 8, -1), neighbor_inds], dim=-1),
        check_valid=True, return_masked=False)

    return NeighborData(floor_inds_plus_one_mask, floor_inds_plus_one, enc_floor_inds_plus_one, neighbor_feat_idx,
                        spatial_shape_plus_one, x.batch_size, grid)


def spvolume_linear_interp(x, pts, batch_inds, masked=True, normalize=None, prune=True, align_corners=False, eps=1e-6):
    """
    Note that the dimension order is different from grid_sample.

    Args:
        x (spconv.SparseConvTensor): Spatial shape (D, H, W)
        pts (torch.Tensor): (num_pts, 3) in [x_D, y_H, z_W], with range [-1, 1]
        batch_inds (torch.Tensor): (num_pts, 1)
        masked (bool): Whether to mask out points in empty voxels
        normalize (bool): Whether to normalize the weights, default to masked
        prune (bool): Whether to prune out zero out_feats

    Returns:
        Tuple[torch.Tensor]:
            out_feats: Shape (num_pts or num_pts_valid, C)
            valid_pts_mask: Shape (num_pts, )
    """
    if normalize is None:
        normalize = masked
    assert not align_corners, 'align_corners=True is not supported'
    device = pts.device
    num_feats = x.features.size(0)

    spatial_shape = pts.new_tensor(x.spatial_shape)
    pt_inds = pts * (spatial_shape / 2) + (spatial_shape / 2 - 0.5)

    if masked:
        assert prune, 'prune=False is not suported when masked=True'
        pt_inds_round = pt_inds.round().long()
        _prepare_mask(x)
        # (num_pts, )
        valid_pts_mask = torch.all(
            (pt_inds_round >= 0) & (pt_inds_round < pt_inds_round.new_tensor(spatial_shape)), dim=-1)
        pt_inds_round.masked_fill_(~valid_pts_mask.unsqueeze(-1), 0)
        valid_pts_mask &= x.mask[batch_inds[:, 0], pt_inds_round[:, 0], pt_inds_round[:, 1], pt_inds_round[:, 2]]
        batch_inds = batch_inds[valid_pts_mask]
        pt_inds = pt_inds[valid_pts_mask]

    pt_inds_floor = pt_inds.floor()
    pt_inds_frac = pt_inds - pt_inds_floor

    grid = torch.tensor(
        [[0, 0, 0], [0, 0, 1], [0, 1, 0], [0, 1, 1],
         [1, 0, 0], [1, 0, 1], [1, 1, 0], [1, 1, 1]], device=pts.device)
    pt_inds_neigh = pt_inds_floor.unsqueeze(1).long() + grid  # (num_pts, 8, 3)

    # (num_valid_feat, ), (num_pts, 8)
    valid_feat_idx, valid_mask = coord_to_feat_idx_maskedsearch(
        x, torch.cat([batch_inds.unsqueeze(1).expand(-1, 8, -1), pt_inds_neigh], dim=-1))

    if not masked:
        valid_pts_mask = torch.any(valid_mask, dim=-1)  # (num_pts, )
        if prune:
            valid_mask = valid_mask[valid_pts_mask]  # (num_pts_valid, 8)
            pt_inds_frac = pt_inds_frac[valid_pts_mask]

    num_pts = valid_mask.size(0)  # num_pts_valid

    frac = (1.0 - grid) - pt_inds_frac.unsqueeze(dim=1)  # (num_pts, 8, 3) = (8, 3) - (num_pts, 1, 3)
    weight = frac.prod(dim=2).abs()[valid_mask]  # (num_pts, 8) -> (num_valid_feat, )

    pts_idx = torch.arange(num_pts, device=device)
    valid_pts_idx = pts_idx.unsqueeze(1).expand(-1, 8)[valid_mask]  # (num_valid_feat, )
    indices = torch.stack([valid_pts_idx, valid_feat_idx], dim=0)

    mat = torch.sparse_coo_tensor(indices, weight, [num_pts, num_feats], device=device)
    out_feats = torch.sparse.mm(mat, x.features)

    if normalize:
        out_feats = out_feats / (
            pts.new_tensor(eps).expand(num_pts) + torch.sparse.sum(mat, dim=1)).unsqueeze(-1)

    return out_feats, valid_pts_mask


def neighbor_spvolume_linear_interp(
        x, pts, batch_inds, masked=True, normalize=None, prune=True, align_corners=False, eps=1e-6):
    """
    Fast linear interpolation using cached neighbor indices.
    Note that the dimension order is different from grid_sample.

    Args:
        x (spconv.SparseConvTensor): Spatial shape (D, H, W)
        pts (torch.Tensor): (num_pts, 3) in [x_D, y_H, z_W], with range [-1, 1]
        batch_inds (torch.Tensor): (num_pts, 1)
        masked (bool): Whether to mask out points in empty voxels
        normalize (bool): Whether to normalize the weights, default to masked
        prune (bool): Whether to prune out zero out_feats

    Returns:
        Tuple[torch.Tensor]:
            out_feats: Shape (num_pts_valid, C)
            valid_pts_mask: Shape (num_pts, )
    """
    if normalize is None:
        normalize = masked
    assert not align_corners, 'align_corners=True is not supported'
    assert prune, 'prune=False is not supported'
    if not hasattr(x, 'neighbor'):
        x.neighbor = build_neighbor(x)

    device = pts.device
    num_feats = x.features.size(0)

    spatial_shape = pts.new_tensor(x.spatial_shape)
    pt_inds = pts * (spatial_shape / 2) + (spatial_shape / 2 - 0.5)

    if masked:
        pt_inds_round = pt_inds.round().long()
        _prepare_mask(x)
        # (num_pts, )
        valid_pts_mask = torch.all(
            (pt_inds_round >= 0) & (pt_inds_round < pt_inds_round.new_tensor(spatial_shape)), dim=-1)
        pt_inds_round.masked_fill_(~valid_pts_mask.unsqueeze(-1), 0)
        valid_pts_mask &= x.mask[batch_inds[:, 0], pt_inds_round[:, 0], pt_inds_round[:, 1], pt_inds_round[:, 2]]
        batch_inds = batch_inds[valid_pts_mask]
        pt_inds = pt_inds[valid_pts_mask]

    pt_inds_floor = pt_inds.floor()
    pt_inds_frac = pt_inds - pt_inds_floor
    pt_inds_floor_plus_one = pt_inds_floor.long() + 1

    if masked:
        # (num_pts_valid, )
        valid_inds_floor_idx = _coord_to_feat_idx_search(
            torch.cat([batch_inds, pt_inds_floor_plus_one], dim=-1), None,
            x.neighbor.spatial_shape_plus_one, x.neighbor.enc_floor_inds_plus_one, None, check_valid=False)
    else:
        # (num_pts_valid, ), (num_pts, )
        valid_inds_floor_idx, valid_pts_mask = _coord_to_feat_idx_maskedsearch(
            torch.cat([batch_inds, pt_inds_floor_plus_one], dim=-1), None,
            x.neighbor.spatial_shape_plus_one,
            x.neighbor.batch_size,
            x.neighbor.floor_inds_plus_one_mask,
            x.neighbor.enc_floor_inds_plus_one, None)
        pt_inds_frac = pt_inds_frac[valid_pts_mask]  # (num_pts_valid, 3)

    valid_feat_idx = x.neighbor.neighbor_feat_idx[valid_inds_floor_idx]  # (num_pts_valid, 8)
    valid_mask = valid_feat_idx >= 0  # (num_pts_valid, 8)
    valid_feat_idx = valid_feat_idx[valid_mask]  # (num_valid_feat, )

    num_pts_valid = valid_mask.size(0)

    frac = (1.0 - x.neighbor.grid) - pt_inds_frac.unsqueeze(dim=1)  # (num_pts_valid, 8, 3) = (8, 3) - (num_pts_valid, 1, 3)
    weight = frac.prod(dim=2).abs()[valid_mask]  # (num_pts_valid, 8) -> (num_valid_feat, )

    pts_idx = torch.arange(num_pts_valid, device=device)
    valid_pts_idx = pts_idx.unsqueeze(1).expand(-1, 8)[valid_mask]  # (num_valid_feat, )
    indices = torch.stack([valid_pts_idx, valid_feat_idx], dim=0)

    mat = torch.sparse_coo_tensor(indices, weight, [num_pts_valid, num_feats], device=device)
    out_feats = torch.sparse.mm(mat, x.features)

    if normalize:
        out_feats = out_feats / (
            pts.new_tensor(eps).expand(num_pts_valid) + torch.sparse.sum(mat, dim=1)).unsqueeze(-1)

    return out_feats, valid_pts_mask
