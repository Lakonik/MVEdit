import torch
import torch.nn.functional as F


def edge_dilation(img, mask, radius=3, iters=7):
    """
    Args:
        img (torch.Tensor): (n, c, h, w)
        mask (torch.Tensor): (n, 1, h, w)
        radius (float): Radius of dilation.

    Returns:
        torch.Tensor: Dilated image.
    """
    if radius == 0 or iters == 0:
        return img

    n, c, h, w = img.size()
    int_radius = round(radius)
    kernel_size = int(int_radius * 2 + 1)
    distance1d_sq = torch.linspace(-int_radius, int_radius, kernel_size, dtype=img.dtype, device=img.device).square()
    kernel_distance = (distance1d_sq.reshape(1, -1) + distance1d_sq.reshape(-1, 1)).sqrt()
    kernel_neg_distance = kernel_distance.max() - kernel_distance + 1

    for _ in range(iters):

        mask_out = F.max_pool2d(mask, kernel_size, stride=1, padding=int_radius)
        do_fill_mask = ((mask_out - mask) > 0.5).squeeze(1)
        # (num_fill, 3) in [ind_n, ind_h, ind_w]
        do_fill = do_fill_mask.nonzero()

        # unfold the image and mask
        mask_unfold = F.unfold(mask, kernel_size, padding=int_radius).reshape(
            n, kernel_size * kernel_size, h, w).permute(0, 2, 3, 1)

        fill_ind = (mask_unfold[do_fill_mask] * kernel_neg_distance.flatten()).argmax(dim=-1)
        do_fill_h = do_fill[:, 1] + fill_ind // kernel_size - int_radius
        do_fill_w = do_fill[:, 2] + fill_ind % kernel_size - int_radius

        img_out = img.clone()
        img_out[do_fill[:, 0], :, do_fill[:, 1], do_fill[:, 2]] = img[
            do_fill[:, 0], :, do_fill_h, do_fill_w]

        img = img_out
        mask = mask_out

    return img
