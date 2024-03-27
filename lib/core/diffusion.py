import torch


def get_noise_scales(alphas_bar, t, num_timesteps, dtype=torch.float32):
    alphas_bar = t.new_tensor(alphas_bar, dtype=torch.float32)
    if t.is_floating_point():
        assert ((t >= 0) & (t <= num_timesteps - 1)).all()
        int_t = t.long()
        frac_t = t - int_t
        alphas_bar_0 = alphas_bar[int_t]
        alphas_bar_1 = alphas_bar[(int_t + 1).clamp(max=num_timesteps - 1)]
        ve_sigma_0 = torch.sqrt((1 - alphas_bar_0) / alphas_bar_0)
        ve_sigma_1 = torch.sqrt((1 - alphas_bar_1) / alphas_bar_1)
        ve_sigma = ve_sigma_0 * (1 - frac_t) + ve_sigma_1 * frac_t
        sqrt_alpha_bar_t = torch.sqrt(1 / (1 + ve_sigma ** 2))
        sqrt_one_minus_alpha_bar_t = torch.sqrt(ve_sigma ** 2 / (1 + ve_sigma ** 2))
    else:
        alpha_bar_t = alphas_bar[t]
        sqrt_alpha_bar_t = torch.sqrt(alpha_bar_t)
        sqrt_one_minus_alpha_bar_t = torch.sqrt(1 - alpha_bar_t)
    return sqrt_alpha_bar_t.to(dtype), sqrt_one_minus_alpha_bar_t.to(dtype)
