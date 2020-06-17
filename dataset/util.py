import torch
from skimage import color


def normalize_lab(lab):
    L_channel = (lab[[0]] - 50) / 100
    ab_channels = lab[1:] / 110
    return L_channel, ab_channels


def unnormalize_lab(L_channel, ab_channels):
    L_channel = L_channel * 100 + 50
    ab_channels = ab_channels * 110
    Lab = torch.cat((L_channel, ab_channels), dim=0)
    assert Lab.shape[0] == 3
    return Lab


def apply_optical_flow_to_hint(ab_hint, ab_mask, flow):
    shifted_ab_hint = torch.zeros_like(ab_hint)
    shifted_ab_mask = torch.zeros_like(ab_mask)

    hint_indices = ab_mask.nonzero()
    nonzero_y = hint_indices[:, 1]
    nonzero_x = hint_indices[:, 2]

    flow_in_CHW = torch.tensor(flow).permute((2, 0, 1))
    shifts_of_hint = flow_in_CHW[:, nonzero_y, nonzero_x]  # Values are in (delta_x, delta_y)!
    y_after_shift = torch.clamp(torch.round(shifts_of_hint[1] + nonzero_y).long(),
                                0, ab_hint.shape[1] - 1)
    x_after_shift = torch.clamp(torch.round(shifts_of_hint[0] + nonzero_x).long(),
                                0, ab_hint.shape[2] - 1)

    shifted_ab_mask[0, y_after_shift, x_after_shift] = 1
    shifted_ab_hint[:, y_after_shift, x_after_shift] = ab_hint[:, nonzero_y, nonzero_x]

    return shifted_ab_hint, shifted_ab_mask


def overlay_hint_on_image(L_channel, ab_channels, ab_mask, ab_hint, L_offset):
    replaced_ab = torch.where(torch.cat((ab_mask, ab_mask), dim=0) > 0, ab_hint, ab_channels)
    replaced_l = torch.where(ab_mask > 0, torch.clamp(L_channel + L_offset, -0.5, 0.5), L_channel)
    replaced_lab = torch.cat((replaced_l * 100 + 50, replaced_ab * 110), dim=0)
    replaced_rgb = color.lab2rgb(replaced_lab.permute((1, 2, 0)).numpy())
    return replaced_rgb
