import torch


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
