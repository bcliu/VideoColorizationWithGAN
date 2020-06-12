def normalize_lab(lab):
    L_channel = (lab[[0]] - 50) / 100
    ab_channels = lab[1:] / 110
    return L_channel, ab_channels
