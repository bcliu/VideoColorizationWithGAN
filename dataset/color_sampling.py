import torch
from skimage import io, color
import numpy as np

original1 = io.imread('/mnt/Research/datasets/val/qing-ep20-01146.png')[4:-4]
original2 = io.imread('/mnt/Research/datasets/val/qing-ep20-04147.png')[4:-4]

lab1 = torch.tensor(color.rgb2lab(original1), device='cuda').unsqueeze(0)
lab2 = torch.tensor(color.rgb2lab(original2), device='cuda').unsqueeze(0)

batch = torch.cat([lab1, lab2], dim=0).permute(0, 3, 1, 2)
L_channel = batch[:, [0], :, :]
ab_channels = batch[:, 1:, :, :]

print(f'Shape: {L_channel.shape}, {ab_channels.shape}')

B, C, H, W = ab_channels.shape

"""
Lab color space:
L: 0 to 100
a, b: -110 to 110
"""

n_samples_list = np.random.geometric(1 / 8, B)
ab_hint = torch.zeros_like(ab_channels)
ab_mask = torch.zeros(B, 1, H, W)

for batch_idx, n_samples in enumerate(n_samples_list):
    samples = np.random.multivariate_normal([H / 2, W / 2],
                                            np.array([[(H / 4) ** 2, 0], [0, (W / 4) ** 2]]),
                                            n_samples).astype(int)
    samples[:, 0] = np.clip(samples[:, 0], a_min=0, a_max=H - 1)
    samples[:, 1] = np.clip(samples[:, 1], a_min=0, a_max=W - 1)

    print(n_samples)
    print(samples)

    patch_sizes = np.random.uniform(low=1, high=9, size=n_samples).astype(int)

    print(patch_sizes)

    for sample_idx in range(samples.shape[0]):
        patch_size = patch_sizes[sample_idx]
        h = samples[sample_idx, 0]
        w = samples[sample_idx, 1]
        patch = ab_channels[[batch_idx], :, h:(h + patch_size), w:(w + patch_size)]
        average = patch.mean(dim=[2, 3], keepdim=True)

        ab_hint[[batch_idx], :, h:(h + patch_size), w:(w + patch_size)] = average
        ab_mask[[batch_idx], :, h:(h + patch_size), w:(w + patch_size)] = 1
