import numpy as np
import torch

"""
Lab color space:
L: 0 to 100
a, b: -110 to 110
"""


def sample_color_hints(images):
    ab_channels = images[:, 1:, :, :]

    B, C, H, W = ab_channels.shape

    n_samples_list = np.random.geometric(1 / 8, B)
    ab_hint = torch.zeros_like(ab_channels)
    ab_mask = torch.zeros(B, 1, H, W)

    bounding_boxes = []

    for batch_idx, n_samples in enumerate(n_samples_list):
        samples = np.random.multivariate_normal([H / 2, W / 2],
                                                np.array([[(H / 4) ** 2, 0], [0, (W / 4) ** 2]]),
                                                n_samples).astype(int)
        samples[:, 0] = np.clip(samples[:, 0], a_min=0, a_max=H - 1)
        samples[:, 1] = np.clip(samples[:, 1], a_min=0, a_max=W - 1)

        patch_sizes = np.random.uniform(low=1, high=9, size=n_samples).astype(int)

        bounding_boxes.append([])

        for sample_idx in range(samples.shape[0]):
            patch_size = patch_sizes[sample_idx]
            h = samples[sample_idx, 0]
            w = samples[sample_idx, 1]
            patch = ab_channels[[batch_idx], :, h:(h + patch_size), w:(w + patch_size)]
            average = patch.mean(dim=[2, 3], keepdim=True)

            ab_hint[[batch_idx], :, h:(h + patch_size), w:(w + patch_size)] = average
            ab_mask[[batch_idx], :, h:(h + patch_size), w:(w + patch_size)] = 1

            bounding_boxes[-1].append((h, w, h + patch_size, w + patch_size))

    return ab_hint, ab_mask, bounding_boxes
