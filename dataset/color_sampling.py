import numpy as np
import torch


def sample_color_hints(ab_channels):
    C, H, W = ab_channels.shape

    n_samples = np.random.geometric(1 / 32, 1)
    ab_hint = torch.zeros_like(ab_channels)
    ab_mask = torch.zeros((1, H, W), device=ab_hint.device)

    bounding_boxes = []

    samples = np.random.multivariate_normal([H / 2, W / 2],
                                            np.array([[(H / 4) ** 2, 0], [0, (W / 4) ** 2]]),
                                            n_samples).astype(int)
    samples[:, 0] = np.clip(samples[:, 0], a_min=0, a_max=H - 1)
    samples[:, 1] = np.clip(samples[:, 1], a_min=0, a_max=W - 1)

    patch_sizes = np.random.uniform(low=1, high=10, size=n_samples).astype(int)

    for sample_idx in range(samples.shape[0]):
        patch_size = patch_sizes[sample_idx]
        h = samples[sample_idx, 0]
        w = samples[sample_idx, 1]
        patch = ab_channels[:, h:(h + patch_size), w:(w + patch_size)]
        average = patch.mean(dim=[1, 2], keepdim=True)

        ab_hint[:, h:(h + patch_size), w:(w + patch_size)] = average
        ab_mask[:, h:(h + patch_size), w:(w + patch_size)] = 1

        bounding_boxes.append((h, w, patch_size))

    return ab_hint, ab_mask, bounding_boxes
