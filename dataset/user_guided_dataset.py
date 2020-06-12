import os

import numpy as np
import skimage.color
from skimage import io
from torch.utils.data import Dataset
from torchvision import transforms
import torch
from dataset.util import normalize_lab

from dataset.color_sampling import sample_color_hints


class UserGuidedVideoDataset(Dataset):
    def __init__(self, path, augmentation: bool):
        self.path = path

        self.files = []
        for filename in os.listdir(path):
            self.files.append(filename)

        transform_list = [transforms.ToPILImage()]
        if augmentation:
            transform_list += [
                transforms.RandomCrop((320, 320)),
                transforms.RandomHorizontalFlip(),
            ]
        self.augmentation = transforms.Compose(transform_list)

    def __getitem__(self, index):
        path = os.path.join(self.path, self.files[index])
        rgb = self.augmentation(io.imread(path))
        lab = skimage.color.rgb2lab(rgb).astype(np.float32)
        lab = torch.tensor(lab).permute((2, 0, 1))
        L_channel, ab_channels = normalize_lab(lab)

        ab_hint, ab_mask, _ = sample_color_hints(ab_channels)
        return L_channel, ab_channels, ab_hint, ab_mask

    def __len__(self):
        return len(self.files)
