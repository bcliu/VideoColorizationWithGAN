import os
from typing import List

import numpy as np
import skimage.color
import torch
from skimage import io
from torch.utils.data import Dataset
from torchvision import transforms

from dataset.color_sampling import sample_color_hints
from dataset.util import normalize_lab


class UserGuidedVideoDataset(Dataset):
    def __init__(self, path, augmentation: bool, files: List[str] = None):
        self.path = path

        if files is None:
            self.files = []
            for filename in os.listdir(path):
                self.files.append(filename)
        else:
            self.files = files

        transform_list = [transforms.ToPILImage()]
        if augmentation:
            transform_list += [
                transforms.RandomCrop((320, 320)),
                transforms.RandomHorizontalFlip(),
            ]
        self.augmentation_applied = augmentation
        self.augmentation = transforms.Compose(transform_list)

    def __getitem__(self, index):
        path = os.path.join(self.path, self.files[index])
        rgb = self.augmentation(io.imread(path))
        lab = skimage.color.rgb2lab(rgb).astype(np.float32)
        lab = torch.tensor(lab).permute((2, 0, 1))

        if not self.augmentation_applied:
            lab = self._crop(lab)

        L_channel, ab_channels = normalize_lab(lab)

        ab_hint, ab_mask, _ = sample_color_hints(ab_channels)
        return L_channel, ab_channels, ab_hint, ab_mask

    def __len__(self):
        return len(self.files)

    @staticmethod
    def _crop(lab_tensor):
        """Crop so that dimensions of all intermediate encoder layer outputs are even numbers"""
        _, H, W = lab_tensor.shape
        cropped_H = H // 16 * 16
        cropped_W = W // 16 * 16
        return lab_tensor[:, :cropped_H, :cropped_W]
