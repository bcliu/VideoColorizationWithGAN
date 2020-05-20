import os

import torch
import numpy as np
import skimage.color
from skimage import io
from torch.utils.data import Dataset
from torchvision import transforms


class VideoDataset(Dataset):
    def __init__(self, path, get_L_channel: bool = False):
        self.path = path
        self.get_L_channel = get_L_channel

        self.files = []
        for filename in os.listdir(path):
            self.files.append(filename)

        self._grayscale_transforms = transforms.Compose([
            transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        self._original_transforms = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def __getitem__(self, index):
        path = os.path.join(self.path, self.files[index])
        # Crop to make it 3 x 352 x 480, so that both x and y are multiples of 32
        original = io.imread(path)[4:-4]

        if self.get_L_channel:
            L_channel = skimage.color.rgb2lab(original).astype(np.float32)[:, :, 0]
            L_channel = torch.tensor(L_channel).unsqueeze(0)
        else:
            L_channel = None

        PIL_image = transforms.ToPILImage()(original)

        normalized_grayscale = self._grayscale_transforms(PIL_image)
        normalized_original = self._original_transforms(PIL_image)

        if self.get_L_channel:
            return normalized_grayscale, normalized_original, L_channel
        else:
            return normalized_grayscale, normalized_original

    def __len__(self):
        return len(self.files)
