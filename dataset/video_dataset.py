import torch
from torch.utils.data import Dataset
from skimage import io
import skimage.color
import os
import numpy as np

class VideoDataset(Dataset):
    def __init__(self, path):
        self.path = path
        self.files = []
        for filename in os.listdir(path):
            self.files.append(filename)

    def __getitem__(self, index):
        path = os.path.join(self.path, self.files[index])
        # Crop to make it 3 x 352 x 480, so that both x and y are multiples of 32
        cropped_image = io.imread(path)[4:-4]
        image = skimage.color.rgb2lab(cropped_image)
        # Normalize so that Lab values are all in the range of [-1, 1)
        # Original ranges are [0, 100), [-128, 128), [-128, 128) respectively
        lab_normalized = (image + [-50, 0, 0]) / [50, 128, 128]
        return lab_normalized.astype(np.float32).transpose((2, 0, 1))

    def __len__(self):
        return len(self.files)
