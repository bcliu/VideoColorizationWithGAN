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
        image = skimage.color.rgb2lab(io.imread(path))
        lab_normalized = (image + [-50, 0, 0]) / [50, 128, 128]
        return lab_normalized.astype(np.float32).transpose((2, 0, 1))

    def __len__(self):
        return len(self.files)
