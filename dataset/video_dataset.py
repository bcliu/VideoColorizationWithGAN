import torch
from torch.utils.data import Dataset
from skimage import io
import os
import numpy as np

class VideoDataset(Dataset):
    def __init__(self):
        self.path = '/home/vltava/ShanghaiProject/test-dataset'
        self.filelist = []
        for filename in os.listdir(os.path.join(self.path, 'bw')):
            self.filelist.append(filename)

    def __getitem__(self, index):
        path = os.path.join(self.path, 'bw', self.filelist[index])
        image = io.imread(path)[:, :, 0]  # Take just one channel is enough since the values are all the same
        image = (image.astype(np.float32) - 128) / 128.  # Normalize
        return np.expand_dims(image, axis=0)

    def __len__(self):
        return len(self.filelist)
