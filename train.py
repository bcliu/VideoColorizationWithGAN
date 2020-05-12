import torch
from model.unet import UNet
from dataset.video_dataset import VideoDataset
from torch.utils.data import DataLoader

model = UNet()

dataset = VideoDataset()

dataloader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=1)

for i_batch, batch in enumerate(dataloader):
    print(batch.shape)

    model(batch)
