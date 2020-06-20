import os

import cv2
import numpy as np
import torch

from dataset.user_guided_dataset import UserGuidedVideoDataset
from model.zhang_model import SIGGRAPHGenerator
from test_utils import predict_user_guided

device = torch.device('cuda')
grayscale_path = 'datasets/bw-frames/all'
keyframe_hints_path = 'datasets/bw-frames/hints'
output_path = 'datasets/colorized-zhang-ab0.7.avi'
model_path = 'checkpoint/siggraph_caffemodel/latest_net_G.pth'
ab_multiplier = 0.7

keyframe_hints_list = []
for filename in os.listdir(keyframe_hints_path):
    keyframe_hints_list.append(filename)
keyframe_hints_list.sort()

video_writer = cv2.VideoWriter(output_path, 0, 29.97, (480, 352))

saved_model = torch.load(model_path, map_location=device)
model = SIGGRAPHGenerator(4, 2)
model.load_state_dict(saved_model)
model = model.to(device)

dataset = UserGuidedVideoDataset(grayscale_path, random_crop=None)

for idx, (L_channel, ab_channels, _, _, _) in enumerate(dataset):
    input_L = L_channel.unsqueeze(0)
    input_ab = torch.zeros_like(ab_channels).unsqueeze(0)
    input_mask = torch.zeros_like(input_L)

    rgb = predict_user_guided(model, device, input_L, input_ab, input_mask, ab_multiplier=ab_multiplier)
    video_writer.write(cv2.cvtColor(np.uint8(rgb * 255), cv2.COLOR_RGB2BGR))

    print(f'{idx}/{len(dataset)}\r')

cv2.destroyAllWindows()
video_writer.release()
