import torch
from skimage import io
import skimage.color
import numpy as np


def load_bw_image(path):
    # Crop so that both dimensions are multiples of 32
    original = io.imread(path)[4:-4].astype(np.float32)
    L_channel = (original[:, :, 0] - 128) / 128
    L_channel = np.expand_dims(L_channel, axis=[0, 1])
    return L_channel


def load_color_image(path):
    # Crop so that both dimensions are multiples of 32
    original = io.imread(path)[4:-4]
    image = skimage.color.rgb2lab(original).astype(np.float32)
    L_channel = (image[:, :, 0] - 50) / 50
    L_channel = np.expand_dims(L_channel, axis=[0, 1])
    return original, L_channel


def create_color_image_from_output(output, L_channel):
    ab_channels = output.squeeze().cpu().detach().numpy()
    Lab = np.concatenate([L_channel[0], ab_channels], axis=0)
    Lab = Lab.transpose((1, 2, 0))
    Lab = (Lab * [50, 128, 128] + [50, 0, 0])
    rgb_output = skimage.color.lab2rgb(Lab)
    return rgb_output


def main():
    pass


if __name__ == '__main__':
    main()
