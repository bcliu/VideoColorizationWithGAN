import torch
from skimage import io
import skimage.color
import numpy as np
from torchvision import transforms
from dataset.video_dataset import colored_to_grayscale, normalize, imagenet_mean, imagenet_std


def load_grayscale(path):
    # Crop so that both dimensions are multiples of 32
    original = io.imread(path)[4:-4]
    PIL_image = transforms.ToPILImage()(original)  # Image already has 3 channels. Just normalize
    return normalize(PIL_image)


def load_grayscale_from_colored(path):
    # Crop so that both dimensions are multiples of 32
    original = io.imread(path)[4:-4]
    grayscale = skimage.color.rgb2gray(original).astype(np.float32)
    PIL_image = transforms.ToPILImage()(grayscale)
    return colored_to_grayscale(PIL_image)


def predict(model, image, device):
    with torch.no_grad():
        output = model(image.to(device))
        output[:, 0] = output[:, 0] * imagenet_std[0] + imagenet_mean[0]
        output[:, 1] = output[:, 1] * imagenet_std[1] + imagenet_mean[1]
        output[:, 2] = output[:, 2] * imagenet_std[2] + imagenet_mean[2]
    scaled = output.squeeze() * 255
    return scaled.int()


def main():
    pass


if __name__ == '__main__':
    main()
