import torch
from skimage import io
from torchvision import transforms
from skimage.color import lab2rgb
from dataset.util import unnormalize_lab

from dataset.video_dataset import colored_to_grayscale, normalize, imagenet_mean, imagenet_std


def load_grayscale(path):
    # Crop so that both dimensions are multiples of 32
    original = io.imread(path)[4:-4]
    PIL_image = transforms.ToPILImage()(original)  # Image already has 3 channels. Just normalize
    return normalize(PIL_image)


def load_grayscale_from_colored(path):
    # Crop so that both dimensions are multiples of 32
    original = io.imread(path)[4:-4]
    PIL_image = transforms.ToPILImage()(original)
    return colored_to_grayscale(PIL_image)


def unnormalize(image):
    output = image.clone().detach()
    output[:, 0] = output[:, 0] * imagenet_std[0] + imagenet_mean[0]
    output[:, 1] = output[:, 1] * imagenet_std[1] + imagenet_mean[1]
    output[:, 2] = output[:, 2] * imagenet_std[2] + imagenet_mean[2]
    return output


def predict(model, image, device):
    with torch.no_grad():
        output = unnormalize(model(image.to(device)))
    clamped = torch.clamp(output, 0, 1)
    return clamped


def predict_user_guided(model, device, input_L, input_ab, input_mask, ab_multiplier):
    """

    :param model:
    :param device:
    :param input_L: Normalized L channel
    :param input_ab: Normalized ab channels
    :param input_mask:
    :return:
    """
    input_L = input_L.to(device)
    input_ab = input_ab.to(device)
    input_mask = input_mask.to(device)

    output = model(input_L, input_ab, input_mask)
    output = output.squeeze()
    input_L = input_L.squeeze(dim=0)

    Lab = unnormalize_lab(input_L, output * ab_multiplier)
    Lab = Lab.permute((1, 2, 0))
    rgb = lab2rgb(Lab.detach().cpu().numpy())
    return rgb


def main():
    pass


if __name__ == '__main__':
    main()
