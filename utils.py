import contextlib

import numpy as np
import torch.nn as nn
import torchvision.transforms as transforms


def extract_sample_images(dataset, number_of_imgs):
    imgs, gts = [], []
    len_dataset = len(dataset)
    rand_ints = np.random.randint(0, len_dataset, number_of_imgs)
    for idx in rand_ints:
        img, _ = dataset[idx]
        imgs.append(img)
    return imgs


def tensor_to_PIL(img, mean=None, std=None, clamp=True):
    """
    Converts tensor to PIL image for visualization
    :param: Original mean used to normalize image, list
    :param: Original std used to normalize image, list
    :return: PIL Image
    """
    if not mean:
        mean = [0.485, 0.456, 0.406]
    if not std:
        std = [0.229, 0.224, 0.225]
    new_std = 1. / np.array(std)
    new_mean = -1 * np.array(mean) / np.array(std)
    unnormalize = transforms.Normalize(mean=new_mean, std=new_std)
    img = img.detach() if img.device.type == 'cpu' else img.cpu().detach()
    unnormalized_img = unnormalize(img)

    if clamp:
        unnormalized_img = unnormalized_img.clamp(0, 1)
    return transforms.ToPILImage()(unnormalized_img)


def set_requires_grad(model, requires_grad):
    for param in model.parameters():
        param.requires_grad = requires_grad


def freeze_bn_stats(model, freeze_bn=True):
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            if freeze_bn:
                m.eval()
            else:
                m.train()


@contextlib.contextmanager
def disable_tracking_bn_stats(model):
    def switch_attr(m):
        if hasattr(m, 'track_running_stats'):
            m.track_running_stats ^= True

    model.apply(switch_attr)
    yield
    model.apply(switch_attr)


