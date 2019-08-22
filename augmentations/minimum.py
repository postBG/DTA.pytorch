import numpy as np
from PIL import Image
from torchvision.transforms import RandomApply


def gaussian_noise(image: Image, mean, std):
    image_as_numpy_array = np.array(image).astype(np.float32)
    noise = np.random.normal(mean, std, image_as_numpy_array.shape)
    image_as_numpy_array += noise.astype(np.float32)
    image_as_numpy_array = np.clip(image_as_numpy_array, 0, 255)
    return Image.fromarray(image_as_numpy_array.astype(np.uint8))


class GaussianNoise(object):
    def __init__(self, mean=0, std=25.5):
        self.mean = mean
        self.std = std

    def __call__(self, pic):
        """
        Args:
            pic (Tensor or numpy.ndarray): Image to be converted to PIL Image.

        Returns:
            PIL Image: Image converted to PIL Image.

        """
        return gaussian_noise(pic, mean=self.mean, std=self.std)


class RandomGaussianNoise(object):
    def __init__(self, mean=0, std=25.5, apply_prob=0.5):
        self.transform = RandomApply([GaussianNoise(mean=mean, std=std)], p=apply_prob)

    def __call__(self, pic):
        """
        Args:
            pic (Tensor or numpy.ndarray): Image to be converted to PIL Image.

        Returns:
            PIL Image: Image converted to PIL Image.

        """
        return self.transform(pic)
