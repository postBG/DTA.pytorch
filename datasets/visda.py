import os

import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder

from augmentations.misc import ToRGB
from datasets.base import AbstractDataSet

VISDA_DEFAULT_ROOT = './data'
VISDA_CHANNEL_STATS = {'mean': [0.485, 0.456, 0.406], 'std': [0.229, 0.224, 0.225]}


class CustomVisdaSource(ImageFolder, AbstractDataSet):
    def __init__(self, root=VISDA_DEFAULT_ROOT, **kwargs):
        root = os.path.join(root, 'train')
        super().__init__(root, **kwargs)

    @staticmethod
    def num_class():
        return 12

    @staticmethod
    def code():
        return "visda_source"

    @staticmethod
    def statistics():
        return VISDA_CHANNEL_STATS


class CustomVisdaTarget(ImageFolder, AbstractDataSet):
    def __init__(self, root=VISDA_DEFAULT_ROOT, **kwargs):
        root = os.path.join(root, 'validation')
        super().__init__(root, **kwargs)

    @staticmethod
    def num_class():
        return 12

    @staticmethod
    def code():
        return "visda_target"

    @staticmethod
    def statistics():
        return VISDA_CHANNEL_STATS

    @classmethod
    def _preprocess_transform(cls):
        return transforms.Compose([ToRGB()])
