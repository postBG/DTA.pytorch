import os

import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder

from augmentations.misc import ToRGB
from datasets.utils import AbstractDataSet

VISDA_DEFAULT_ROOT = '/data3/visda'
VISDA_CHANNEL_STATS_source = {'mean': [0.485, 0.456, 0.406], 'std': [0.229, 0.224, 0.225]}
VISDA_CHANNEL_STATS_target = {'mean': [0.485, 0.456, 0.406], 'std': [0.229, 0.224, 0.225]}


class CustomVisdaSource(ImageFolder, AbstractDataSet):
    def __init__(self, root=VISDA_DEFAULT_ROOT, is_train=True, **kwargs):
        root = os.path.join(root, 'train' if is_train else 'source_validation')
        super().__init__(root, **kwargs)

    @staticmethod
    def num_class():
        return 12

    @staticmethod
    def code():
        return "visda_source"

    @staticmethod
    def statistics():
        return VISDA_CHANNEL_STATS_source

    @classmethod
    def img_size(cls):
        return 224


class CustomVisdaTarget(ImageFolder, AbstractDataSet):
    def __init__(self, root=VISDA_DEFAULT_ROOT, is_train=True, **kwargs):
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
        return VISDA_CHANNEL_STATS_target

    @classmethod
    def _preprocess_transform(cls):
        return transforms.Compose([ToRGB()])

    @classmethod
    def img_size(cls):
        return 224
