import abc
import random

import torch.utils.data as data
from torchvision import transforms as transforms

from augmentations.misc import Identity


class AbstractDataSet(object):
    @staticmethod
    @abc.abstractmethod
    def num_class():
        raise NotImplementedError

    @staticmethod
    @abc.abstractmethod
    def code():
        raise NotImplementedError

    @staticmethod
    @abc.abstractmethod
    def statistics():
        raise NotImplementedError

    @classmethod
    def train_transform_config(cls, transform_type=None):
        config = cls._add_additional_transform(
            {
                'visda_standard_source': transforms.Compose([transforms.RandomResizedCrop(size=224, scale=(0.75, 1.33)),
                                                             transforms.RandomHorizontalFlip(),
                                                             transforms.ToTensor(),
                                                             transforms.Normalize(**cls.statistics())]),
                'visda_standard_target': transforms.Compose([transforms.Resize((224, 224)),
                                                             transforms.RandomHorizontalFlip(),
                                                             transforms.ToTensor(),
                                                             transforms.Normalize(**cls.statistics())]),
            })
        return config[transform_type] if transform_type else config

    @classmethod
    def eval_transform_config(cls, transform_type=None):
        config = cls._add_additional_transform(
            {
                'visda_standard': transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor(),
                                                      transforms.Normalize(**cls.statistics())])
            })
        return config[transform_type] if transform_type else config

    @classmethod
    def _add_additional_transform(cls, default_transform):
        new_transform_config = {}
        for transform_type, transform in default_transform.items():
            new_transform_config[transform_type] = transforms.Compose([
                cls._preprocess_transform(),
                transform,
                cls._postprocess_transform()
            ])
        return new_transform_config

    @classmethod
    def _preprocess_transform(cls):
        return Identity()

    @classmethod
    def _postprocess_transform(cls):
        return Identity()


class CombinedDataSet(data.Dataset):
    """
    source_dataset and augmented_source_dataset must be aligned
    """

    def __init__(self, source_dataset, target_dataset):
        self.source_dataset = source_dataset
        self.target_dataset = target_dataset

    def __getitem__(self, index):
        source_index = index % len(self.source_dataset)
        target_index = (index + random.randint(0, len(self.target_dataset) - 1)) % len(self.target_dataset)

        return self.source_dataset[source_index], self.target_dataset[target_index]

    def __len__(self):
        return max(len(self.source_dataset), len(self.target_dataset))
