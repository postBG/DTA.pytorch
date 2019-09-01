import random

import torch.utils.data as data
from torch.utils.data import Dataset
from torchvision import transforms as transforms

from augmentations.misc import Identity
from augmentations.standard import RandomHorizontalFlip


# TODO: Refactor this
class AbstractDataSet(object):
    @staticmethod
    def num_class():
        raise NotImplementedError

    @staticmethod
    def code():
        raise NotImplementedError

    @staticmethod
    def statistics():
        raise NotImplementedError

    @classmethod
    def img_size(cls):
        raise NotImplementedError

    @classmethod
    def train_transform_config(cls, transform_type=None):
        config = cls._add_additional_transform(
            {
                'visda_standard_source': transforms.Compose([transforms.RandomResizedCrop(size=224, scale=(0.75, 1.33)),
                                                             RandomHorizontalFlip(),
                                                             transforms.ToTensor(),
                                                             transforms.Normalize(**cls.statistics())]),
                'visda_standard_target': transforms.Compose([transforms.Resize((224, 224)),
                                                             RandomHorizontalFlip(),
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
        try:
            return config[transform_type] if transform_type else config
        except KeyError:
            return config['none']

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


class DomainClassifyingDataSet(data.Dataset):
    def __init__(self, source_dataset, target_dataset):
        self.source_dataset = source_dataset
        self.target_dataset = target_dataset

    def __getitem__(self, index):
        if index < len(self.source_dataset):
            inputs, _ = self.source_dataset[index]
            return inputs, 0
        index = index - len(self.source_dataset)
        inputs, _ = self.target_dataset[index]
        return inputs, 1

    def __len__(self):
        return len(self.source_dataset) + len(self.target_dataset)


class JointDataset(Dataset):

    def __init__(self, dataset1, dataset2):
        """
        Returns image, label, and domain (0 for dataset1, 1 for dataset2)
        """
        self.dataset1 = dataset1
        self.dataset2 = dataset2
        self.len_dataset1 = len(dataset1)
        self.len_dataset2 = len(dataset2)

        self.combined_len = self.len_dataset1 + self.len_dataset2

    def __len__(self):
        return self.combined_len

    def __getitem__(self, idx):
        if idx < self.len_dataset1:
            img, label = self.dataset1[idx]
            domain = 0.0
        else:
            img, label = self.dataset2[idx - self.len_dataset1]
            domain = 1.0
        return img, float(label), domain
