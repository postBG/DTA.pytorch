import math

import torch


class HideAndSeek(object):
    def __init__(self, img_size, num_patches=16, hide_prob=0.5):
        self.img_size = img_size
        self.num_patches = num_patches
        self.sqrt_num_patches = int(math.sqrt(num_patches))
        self.hide_prob = hide_prob
        self.patch_size = int(img_size / self.sqrt_num_patches)

        if self.img_size % self.patch_size != 0:
            raise ValueError("patch_size cannot divide by img_size")
        self.patch_list = self.get_random_patch_list()

    def __call__(self, zero_centered_pic_tensor: torch.Tensor):
        """
        :param zero_centered_pic_tensor: 0 centered normalized tensor
        :return: pic with hidden patches
        """
        hide_pic = zero_centered_pic_tensor.clone()
        hide_indicators = torch.rand(self.num_patches) < self.hide_prob
        for (patch_x, patch_y), indicator in zip(self.patch_list, hide_indicators):
            if indicator:
                hide_pic[:, patch_x:patch_x + self.patch_size, patch_y:patch_y + self.patch_size] = 0

        return hide_pic

    def get_random_patch_list(self):
        patch_list = [(x * self.patch_size, y * self.patch_size)
                      for y in range(self.sqrt_num_patches)
                      for x in range(self.sqrt_num_patches)]
        return patch_list
