from typing import Tuple

import numpy as np
import torchvision.transforms as transforms


def get_test_transforms(input_size,
                        pixel_mean,
                        pixel_std):

  return transforms.Compose([
      transforms.Resize(input_size),
      transforms.ToTensor(),
      transforms.Normalize(mean=pixel_mean, std=pixel_std)
  ])


def get_train_transforms(input_size,
                         pixel_mean,
                         pixel_std):
  return transforms.Compose([
      transforms.Resize(input_size),
      transforms.RandomHorizontalFlip(p=0.5),
      transforms.ColorJitter(brightness=.1),
      transforms.ToTensor(),
      transforms.Normalize(mean=pixel_mean, std=pixel_std),
  ])
