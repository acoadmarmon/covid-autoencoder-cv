from typing import Tuple

import numpy as np
import torchvision.transforms as transforms


def get_test_transforms(input_size,
                        pixel_mean,
                        pixel_std):

  return transforms.Compose([
      transforms.Resize(input_size),
      transforms.Grayscale(),
      transforms.ToTensor(),
      transforms.Normalize(mean=pixel_mean, std=pixel_std)
  ])

def get_no_transforms():

  return transforms.Compose([
      transforms.Grayscale(),
      transforms.ToTensor()
  ])

def get_train_transforms(input_size,
                         pixel_mean,
                         pixel_std):
  return transforms.Compose([
      transforms.Resize(input_size),
      transforms.RandomHorizontalFlip(p=.5),
      transforms.ColorJitter(brightness=.1),
      transforms.Grayscale(),
      transforms.ToTensor(),
      transforms.Normalize(mean=pixel_mean, std=pixel_std),
  ])
