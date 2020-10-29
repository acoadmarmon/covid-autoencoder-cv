from torchvision import datasets, models, transforms
import torch
import data_transforms
import os

import os
from typing import Tuple

import numpy as np
from PIL import Image
from sklearn.preprocessing import StandardScaler


def compute_mean_and_std(root_dir):
  image_list = []
  for path, subdirs, files in os.walk(root_dir):
    for name in files:
        image = np.asarray(Image.open(os.path.join(path, name)).convert('L'))
        image_list.append(image.flatten())
  
  scaler = StandardScaler()
  pixel_values = np.concatenate(image_list).reshape(-1, 1)
  pixel_values = (pixel_values - np.min(pixel_values)) / (np.max(pixel_values) - np.min(pixel_values))
  scaler.fit(pixel_values)
  mean = scaler.mean_
  std = scaler.scale_
  return mean, std

def get_image_dataset(root_dir='./', split='train', resize=(224, 224), batch_size=16, shuffle=True, num_workers=0):
    
    mean, std = [0.63886562], [0.27450625]
    curr_transforms = {
    'train': data_transforms.get_train_transforms(resize, mean, std),
    'val': data_transforms.get_test_transforms(resize, mean, std)}
    img_folder = datasets.ImageFolder(os.path.join(root_dir, split), curr_transforms[split])
    print(img_folder.classes)

    return img_folder, torch.utils.data.DataLoader(img_folder, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)