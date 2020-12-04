
import torch
import models
import image_loader
import torchvision
import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
import pandas as pd

model = models.Autoencoder()
model.load_state_dict(torch.load('../trained-models/covid-autoencoder/20201111-144840'))

num_val_images = len([i for i in os.listdir('../images/val/non_covid/')]) + len([i for i in os.listdir('../images/val/covid/')])
num_train_images = len([i for i in os.listdir('../images/train/non_covid/')]) + len([i for i in os.listdir('../images/train/covid/')])
val_img_folder, val_dataset = image_loader.get_image_dataset(root_dir='..\images', split='val', batch_size=num_val_images, num_workers=0)
train_img_folder, train_dataset = image_loader.get_image_dataset(root_dir='..\images', split='train', batch_size=num_train_images, num_workers=0)

class_names = val_img_folder.classes
# Get a batch of training data
inputs, classes = next(iter(val_dataset))
val_vector = model.encoder(inputs).detach()
inputs, classes = next(iter(train_dataset))
train_vector = model.encoder(inputs).detach()

embeddings = pd.DataFrame(data=np.concatenate([val_vector.numpy(), train_vector.numpy()]))
embeddings['image_names'] = [i[0] for i in val_img_folder.imgs] + [i[0] for i in train_img_folder.imgs]
embeddings['image_labels'] = [i[1] for i in val_img_folder.imgs] + [i[1] for i in train_img_folder.imgs]
embeddings.to_csv('../outputs/embeddings_500.csv')