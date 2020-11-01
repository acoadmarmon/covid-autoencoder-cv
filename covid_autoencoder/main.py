from __future__ import print_function, division

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
import copy

import models
import image_loader
import subprocess
import argparse
import datetime

subprocess.run('..\cuda_params.bat')
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

def train_model(model, criterion, optimizer, scheduler, num_epochs=25, model_type='autoencoder', batch_size=4, model_save_dir='./'):
    since = time.time()

    train_img_folder, train_dataset = image_loader.get_image_dataset(root_dir='..\images', split='train', batch_size=batch_size, num_workers=0)
    val_img_folder, val_dataset = image_loader.get_image_dataset(root_dir='..\images', split='val', batch_size=batch_size, num_workers=0)

    image_datasets = {'train': train_img_folder, 'val': val_img_folder}
    dataloaders = {'train': train_dataset, 'val': val_dataset}

    best_model_wts = copy.deepcopy(model.state_dict())
    best_loss = np.inf

    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            #running_corrects = 0

            # Iterate over data.
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    preds = None
                    loss = None
                    if model_type == 'autoencoder':
                        preds = outputs
                        loss = criterion(outputs, inputs)
                    else:
                        _, preds = torch.max(outputs, 1)
                        loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                #running_corrects += torch.sum(preds == labels.data)
            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            #epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print('{} Loss: {:.4f}'.format(
                phase, epoch_loss))

            # deep copy the model
            if phase == 'val' and epoch_loss < best_loss:
                best_loss = epoch_loss
                best_model_wts = copy.deepcopy(model.state_dict())

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    #print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    torch.save(model.state_dict(), model_save_dir)
    return model


def main(args):
    curr_dt = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    batch_size = args.batch_size
    model_save_dir = os.path.join(args.model_dir, args.model_name, curr_dt)

    model = None
    criterion = None
    if args.model_type == 'autoencoder':
        model = models.Autoencoder()
        criterion = nn.MSELoss()
    else:
        model = torchvision.models.resnet18(pretrained=True)
        num_features = model.fc.in_features
        model.fc = nn.Linear(num_features, 2)
        criterion = nn.CrossEntropyLoss()
    
    model = model.to(device)

    optimizer = None
    if args.optimizer == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=1e-5)
    else:
        optimizer = optim.SGD(model.parameters(), lr=args.learning_rate, momentum=.9)
    
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=args.lr_step_size, gamma=args.lr_gamma)
    model = train_model(model, criterion, optimizer, exp_lr_scheduler, num_epochs=args.epochs, model_type=args.model_type, batch_size=batch_size, model_save_dir=model_save_dir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_dir',type=str,default='../trained-models/',help='The directory where the model will be stored.')
    parser.add_argument('--model_name',type=str,default='covid-autoencoder')
    parser.add_argument('--dropout',type=float,default=.2,help='Weight decay for convolutions.')
    parser.add_argument('--learning_rate',type=float,default=0.001,help='Initial learning rate.')
    parser.add_argument('--lr_step_size',type=int,default=7)
    parser.add_argument('--lr_gamma', type=float,default=.1)
    parser.add_argument('--epochs',type=int,default=25)
    parser.add_argument('--optimizer', type=str, default='adam')
    parser.add_argument('--batch-size',type=int,default=16)
    parser.add_argument('--model-type',type=str,default='autoencoder')

    args = parser.parse_args()

    main(args)
