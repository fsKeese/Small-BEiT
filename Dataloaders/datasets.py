import os
import numpy as np
import random
import math

## PyTorch
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
import torch.optim as optim

## Torchvision
import torchvision
from torchvision.datasets import CIFAR10
from torchvision.datasets import STL10
from torchvision import transforms

## PyTorch Lightning
import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint



def get_CIFAR10(DATASET_PATH):
    ## PREPARE CIFAR10

    test_transform = transforms.Compose([transforms.ToTensor(),
                                         transforms.Normalize([0.49139968, 0.48215841, 0.44653091],
                                                              [0.24703223, 0.24348513, 0.26158784])
                                         ]) #Normalizing using dataset MEAN and STD
    # For training, we add some augmentation. Networks are too powerful and would overfit.
    train_transform = transforms.Compose([transforms.RandomHorizontalFlip(),
                                          transforms.RandomResizedCrop((32, 32), scale=(0.8, 1.0), ratio=(0.9, 1.1)),
                                          transforms.ToTensor(),
                                          transforms.Normalize([0.49139968, 0.48215841, 0.44653091],
                                                               [0.24703223, 0.24348513, 0.26158784])
                                          ])
    # Loading the training dataset. We need to split it into a training and validation part
    # need to do a little trick because the validation set should not use the augmentation.
    train_dataset = CIFAR10(root=DATASET_PATH, train=True, transform=train_transform, download=True)
    val_dataset = CIFAR10(root=DATASET_PATH, train=True, transform=test_transform, download=True)
    pl.seed_everything(42)
    train_set, val_set = torch.utils.data.random_split(train_dataset, [45000, 5000])

    # Loading the test set
    test_set = CIFAR10(root=DATASET_PATH, train=False, transform=test_transform, download=True)

    # Define a set of data loaders.
    train_loader = data.DataLoader(train_set, batch_size=128, shuffle=True, drop_last=True, pin_memory=True,
                                   num_workers=4)
    val_loader = data.DataLoader(val_set, batch_size=128, shuffle=False, drop_last=False, num_workers=4)
    test_loader = data.DataLoader(test_set, batch_size=128, shuffle=False, drop_last=False, num_workers=4)

    return train_loader, test_loader, val_loader

def get_STL10_unlabeled(DATASET_PATH):
    ##PREPARE STL10

    STL_train_set = STL10(root=DATASET_PATH, split="unlabeled",
                          download=True)  # Shape : Batch, Channel, w, h. Bei CIFAR: Batch, w, h , channel
    STL_train_set.data = STL_train_set.data.transpose(0, 2, 3, 1)

    #Calculating MEAN and STD of dataset, using random 10.000 samples, not whole dataset. Whole dataset might lead to shutdown of kernel.
    DATA_MEANS = (STL_train_set.data[random.sample(range(1, 100000), 10000)] / 255.0).mean(axis=(0, 1, 2))
    DATA_STD = (STL_train_set.data[random.sample(range(1, 100000), 10000)] / 255.0).std(axis=(0, 1, 2))

    STL_test_transform = transforms.Compose([transforms.Resize((32, 32)),
                                             transforms.ToTensor(),
                                             transforms.Normalize(DATA_MEANS, DATA_STD)])
    STL_train_transform = transforms.Compose([  # transforms.Resize((32,32)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomResizedCrop((32, 32), scale=(0.8, 1.0), ratio=(0.9, 1.1)),
        transforms.ToTensor(),
        transforms.Normalize(DATA_MEANS, DATA_STD)])

    STL_train_dataset = STL10(root=DATASET_PATH, split="unlabeled", transform=STL_train_transform, download=True)
    STL_val_dataset = STL10(root=DATASET_PATH, split="unlabeled", transform=STL_test_transform, download=True)

    pl.seed_everything(42)
    STL_train_set, STL_val_set = torch.utils.data.random_split(STL_train_dataset, [80000, 20000])

    STL_test_set = STL10(root=DATASET_PATH, split="test", transform=STL_test_transform, download=True)


    STL_train_loader = data.DataLoader(STL_train_set, batch_size=128, shuffle=True, drop_last=True, pin_memory=True,
                                       num_workers=4)
    STL_val_loader = data.DataLoader(STL_val_set, batch_size=128, shuffle=False, drop_last=False, num_workers=4)
    STL_test_loader = data.DataLoader(STL_test_set, batch_size=128, shuffle=False, drop_last=False, num_workers=4)

    return STL_train_loader, STL_test_loader, STL_val_loader

def get_STL10_labeled(DATASET_PATH):
    ##PREPARE STL10 LABELED

    STL_train_set = STL10(root=DATASET_PATH, split="unlabeled",
                          download=True)  # Shape : Batch, Channel, w, h. Bei CIFAR: Batch, w, h , channel
    STL_train_set.data = STL_train_set.data.transpose(0, 2, 3, 1)

    DATA_MEANS = (STL_train_set.data[random.sample(range(1, 100000), 10000)] / 255.0).mean(axis=(0, 1, 2))
    DATA_STD = (STL_train_set.data[random.sample(range(1, 100000), 10000)] / 255.0).std(axis=(0, 1, 2))

    STL_test_transform = transforms.Compose([transforms.Resize((32, 32)),
                                             transforms.ToTensor(),
                                             transforms.Normalize(DATA_MEANS, DATA_STD)])
    STL_train_transform = transforms.Compose([transforms.RandomHorizontalFlip(),
        transforms.RandomResizedCrop((32, 32), scale=(0.8, 1.0), ratio=(0.9, 1.1)),
        transforms.ToTensor(),
        transforms.Normalize(DATA_MEANS, DATA_STD)])

    STL_train_dataset_labeled = STL10(root=DATASET_PATH, split="train", transform=STL_train_transform, download=True)

    pl.seed_everything(42)
    STL_train_set, STL_val_set = torch.utils.data.random_split(STL_train_dataset_labeled, [4000, 1000])

    STL_test_set = STL10(root=DATASET_PATH, split="test", transform=STL_test_transform, download=True)

    STL_train_loader_labeled = data.DataLoader(STL_train_set, batch_size=128, shuffle=True, drop_last=True,
                                               pin_memory=True, num_workers=4)
    STL_val_loader_labeled = data.DataLoader(STL_val_set, batch_size=128, shuffle=False, drop_last=False, num_workers=4)
    STL_test_loader_labeled = data.DataLoader(STL_test_set, batch_size=128, shuffle=False, drop_last=False,
                                              num_workers=4)
    return STL_train_loader_labeled, STL_test_loader_labeled, STL_val_loader_labeled

