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

from .Backbone import Small_VisionTransformer, Small_BEiT_Mask, Small_BEiT_Mask_und_Rotation
from.Utils import  img_to_patch, rotate_batch

device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

#Creating Lightning Module of the Backbone Architectures, for simple and automatic Training. Also Tensorboard to spectate the training is available.

#Small ViT ist the Small BEiT just with classification task
class Small_ViT_pl(pl.LightningModule):

    def __init__(self, model_kwargs, lr):
        super().__init__()
        self.save_hyperparameters()
        self.model = Small_VisionTransformer(**model_kwargs)

    def forward(self, x):
        return self.model(x)

    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), lr=self.hparams.lr)
        lr_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[100,150], gamma=0.1)
        return [optimizer], [lr_scheduler]

    def _calculate_loss(self, batch, mode="train"):
        imgs, labels = batch
        preds = self.model(imgs)
        loss = F.cross_entropy(preds, labels)
        acc = (preds.argmax(dim=-1) == labels).float().mean()

        self.log(f'{mode}_loss', loss)
        self.log(f'{mode}_acc', acc)
        return loss

    def training_step(self, batch, batch_idx):
        loss = self._calculate_loss(batch, mode="train")
        return loss

    def validation_step(self, batch, batch_idx):
        self._calculate_loss(batch, mode="val")

    def test_step(self, batch, batch_idx):
        self._calculate_loss(batch, mode="test")

class Small_BEiT_Mask_pl(pl.LightningModule):
    def __init__(self, **kwargs):
        super().__init__()
        self.model = Small_BEiT_Mask(**kwargs)

    def forward(self, x):
        return self.model(x)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=3e-4)
        lr_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[100, 150], gamma=0.1)
        return [optimizer], [lr_scheduler]

    def training_step(self, train_batch, batch_idx):
        imgs, _ = train_batch
        preds, idx = self.model(imgs)
        patches = img_to_patch(imgs, 4)
        labels = patches[idx[0], idx[1]]
        loss = F.mse_loss(preds, labels)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, val_batch, batch_idx):
        imgs, _ = val_batch
        preds, idx = self.model(imgs) # Backbone Model outputs predictions and the indexes of the Masks
        patches = img_to_patch(imgs, 4)
        labels = patches[idx[0], idx[1]] # Labels in this task are the patches which have been masked
        loss = F.mse_loss(preds, labels)
        self.log('val_loss', loss)
        return loss


class Small_BEiT_Mask_und_Rotation_pl(pl.LightningModule):
    def __init__(self, **kwargs):
        super().__init__()
        self.model = Small_BEiT_Mask_und_Rotation(**kwargs)

    def forward(self, x):
        return self.model(x)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=3e-4)
        lr_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[100, 150], gamma=0.1)
        return [optimizer], [lr_scheduler]

    def training_step(self, train_batch, batch_idx):
        imgs, _ = train_batch
        imgs, rot_labels = rotate_batch(imgs) # using rotate_batch function to rotate all pictures in the batch randomly 0°, 90°, 180° or 270°
        rot_labels = torch.tensor(rot_labels)
        rot_labels = rot_labels.to(device)
        preds_mask, idx, preds_rot = self.model(imgs) # Backbone model outputs preditcions of both tasks and indexes of masked patches
        patches = img_to_patch(imgs, 4)
        labels_mask = patches[idx[0], idx[1]]
        loss_mask = F.mse_loss(preds_mask, labels_mask)
        loss_rot = F.cross_entropy(preds_rot, rot_labels) # Cross Entropy just like classification task
        loss = loss_mask + loss_rot # No learnable weights used, just sum of both losses
        self.log('train_loss', loss)
        return loss

    def validation_step(self, val_batch, batch_idx):
        imgs, _ = val_batch
        imgs, rot_labels = rotate_batch(imgs)
        rot_labels = torch.tensor(rot_labels)
        rot_labels = rot_labels.to(device)
        preds_mask, idx, preds_rot = self.model(imgs)
        patches = img_to_patch(imgs, 4)
        labels_mask = patches[idx[0], idx[1]]
        loss_mask = F.mse_loss(preds_mask, labels_mask)
        loss_rot = F.cross_entropy(preds_rot, rot_labels)
        loss = loss_mask + loss_rot
        self.log('val_loss', loss)
        return loss
