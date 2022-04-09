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

def img_to_patch(x, patch_size, flatten_channels=True):
    """
    Inputs:
        x - torch.Tensor representing the image of shape [B, C, H, W]
        patch_size - Number of pixels per dimension of the patches (integer)
        flatten_channels - If True, the patches will be returned in a flattened format
                           as a feature vector instead of a image grid.
    """
    B, C, H, W = x.shape
    x = x.reshape(B, C, H//patch_size, patch_size, W//patch_size, patch_size)
    x = x.permute(0, 2, 4, 1, 3, 5) # [B, H', W', C, p_H, p_W]
    x = x.flatten(1,2)              # [B, H'*W', C, p_H, p_W]
    if flatten_channels:
        x = x.flatten(2,4)          # [B, H'*W', C*p_H*p_W]
    return x

def rotate_batch(batch):
    # Randomly rotate the batch 0°. 90°, 180° or 270°.
    rotation_labels = []
    for i, image in enumerate(batch):
        rot_amount = np.random.randint(0, 4)
        rotated_image = image.rot90(rot_amount,[1,2])
        batch[i] = rotated_image
        rotation_labels.append(rot_amount)
    return batch, rotation_labels

def update_state_dict_mask(PRETRAIN_CHECKPOINT, model_dict):
    # Creating Pretrained_dict from a pretrained model with mask task. Using everything except last instance of MLP Head.
    pretrained_dict = torch.load(PRETRAIN_CHECKPOINT)
    pretrained_dict = {k: v for k, v in pretrained_dict['state_dict'].items() if
                       k in model_dict and k not in ["model.mlp_head.1.weight", "model.mlp_head.1.bias"]}
    return pretrained_dict

def update_state_dict_mask_rot(PRETRAIN_CHECKPOINT, model_dict):
    # Creating Pretrained_dict from a pretrained model with mask and rot task. Again using everything except last instance of MLP Head.
    # Also taking the mean of the first instance of both parallel trained MLP Heads - mlp_head_mask and mlp_head_rot.
    pretrained_dict = torch.load(PRETRAIN_CHECKPOINT)
    average_mlp_weight = (pretrained_dict['state_dict']['model.mlp_head_mask.0.weight'] + pretrained_dict['state_dict'][
        'model.mlp_head_rot.0.weight']) / 2
    average_mlp_bias = (pretrained_dict['state_dict']['model.mlp_head_mask.0.bias'] + pretrained_dict['state_dict'][
        'model.mlp_head_mask.0.bias']) / 2

    pretrained_dict = {k: v for k, v in pretrained_dict['state_dict'].items() if
                       k in model_dict and k not in ["model.mlp_head_mask.0.weight",
                                                     "model.mlp_head_mask.0.bias",
                                                     "model.mlp_head_mask.1.weight",
                                                     "model.mlp_head_mask.1.bias",
                                                     "model.mlp_head_rot.0.weight",
                                                     "model.mlp_head_rot.0.bias",
                                                     "model.mlp_head_rot.1.weight",
                                                     "model.mlp_head_rot.1.bias"]}
    pretrained_dict["model.mlp_head.0.weight"] = average_mlp_weight
    pretrained_dict["model.mlp_head.0.bias"] = average_mlp_bias
    return pretrained_dict
