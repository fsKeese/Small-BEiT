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

from Dataloaders import get_CIFAR10, get_STL10_labeled, get_STL10_unlabeled
from Modell import Small_VisionTransformer, Small_BEiT_Mask, Small_BEiT_Mask_und_Rotation # Backbone Architekuren
from Modell import Small_ViT_pl, Small_BEiT_Mask_pl, Small_BEiT_Mask_und_Rotation_pl # Pytorch Lightning Modelle für Trainer
from Modell import update_state_dict_mask, update_state_dict_mask_rot
from args import args

DATASET_PATH = "Data/"



def finetune():
    CHECKPOINT_PATH = "Checkpoints/Finetune_Checkpoints/"
    dataset = args.dataset
    pretrained_path = args.finetune
    pretrained = args.pretrained

    model_kwargs = {
            'embed_dim': 256,
            'hidden_dim': 512,
            'num_heads': 8,
            'num_layers': 6,
            'patch_size': 4,
            'num_channels': 3,
            'num_patches': 64,
            'num_classes': 10,
            'dropout': 0.2
        }

    if dataset == "CIFAR10":
        train_loader, test_loader, val_loader = get_CIFAR10(DATASET_PATH=DATASET_PATH)
        model = Small_ViT_pl(model_kwargs, lr=3e-4)
        model_dict = model.state_dict()

        #Checking if Path for a pretrained Model was set.
        if pretrained_path == "None":
            print("Initialise Model random, no pretrained Model Path has been set. If this is not correct, make sure the flag --finetune with the Path has been used and restart training with given path.")
        else:
            #Checking what kind of pretrained model was set.
            if pretrained == "mask":
                updated_state_dict = update_state_dict_mask(PRETRAIN_CHECKPOINT=pretrained_path, model_dict=model_dict)
                model_dict.update(updated_state_dict)
                model.load_state_dict(model_dict)
                print("State Dict updated with checkpoint weights.")
            else:
                updated_state_dict = update_state_dict_mask_rot(PRETRAIN_CHECKPOINT=pretrained_path, model_dict=model_dict)
                model_dict.update(updated_state_dict)
                model.load_state_dict(model_dict)
                print("State Dict updated with checkpoint weights.")

        print(
            "To spectate training, open the tensorboard --logdir Checkpoints/Finetune_Checkpoints/" + "Small_BEiT_" + dataset + "_finetuning_pretrained_with_" +pretrained)
        trainer = pl.Trainer(default_root_dir=os.path.join(CHECKPOINT_PATH, "Small_BEiT_"+dataset+"_finetuning_pretrained_with_"+pretrained),
                             gpus=1 if str(device) == "cuda:0" else 0, max_epochs=180,
                             callbacks=[ModelCheckpoint(save_weights_only=True, mode="max", monitor="val_acc"),
                                        LearningRateMonitor("epoch")])
        trainer.logger._log_graph = True  # If True, we plot the computation graph in tensorboard
        trainer.logger._default_hp_metric = None  # Optional logging argument that we don't need

        trainer.fit(model, train_loader, val_loader)

        ## After training Print Result on test set and val set
        val_result = trainer.test(model, test_dataloaders=val_loader, verbose=False)
        test_result = trainer.test(model, test_dataloaders=test_loader, verbose=False)
        result = {"test": test_result[0]["test_acc"], "val": val_result[0]["test_acc"]}
        print(result)

    elif dataset == "STL10":
        train_loader, test_loader, val_loader = get_STL10_labeled(DATASET_PATH=DATASET_PATH)
        model = Small_ViT_pl(model_kwargs, lr=3e-4)

        # Checking if Path for a pretrained Model was set.
        if pretrained_path == "None":
            print("Initialise Model random, no pretrained Model Path has been set. If this is not correct, make sure the flag --finetune with the Path has been used.")
        else:
            # Checking what kind of pretrained model was set and create state dict of pretrained Model.
            if pretrained == "mask":
                updated_state_dict = update_state_dict_mask(PRETRAIN_CHECKPOINT=pretrained_path, model_dict=model_dict)
                model_dict.update(updated_state_dict)
                model.load_state_dict(model_dict)
                print("State Dict updated with checkpoint weights.")
            else:
                updated_state_dict = update_state_dict_mask_rot(PRETRAIN_CHECKPOINT=pretrained_path, model_dict=model_dict)
                model_dict.update(updated_state_dict)
                model.load_state_dict(model_dict)
                print("State Dict updated with checkpoint weights.")


        print("To spectate training, open the tensorboard --logdir Checkpoints/Finetune_Checkpoints/" + "Small_BEiT_"+dataset+"_finetuning_pretrained_with_"+pretrained)

        trainer = pl.Trainer(default_root_dir=os.path.join(CHECKPOINT_PATH, "Small_BEiT_"+dataset+"_finetuning_pretrained_with_"+pretrained),
                             gpus=1 if str(device) == "cuda:0" else 0, max_epochs=180,
                             callbacks=[ModelCheckpoint(save_weights_only=True, mode="max", monitor="val_acc"),
                                        LearningRateMonitor("epoch")])
        trainer.logger._log_graph = True  # If True, we plot the computation graph in tensorboard
        trainer.logger._default_hp_metric = None  # Optional logging argument that we don't need

        trainer.fit(model, train_loader, val_loader)

        ## After training Print Result on test set and val set
        val_result = trainer.test(model, test_dataloaders=val_loader, verbose=False)
        test_result = trainer.test(model, test_dataloaders=test_loader, verbose=False)
        result = {"test": test_result[0]["test_acc"], "val": val_result[0]["test_acc"]}
        print(result)

def pretrain():
    DATASET_PATH = "Data/"
    pretraining_method = args.method
    dataset = args.dataset
    #setting checkpoint_path and loading data
    if pretraining_method == "pretrain_mask":
        CHECKPOINT_PATH = "Checkpoints/Pretrain_mask_Checkpoints/"
    else:
        CHECKPOINT_PATH = "Checkpoints/Pretrain_mask_rot_Checkpoints"

    if dataset == "CIFAR10":
        train_loader, test_loader, val_loader = get_CIFAR10(DATASET_PATH=DATASET_PATH)
    else:
        train_loader, test_loader, val_loader = get_STL10_unlabeled(DATASET_PATH=DATASET_PATH)

    #initialize training
    if pretraining_method == "pretrain_mask":

        print("To spectate training, open the tensorboard --logdir Checkpoints/Pretrain_mask_Checkpoints/" + "Small_BEiT_"+dataset+pretraining_method)
        lightning_model = Small_BEiT_Mask_pl(embed_dim=256, hidden_dim=512, num_channels=3,
                                          num_heads=8, num_layers=6, patch_size=4, num_patches=64)

        trainer = pl.Trainer(default_root_dir=os.path.join(CHECKPOINT_PATH, "Small_BEiT_"+dataset+pretraining_method),
                             gpus=1 if str(device) == "cuda:0" else 0,
                             max_epochs=180,
                             callbacks=[ModelCheckpoint(save_weights_only=True, mode="min", monitor="val_loss"),
                                        LearningRateMonitor("epoch")])

        trainer.fit(lightning_model, train_loader, val_loader)

    else:
        lightning_model = Small_BEiT_Mask_und_Rotation_pl(embed_dim=256, hidden_dim=512, num_channels=3,
                                          num_heads=8, num_layers=6, patch_size=4, num_patches=64, num_rot=4)
        print("To spectate training, open the tensorboard --logdir Checkpoints/Pretrain_mask_rot_Checkpoints/" + "Small_BEiT_" + dataset + pretraining_method)
        trainer = pl.Trainer(default_root_dir=os.path.join(CHECKPOINT_PATH, "Small_BEiT"+dataset+pretraining_method),
                             gpus=1 if str(device) == "cuda:0" else 0,
                             max_epochs=180,
                             callbacks=[ModelCheckpoint(save_weights_only=True, mode="min", monitor="val_loss"),
                                        LearningRateMonitor("epoch")])

        trainer.fit(lightning_model, train_loader, val_loader)



if __name__ == '__main__':
    device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    print("Device:", device)
    if args.method == 'finetune':
        finetune()
    elif args.method == 'pretrain_mask':
        pretrain()
    elif args.method == 'pretrain_mask_rot':
        pretrain()
    else:
        print('Flag --method must be set. Check for typos!')
















