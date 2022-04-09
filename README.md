# Small BEiT

Before starting make sure that all requirements are installed. Use pip install -r requirements.txt.
Check your torch versions fit your CUDA version.

##For Pretraining the Small-BEiT on CIFAR10 and/or STL10 use following commands:

CIFAR10 - Mask task:

* python main.py --dataset CIFAR10 --method pretrain_mask

CIFAR10 - Mask and Rotation task:

* python main.py --dataset CIFAR10 --method pretrain_mask_rot

For STL - just change the dataset:

STL10 - Mask task:

* python main.py --dataset STL10 --method pretrain_mask

STL10 - Mask and Rotation task:

* python main.py --dataset STL10 --method pretrain_mask_rot




##For Finetuning the Small-BEiT on CIFAR10 and/or STL10 use following commands:

For finetuning it is extremely important to add the to be finetuned Checkpoint PATH with flag --finetune and set the method of the checkpoint with --pretrained. Make sure to set the correct method which has been used for the pretrained Model. Chose from mask and mask_rot.
Use for Checkpoint PATH the Path to the ckpt file. All files are saved in the checkpoint folders.

CIFAR10 finetune with mask pretraining initialization

* python main.py --dataset CIFAR10 --method finetune --finetune PATH_TO_CHECKPOINT --pretrained mask

CIFAR10 finetune with mask and rot pretraining initialization

* python main.py --dataset CIFAR10 --method finetune --finetune PATH_TO_CHECKPOINT --pretrained mask_rot

STL10 finetune with mask pretraining initialization

* python main.py --dataset STL10 --method finetune --finetune PATH_TO_CHECKPOINT --pretrained mask

STL10 finetune with mask and rot pretraining initialization

* python main.py --dataset STL10 --method finetune --finetune PATH_TO_CHECKPOINT --pretrained mask_rot

##Training Small-BEiT From Scratch

For training the Small-BEiT from scratch use the finetuning flags, without the pretrained model:

CIFAR10

*python main.py --dataset CIFAR10 --method finetune

STL10

*python main.py --dataset STL10 --method finetune