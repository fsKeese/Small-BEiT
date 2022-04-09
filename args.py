import argparse

parser = argparse.ArgumentParser(description='RecPlay')

#Dataset
##############
parser.add_argument('-d', '--dataset', type=str, help="Set dataset", default="None", choices=['CIFAR10', 'STL10'])
##############

#Method
##############
parser.add_argument('-m', '--method', type=str, help='Set train method', default='None', choices=['pretrain_mask', 'pretrain_mask_rot', 'finetune'])
##############

#Finetuning
##############
parser.add_argument('-f', '--finetune', type=str, help='Set Path of Pretrained Model if available', default='None')
parser.add_argument('-p', '--pretrained', type=str, help='Set how the Model was pretrained', default='From_Scratch', choices=['mask', 'mask_rot'])
##############

parser.print_help()

args = parser.parse_args()