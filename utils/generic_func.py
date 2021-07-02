import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import numpy as np
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt
from Albumentationtransform import AlbumentationTransforms

import albumentations as A


def get_data_set(name_of_dataset="cifar", transform_list=[]):
    transformation_list = []
    if "normalize" in transform_list:
        transformation_list.append(A.Normalize(mean=(0.4914, 0.4822, 0.4465), std=(0.2023, 0.1994, 0.2010)))
    if "randomcrop" in transform_list:
        transformation_list.append(A.RandomCrop(width=32, height=32))
    if "random_rotate" in transform_list:
        transformation_list.append(A.Rotate(limit=5))
    if "cutout" in transform_list:
        transformation_list.append(A.CoarseDropout(p=0.5, max_holes = 1, max_height=16, max_width=16, min_holes = 1, min_height=16, min_width=16, fill_value=(0.4914, 0.4822, 0.4465), mask_fill_value = None))
    if "shift_scale_rotate" in transform_list:
        transformation_list.append(A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.05, rotate_limit=15, p=0.5))
    if "grayscale" in transform_list:
        transformation_list.append(A.ToGray(p=0.5))

    albumentation_transformation_list = AlbumentationTransforms(transformation_list)

    if name_of_dataset == "cifar":

    # For train set we are using modified data class because need to apply various transformation
        train_set = torchvision.datasets.CIFAR10(root='/', train=True, download=True,
                                    transform=albumentation_transformation_list)

        # For test set we are using default CIFAR 10 of pytorch 
        test_set = torchvision.datasets.CIFAR10(root='/', train=False,
                                        download=True, transform=transforms.Compose([transforms.ToTensor(),
                                                                                        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
                                                                                    ]))

    return train_set, test_set
  


def plot_graph(train_loss, train_acc, test_loss, test_acc):
  fig, axs = plt.subplots(2,2,figsize=(15,10))
  axs[0, 0].plot(train_loss)
  axs[0, 0].set_title("Training Loss")
  axs[1, 0].plot(train_acc)
  axs[1, 0].set_title("Training Accuracy")
  axs[0, 1].plot(test_loss)
  axs[0, 1].set_title("Test Loss")
  axs[1, 1].plot(test_acc)
  axs[1, 1].set_title("Test Accuracy")