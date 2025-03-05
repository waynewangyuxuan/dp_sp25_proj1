import os
import sys
import torch
import matplotlib.pyplot as plt
import numpy as np

# Add src directory to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.data_module import CIFAR10DataModule
from data.augmentations import cutmix

def denormalize(tensor, mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010]):
    """Denormalize a tensor image with mean and standard deviation."""
    mean = torch.tensor(mean).view(3, 1, 1)
    std = torch.tensor(std).view(3, 1, 1)
    return tensor * std + mean

def visualize_cutmix():
    """Visualize CutMix augmentation."""
    # Create data module
    data_module = CIFAR10DataModule(
        data_dir 