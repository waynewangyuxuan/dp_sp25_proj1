import os
import pickle
import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from typing import Optional, List, Tuple

from .cifar10_dataset import CIFAR10Dataset
from .augmentations import CutMixCollator

class CIFAR10DataModule:
    """Data module for CIFAR-10 dataset."""
    
    def __init__(self, 
                 data_dir: str = 'data/cifar10',
                 batch_size: int = 128,
                 num_workers: int = 4,
                 pin_memory: bool = True,
                 use_cutmix: bool = True,
                 cutmix_alpha: float = 1.0,
                 cutmix_prob: float = 0.5):
        """
        Args:
            data_dir: Base path to the CIFAR-10 dataset directory
            batch_size: Batch size for training and validation
            num_workers: Number of workers for data loading
            pin_memory: Whether to pin memory for faster data transfer to GPU
            use_cutmix: Whether to use CutMix augmentation
            cutmix_alpha: Alpha parameter for CutMix Beta distribution
            cutmix_prob: Probability of applying CutMix to a batch
        """
        self.data_dir = os.path.join(data_dir, 'cifar-10-python/cifar-10-batches-py')
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.use_cutmix = use_cutmix
        self.cutmix_alpha = cutmix_alpha
        self.cutmix_prob = cutmix_prob
        
        # Define transformations
        self.train_transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            # Enhanced color jittering
            transforms.ColorJitter(
                brightness=0.3,
                contrast=0.3,
                saturation=0.3,
                hue=0.1
            ),
            # Add random perspective for more variety
            transforms.RandomPerspective(distortion_scale=0.2, p=0.5),
            # Add random erasing (similar to Cutout but built into torchvision)
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.4914, 0.4822, 0.4465],
                std=[0.2023, 0.1994, 0.2010]
            ),
            transforms.RandomErasing(p=0.3, scale=(0.02, 0.2), ratio=(0.3, 3.3), value=0)
        ])
        
        self.val_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.4914, 0.4822, 0.4465],
                std=[0.2023, 0.1994, 0.2010]
            )
        ])
        
        # We'll handle CutMix in the trainer instead of using a collator
        self.cutmix_collator = None
    
    def setup(self):
        """Set up the datasets."""
        # Training dataset (batches 1-4)
        self.train_dataset = CIFAR10Dataset(
            root_dir=self.data_dir,
            transform=self.train_transform,
            is_train=True,
            batch_files=['data_batch_1', 'data_batch_2', 'data_batch_3', 'data_batch_4']
        )
        
        # Validation dataset (batch 5)
        self.val_dataset = CIFAR10Dataset(
            root_dir=self.data_dir,
            transform=self.val_transform,
            is_train=True,
            batch_files=['data_batch_5']
        )
        
        # Test dataset (unlabeled)
        self.test_dataset = CIFAR10Dataset(
            root_dir=self.data_dir,
            transform=self.val_transform,
            is_train=False
        )
    
    def train_dataloader(self) -> DataLoader:
        """Create the training data loader."""
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory
        )
    
    def val_dataloader(self) -> DataLoader:
        """Create the validation data loader."""
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory
        )
    
    def test_dataloader(self) -> DataLoader:
        """Create the test data loader."""
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory
        ) 