import os
from typing import Optional, Tuple
import torch
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from .cifar10_dataset import CIFAR10Dataset

class CIFAR10DataModule:
    """Data Module for CIFAR-10 dataset"""
    
    def __init__(self,
                 data_dir: str,
                 batch_size: int = 128,
                 num_workers: int = 4,
                 pin_memory: bool = True):
        """
        Args:
            data_dir: Base path to the CIFAR-10 dataset directory
            batch_size: Number of samples per batch
            num_workers: Number of workers for DataLoader
            pin_memory: Whether to pin memory in DataLoader
        """
        # Update data_dir to point to the actual batch files
        self.data_dir = os.path.join(data_dir, 'cifar-10-python', 'cifar-10-batches-py')
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        
        # Define transforms
        self.train_transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.Normalize(
                mean=[0.4914, 0.4822, 0.4465],
                std=[0.2023, 0.1994, 0.2010]
            )
        ])
        
        self.val_transform = transforms.Compose([
            transforms.Normalize(
                mean=[0.4914, 0.4822, 0.4465],
                std=[0.2023, 0.1994, 0.2010]
            )
        ])
        
        # Use the same transform as validation for test
        self.test_transform = self.val_transform
        
    def setup(self) -> None:
        """Set up the datasets"""
        # Training data (first 4 batches)
        self.train_dataset = CIFAR10Dataset(
            self.data_dir,
            ['data_batch_1', 'data_batch_2', 'data_batch_3', 'data_batch_4'],
            transform=self.train_transform
        )
        
        # Validation data (batch 5)
        self.val_dataset = CIFAR10Dataset(
            self.data_dir,
            ['data_batch_5'],
            transform=self.val_transform
        )
        
        # Test data (unlabeled)
        self.test_dataset = CIFAR10Dataset(
            self.data_dir,
            batch_files=None,  # Not used for test dataset
            transform=self.test_transform,
            is_test=True
        )
    
    def train_dataloader(self) -> DataLoader:
        """Create training DataLoader"""
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory
        )
    
    def val_dataloader(self) -> DataLoader:
        """Create validation DataLoader"""
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory
        )
    
    def test_dataloader(self) -> DataLoader:
        """Create test DataLoader"""
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory
        ) 