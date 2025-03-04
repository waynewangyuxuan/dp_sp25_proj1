import os
import torch
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from .cifar10_dataset import CIFAR10Dataset
import numpy as np
import torchvision.transforms as T
from torch.utils.data import Dataset

class Cutout:
    """Randomly mask out one or more patches from an image."""
    def __init__(self, holes: int, length: int):
        self.holes = holes
        self.length = length

    def __call__(self, img):
        h = img.size(1)
        w = img.size(2)
        
        mask = np.ones((h, w), np.float32)
        
        for n in range(self.holes):
            y = np.random.randint(h)
            x = np.random.randint(w)
            
            y1 = np.clip(y - self.length // 2, 0, h)
            y2 = np.clip(y + self.length // 2, 0, h)
            x1 = np.clip(x - self.length // 2, 0, w)
            x2 = np.clip(x + self.length // 2, 0, w)
            
            mask[y1:y2, x1:x2] = 0
            
        mask = torch.from_numpy(mask)
        mask = mask.expand_as(img)
        img = img * mask
        
        return img

class Mixup:
    def __init__(self, alpha: float = 1.0):
        self.alpha = alpha
    
    def __call__(self, batch, targets):
        if self.alpha > 0:
            lam = np.random.beta(self.alpha, self.alpha)
        else:
            lam = 1

        batch_size = batch.size(0)
        index = torch.randperm(batch_size)

        mixed_batch = lam * batch + (1 - lam) * batch[index, :]
        target_a, target_b = targets, targets[index]
        
        return mixed_batch, (target_a, target_b, lam)

class MixupDataset(Dataset):
    """Mixup Dataset wrapper"""
    def __init__(self, dataset, alpha=1.0):
        self.dataset = dataset
        self.alpha = alpha

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        input, target = self.dataset[idx]
        
        if self.alpha > 0:
            lam = np.random.beta(self.alpha, self.alpha)
            rand_idx = torch.randint(len(self.dataset), (1,)).item()
            
            rand_input, rand_target = self.dataset[rand_idx]
            input = lam * input + (1 - lam) * rand_input
            return input, (target, rand_target, lam)
        
        return input, target

class CIFAR10DataModule:
    """Data module for CIFAR-10 dataset"""
    
    def __init__(self,
                 data_dir: str = 'data/cifar10',
                 batch_size: int = 128,
                 num_workers: int = 4,
                 pin_memory: bool = True):
        """
        Args:
            data_dir: Path to CIFAR-10 dataset
            batch_size: Batch size for data loaders
            num_workers: Number of workers for data loading
            pin_memory: Whether to pin memory for faster data transfer
        """
        self.data_dir = os.path.join(data_dir, 'cifar-10-python', 'cifar-10-batches-py')  # Updated path
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        
        # Enhanced augmentation parameters
        self.cutout = Cutout(holes=1, length=16)
        self.mixup = Mixup(alpha=1.0)
        
        # Define transforms
        self.train_transform = T.Compose([
            T.RandomCrop(32, padding=4),
            T.RandomHorizontalFlip(),
            T.RandomRotation(15),
            T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            T.ToTensor(),
            T.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            self.cutout
        ])
        
        # Validation/Test data transformation
        self.val_transform = T.Compose([
            T.ToTensor(),
            T.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ])
    
    def setup(self):
        """Setup train, val, and test datasets"""
        train_dataset = CIFAR10Dataset(
            root_dir=self.data_dir,
            transform=self.train_transform,
            is_train=True
        )
        
        # Split into train and val
        train_size = int(0.8 * len(train_dataset))
        val_size = len(train_dataset) - train_size
        
        train_dataset, val_dataset = torch.utils.data.random_split(
            train_dataset, [train_size, val_size]
        )
        
        # Apply Mixup to training dataset only
        self.train_dataset = MixupDataset(train_dataset, alpha=1.0)
        self.val_dataset = val_dataset
        
        # Setup test dataset
        self.test_dataset = CIFAR10Dataset(
            root_dir=self.data_dir,
            transform=self.val_transform,
            is_train=False
        )
    
    def train_dataloader(self) -> DataLoader:
        """Get training data loader"""
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            collate_fn=self._mixup_collate if self.training else None
        )
    
    def val_dataloader(self) -> DataLoader:
        """Get validation data loader"""
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory
        )
    
    def test_dataloader(self) -> DataLoader:
        """Get test data loader"""
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory
        )

    def _mixup_collate(self, batch):
        images = torch.stack([b[0] for b in batch])
        targets = torch.tensor([b[1] for b in batch])
        return self.mixup(images, targets) 