import os
import torch
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data.data_module import CIFAR10DataModule

def test_dataloader():
    """Test the CIFAR-10 data loading functionality"""
    # Initialize data module
    data_module = CIFAR10DataModule(
        data_dir='data/cifar10',
        batch_size=128,
        num_workers=4,
        pin_memory=True
    )
    
    # Setup the datasets
    data_module.setup()
    
    # Get data loaders
    train_loader = data_module.train_dataloader()
    val_loader = data_module.val_dataloader()
    test_loader = data_module.test_dataloader()
    
    # Test loading a training batch
    images, labels = next(iter(train_loader))
    print(f"\nTraining batch shapes:")
    print(f"Images shape: {images.shape}")
    print(f"Labels shape: {labels.shape}")
    print(f"\nFirst 10 labels: {labels[:10]}")
    
    # Test loading a test batch
    test_images, test_indices = next(iter(test_loader))
    print(f"\nTest batch shapes:")
    print(f"Images shape: {test_images.shape}")
    print(f"Indices shape: {test_indices.shape}")
    print(f"First 10 indices: {test_indices[:10]}")
    
    # Print dataset sizes
    print(f"\nDataset sizes:")
    print(f"Training dataset size: {len(data_module.train_dataset)}")
    print(f"Validation dataset size: {len(data_module.val_dataset)}")
    print(f"Test dataset size: {len(data_module.test_dataset)}")
    
    # Print class names
    print(f"\nClass names:")
    print(data_module.train_dataset.label_names)

if __name__ == '__main__':
    test_dataloader() 