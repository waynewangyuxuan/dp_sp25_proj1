import os
import sys
import torch

# Add src directory to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from data.data_module import CIFAR10DataModule

def check_test_data():
    """Check the test data loader and print statistics"""
    # Create data module
    data_module = CIFAR10DataModule(
        data_dir="data/cifar10",
        batch_size=128,
        num_workers=4
    )
    
    # Setup data module
    data_module.setup()
    
    # Get test dataloader
    test_loader = data_module.test_dataloader()
    
    # Get a batch of data
    images, indices = next(iter(test_loader))
    
    # Print shapes
    print(f"Test images shape: {images.shape}")
    print(f"Test indices shape: {indices.shape}")
    
    # Print statistics
    print(f"Test dataset size: {len(data_module.test_dataset)}")
    print(f"Number of batches: {len(test_loader)}")
    
    print("Test data check passed!")

if __name__ == "__main__":
    check_test_data() 