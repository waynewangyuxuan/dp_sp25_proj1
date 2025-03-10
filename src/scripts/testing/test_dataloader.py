import os
import sys
import torch
import matplotlib.pyplot as plt
from torchvision.utils import make_grid

# Add src directory to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from data.data_module import CIFAR10DataModule

def test_dataloader():
    """Test the CIFAR10 data module and visualize some samples"""
    # Create data module
    data_module = CIFAR10DataModule(
        data_dir="data/cifar10",
        batch_size=16,
        num_workers=4
    )
    
    # Setup data module
    data_module.setup()
    
    # Get train dataloader
    train_loader = data_module.train_dataloader()
    
    # Get a batch of data
    images, labels = next(iter(train_loader))
    
    # Print shapes
    print(f"Images shape: {images.shape}")
    print(f"Labels shape: {labels.shape}")
    
    # Print label values
    print(f"Labels: {labels}")
    
    # Print image statistics
    print(f"Image min value: {images.min()}")
    print(f"Image max value: {images.max()}")
    print(f"Image mean: {images.mean()}")
    print(f"Image std: {images.std()}")
    
    # Create a grid of images
    grid = make_grid(images[:16], nrow=4, normalize=True)
    
    # Convert to numpy for visualization
    grid_np = grid.permute(1, 2, 0).numpy()
    
    # Plot the grid
    plt.figure(figsize=(10, 10))
    plt.imshow(grid_np)
    plt.title("CIFAR-10 Training Images")
    plt.axis('off')
    plt.savefig("cifar10_samples.png")
    print("Saved sample images to cifar10_samples.png")
    
    print("Dataloader test passed!")

if __name__ == "__main__":
    test_dataloader() 