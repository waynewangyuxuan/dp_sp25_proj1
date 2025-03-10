import os
import sys
import torch
import matplotlib.pyplot as plt
from torchvision.utils import make_grid

# Add src directory to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from data.cutmix import cutmix

def test_cutmix():
    """Test the CutMix augmentation and visualize the results"""
    # Create random images and labels
    batch_size = 8
    images = torch.randn(batch_size, 3, 32, 32)
    labels = torch.randint(0, 10, (batch_size,))
    
    # Apply CutMix
    mixed_images, mixed_labels = cutmix(images, labels, alpha=1.0)
    
    # Print shapes
    print(f"Original images shape: {images.shape}")
    print(f"Mixed images shape: {mixed_images.shape}")
    print(f"Original labels shape: {labels.shape}")
    print(f"Mixed labels shape: {mixed_labels.shape}")
    
    # Normalize images for visualization
    orig_grid = make_grid(images, nrow=4, normalize=True)
    mixed_grid = make_grid(mixed_images, nrow=4, normalize=True)
    
    # Convert to numpy for visualization
    orig_np = orig_grid.permute(1, 2, 0).numpy()
    mixed_np = mixed_grid.permute(1, 2, 0).numpy()
    
    # Plot the grids
    plt.figure(figsize=(12, 6))
    
    plt.subplot(1, 2, 1)
    plt.imshow(orig_np)
    plt.title("Original Images")
    plt.axis('off')
    
    plt.subplot(1, 2, 2)
    plt.imshow(mixed_np)
    plt.title("CutMix Images")
    plt.axis('off')
    
    plt.savefig("cutmix_samples.png")
    print("Saved CutMix visualization to cutmix_samples.png")
    
    print("CutMix test passed!")

if __name__ == "__main__":
    test_cutmix() 