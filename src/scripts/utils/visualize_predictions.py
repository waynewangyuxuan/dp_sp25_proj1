import os
import sys
import random
import numpy as np
import torch
import matplotlib.pyplot as plt
from PIL import Image
import torchvision.transforms as transforms
from pathlib import Path

# Add the src directory to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from models import ModelFactory
from data.data_module import CIFAR10DataModule

# Import the config class that's causing the unpickling error
sys.path.append('/scratch/yw5954/dp_sp25_proj1/src')
try:
    from configs.resnet_randaugment_config import ResNetRandAugmentConfig
    # Add the class to safe globals if using PyTorch 2.6+
    try:
        torch.serialization.add_safe_globals([ResNetRandAugmentConfig])
    except AttributeError:
        # PyTorch version doesn't have this function, which is fine for older versions
        pass
except ImportError:
    print("Warning: Could not import ResNetRandAugmentConfig, but will try to load model anyway")

def denormalize(tensor):
    """Denormalize a tensor using CIFAR-10 mean and std"""
    mean = torch.tensor([0.4914, 0.4822, 0.4465]).view(3, 1, 1)
    std = torch.tensor([0.2023, 0.1994, 0.2010]).view(3, 1, 1)
    return tensor * std + mean

def visualize_predictions(model_path, num_images=10, output_path=None):
    """
    Visualize predictions from a trained model on random validation images
    
    Args:
        model_path: Path to the trained model checkpoint
        num_images: Number of images to visualize
        output_path: Path to save the visualization
    """
    # Set device
    device = torch.device("cpu")  # Force CPU usage to avoid GPU memory issues
    print(f"Using device: {device}")
    
    # Load the model
    print(f"Loading model from {model_path}")
    try:
        # First try with weights_only=False to handle custom classes
        checkpoint = torch.load(model_path, map_location=device, weights_only=False)
        print("Successfully loaded checkpoint with weights_only=False")
    except Exception as e:
        print(f"Error loading with weights_only=False: {e}")
        try:
            # Try with pickle module directly as a fallback
            import pickle
            with open(model_path, 'rb') as f:
                checkpoint = pickle.load(f)
            print("Successfully loaded checkpoint with pickle")
        except Exception as e2:
            print(f"Error loading with pickle: {e2}")
            # Last resort: try to load just the state dict
            try:
                checkpoint = torch.load(model_path, map_location=device, weights_only=True)
                print("Successfully loaded checkpoint with weights_only=True")
            except Exception as e3:
                print(f"All loading methods failed. Last error: {e3}")
                print("Cannot proceed without loading the model.")
                return
    
    # Extract model configuration from checkpoint
    if isinstance(checkpoint, dict) and 'model_config' in checkpoint:
        model_config = checkpoint['model_config']
        model_name = model_config.get('model_name', 'resnet_small')
        print(f"Model name from config: {model_name}")
        
        # Create model with the same configuration
        model = ModelFactory.create(
            model_name,
            **{k: v for k, v in model_config.items() if k != 'model_name'}
        )
    else:
        print("No model config found in checkpoint, using default resnet_small")
        model = ModelFactory.create('resnet_small')
    
    # Load model weights
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    elif isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
        model.load_state_dict(checkpoint['state_dict'])
    else:
        try:
            model.load_state_dict(checkpoint)
        except Exception as e:
            print(f"Error loading state dict: {e}")
            print("Trying to continue with uninitialized model (results may be random)")
    
    model = model.to(device)
    model.eval()
    
    # Load CIFAR-10 validation dataset
    data_module = CIFAR10DataModule(data_dir='/scratch/yw5954/dp_sp25_proj1/data/cifar10')
    data_module.setup()
    val_dataset = data_module.val_dataset
    
    # Get class names
    class_names = val_dataset.label_names
    print(f"Class names: {class_names}")
    
    # Select random indices
    indices = random.sample(range(len(val_dataset)), num_images)
    
    # Create figure
    fig, axes = plt.subplots(num_images, 1, figsize=(10, num_images * 3))
    if num_images == 1:
        axes = [axes]
    
    # Get predictions for each image
    for i, idx in enumerate(indices):
        # Get image and label
        image, label = val_dataset[idx]
        
        # Get prediction
        with torch.no_grad():
            image_tensor = image.unsqueeze(0).to(device)
            output = model(image_tensor)
            _, predicted = torch.max(output, 1)
            
            # Get probabilities
            probabilities = torch.nn.functional.softmax(output, dim=1)[0]
        
        # Convert image tensor to numpy for display
        image_np = denormalize(image).permute(1, 2, 0).cpu().numpy()
        image_np = np.clip(image_np, 0, 1)
        
        # Plot image
        ax = axes[i]
        ax.imshow(image_np)
        
        # Get true and predicted class names
        true_class = class_names[label]
        pred_class = class_names[predicted.item()]
        
        # Set title with true and predicted classes
        title = f"True: {true_class}, Predicted: {pred_class}"
        if true_class == pred_class:
            title += " ✓"
        else:
            title += " ✗"
        
        ax.set_title(title)
        ax.axis('off')
        
        # Add probability bar chart as an inset
        inset_ax = ax.inset_axes([0.65, 0.05, 0.3, 0.3])
        top_k = 3
        top_probs, top_indices = torch.topk(probabilities, top_k)
        top_classes = [class_names[idx.item()] for idx in top_indices]
        
        # Plot top-k probabilities
        bars = inset_ax.barh(range(top_k), top_probs.cpu().numpy(), color='skyblue')
        inset_ax.set_yticks(range(top_k))
        inset_ax.set_yticklabels(top_classes)
        inset_ax.set_xlim(0, 1)
        inset_ax.set_title('Top Predictions', fontsize=8)
    
    plt.tight_layout()
    
    # Save or show the figure
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Visualization saved to {output_path}")
    else:
        plt.show()

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Visualize model predictions on CIFAR-10 validation images")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the trained model checkpoint")
    parser.add_argument("--num_images", type=int, default=10, help="Number of images to visualize")
    parser.add_argument("--output_path", type=str, default=None, help="Path to save the visualization")
    
    args = parser.parse_args()
    
    visualize_predictions(
        model_path=args.model_path,
        num_images=args.num_images,
        output_path=args.output_path
    ) 