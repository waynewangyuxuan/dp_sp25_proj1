import os
import sys
import random
import numpy as np
import torch
import matplotlib.pyplot as plt
from PIL import Image
import torchvision.transforms as transforms
from pathlib import Path
from matplotlib import gridspec

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

def visualize_predictions(model_path, num_correct=8, num_incorrect=4, output_path=None):
    """
    Visualize predictions from a trained model on random validation images
    
    Args:
        model_path: Path to the trained model checkpoint
        num_correct: Number of correct predictions to show
        num_incorrect: Number of incorrect predictions to show
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
    
    # Find correct and incorrect predictions
    correct_samples = []
    incorrect_samples = []
    
    # We'll need to check more samples than we display to ensure we get enough correct/incorrect
    max_samples_to_check = 200
    indices_to_check = random.sample(range(len(val_dataset)), min(max_samples_to_check, len(val_dataset)))
    
    print("Finding correct and incorrect predictions...")
    for idx in indices_to_check:
        # Get image and label
        image, label = val_dataset[idx]
        
        # Get prediction
        with torch.no_grad():
            image_tensor = image.unsqueeze(0).to(device)
            output = model(image_tensor)
            _, predicted = torch.max(output, 1)
            
            # Get probabilities
            probabilities = torch.nn.functional.softmax(output, dim=1)[0]
        
        # Check if prediction is correct
        is_correct = (predicted.item() == label)
        
        # Store sample information
        sample_info = {
            'idx': idx,
            'image': image,
            'label': label,
            'predicted': predicted.item(),
            'probabilities': probabilities.cpu()
        }
        
        # Add to appropriate list
        if is_correct and len(correct_samples) < num_correct:
            correct_samples.append(sample_info)
        elif not is_correct and len(incorrect_samples) < num_incorrect:
            incorrect_samples.append(sample_info)
            
        # Check if we have enough samples
        if len(correct_samples) >= num_correct and len(incorrect_samples) >= num_incorrect:
            break
    
    print(f"Found {len(correct_samples)} correct and {len(incorrect_samples)} incorrect predictions")
    
    # If we don't have enough samples, adjust our expectations
    num_correct = min(num_correct, len(correct_samples))
    num_incorrect = min(num_incorrect, len(incorrect_samples))
    total_samples = num_correct + num_incorrect
    
    # Create a grid layout for A4 proportions (√2:1 ratio)
    # A4 is 210mm × 297mm, so aspect ratio is 1:1.414
    a4_ratio = 1.414
    
    # Determine grid dimensions based on total samples
    if total_samples <= 6:
        grid_cols = 3
        grid_rows = 2
    elif total_samples <= 12:
        grid_cols = 4
        grid_rows = 3
    else:
        grid_cols = 4
        grid_rows = 4
    
    # Create figure with A4 proportions
    fig_width = 8.27  # inches (A4 width)
    fig_height = 11.69  # inches (A4 height)
    fig = plt.figure(figsize=(fig_width, fig_height))
    
    # Create grid layout
    gs = gridspec.GridSpec(grid_rows, grid_cols, figure=fig)
    
    # Combine samples and shuffle to mix correct and incorrect
    all_samples = correct_samples[:num_correct] + incorrect_samples[:num_incorrect]
    random.shuffle(all_samples)
    
    # Plot each sample
    for i, sample in enumerate(all_samples):
        if i >= grid_rows * grid_cols:
            break
            
        # Calculate grid position
        row = i // grid_cols
        col = i % grid_cols
        
        # Create subplot
        ax = fig.add_subplot(gs[row, col])
        
        # Get image and convert to numpy
        image = sample['image']
        image_np = denormalize(image).permute(1, 2, 0).cpu().numpy()
        image_np = np.clip(image_np, 0, 1)
        
        # Plot image
        ax.imshow(image_np)
        
        # Get true and predicted class names
        true_class = class_names[sample['label']]
        pred_class = class_names[sample['predicted']]
        
        # Set title with true and predicted classes
        is_correct = (sample['label'] == sample['predicted'])
        title = f"True: {true_class}\nPred: {pred_class}"
        if is_correct:
            title += " ✓"
            ax.set_title(title, color='green', fontsize=9)
        else:
            title += " ✗"
            ax.set_title(title, color='red', fontsize=9)
        
        ax.axis('off')
        
        # Add probability bar chart as an inset
        inset_ax = ax.inset_axes([0.55, 0.02, 0.4, 0.3])
        top_k = 3
        top_probs, top_indices = torch.topk(sample['probabilities'], top_k)
        top_classes = [class_names[idx.item()] for idx in top_indices]
        
        # Plot top-k probabilities
        bars = inset_ax.barh(range(top_k), top_probs.numpy(), color='skyblue')
        inset_ax.set_yticks(range(top_k))
        inset_ax.set_yticklabels([c[:4] for c in top_classes], fontsize=7)  # Abbreviate class names
        inset_ax.set_xlim(0, 1)
        inset_ax.set_xticks([0, 0.5, 1.0])
        inset_ax.set_xticklabels(['0', '0.5', '1.0'], fontsize=6)
        inset_ax.tick_params(axis='both', which='both', length=0)
        for spine in inset_ax.spines.values():
            spine.set_visible(False)
    
    # Add a title for the entire figure
    plt.suptitle("CIFAR-10 Model Predictions", fontsize=16, y=0.98)
    
    # Add a subtitle with model information
    if isinstance(checkpoint, dict) and 'model_config' in checkpoint:
        model_info = f"Model: {model_name}"
        if 'val_acc' in checkpoint:
            model_info += f" | Validation Accuracy: {checkpoint['val_acc']:.2f}%"
        plt.figtext(0.5, 0.94, model_info, ha='center', fontsize=10)
    
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    
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
    parser.add_argument("--correct", type=int, default=8, help="Number of correct predictions to show")
    parser.add_argument("--incorrect", type=int, default=4, help="Number of incorrect predictions to show")
    parser.add_argument("--output_path", type=str, default=None, help="Path to save the visualization")
    
    args = parser.parse_args()
    
    visualize_predictions(
        model_path=args.model_path,
        num_correct=args.correct,
        num_incorrect=args.incorrect,
        output_path=args.output_path
    ) 