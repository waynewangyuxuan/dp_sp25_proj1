import os
import sys
import argparse
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from datetime import datetime
from torch.serialization import add_safe_globals

# Add src directory to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from models import ModelFactory
from data.data_module import CIFAR10DataModule
from configs.train_config import TrainingConfig

# Add TrainingConfig to safe globals for PyTorch 2.6+ compatibility
add_safe_globals([TrainingConfig])

def get_evaluation_dir(model_path, output_dir="outputs"):
    """Create and return the evaluation directory path."""
    model_filename = os.path.basename(model_path)
    model_name = os.path.splitext(model_filename)[0]
    
    # Create a directory for this evaluation
    timestamp = Path(model_path).parent.name
    eval_dir = os.path.join(output_dir, "evaluations", f"{model_name}_{timestamp}")
    os.makedirs(eval_dir, exist_ok=True)
    
    return eval_dir

def get_best_models_dir(config: TrainingConfig) -> str:
    """Get the directory for storing best performing models"""
    return os.path.join(config.output_dir, "best_models")

def create_evaluation_folder(config: TrainingConfig, model_path: str) -> str:
    """Create a folder for this evaluation run containing the CSV and model link"""
    # Get validation accuracy from model checkpoint
    checkpoint = torch.load(model_path, map_location='cpu')
    val_acc = checkpoint['best_acc']
    
    # Create timestamp-based folder name with validation accuracy
    timestamp = datetime.now().strftime("%Y_%m_%d_%H_%M")
    # Extract model name from the path to include in folder name
    model_filename = os.path.basename(model_path)
    folder_name = f"{config.experiment_name}_val_acc_{val_acc:.2f}_{timestamp}_model_{model_filename}"
    
    # Create evaluation directory if it doesn't exist
    eval_dir = get_evaluation_dir(model_path)
    os.makedirs(eval_dir, exist_ok=True)
    
    # Create specific evaluation folder
    eval_folder = os.path.join(eval_dir, folder_name)
    os.makedirs(eval_folder, exist_ok=True)
    
    return eval_folder

def load_model(model_path, device='cpu'):
    """Load a trained model from a checkpoint file."""
    # Load checkpoint
    try:
        checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    except RuntimeError as e:
        if "Weights only load failed" in str(e):
            print("Retrying with weights_only=True...")
            checkpoint = torch.load(model_path, map_location=device, weights_only=True)
        else:
            raise e
    
    # Get model configuration
    config = checkpoint.get('config', TrainingConfig())
    model_name = getattr(config, 'model_name', 'resnet_small')
    
    # Create model with the correct architecture
    model = ModelFactory.create(model_name)
    
    # Load model weights
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    return model, config

def test_time_augmentation(model, image, num_transforms=8, device='cuda'):
    """Apply test-time augmentation to an image and average predictions."""
    model.eval()
    
    # Original image prediction
    with torch.no_grad():
        original_output = model(image.unsqueeze(0).to(device))
    
    # Initialize augmented predictions tensor
    all_outputs = original_output.clone()
    
    # Apply horizontal flip
    with torch.no_grad():
        flipped_image = torch.flip(image, dims=[-1])
        flipped_output = model(flipped_image.unsqueeze(0).to(device))
        all_outputs += flipped_output
    
    # Apply random crops and flips for additional transforms
    if num_transforms > 2:
        from torchvision import transforms
        
        # Create augmentation transforms
        crop_size = image.shape[-1]
        padding = crop_size // 8  # Add some padding for random crops
        
        transform = transforms.Compose([
            transforms.Pad(padding=padding),
            transforms.RandomCrop(size=crop_size),
            transforms.RandomHorizontalFlip(),
        ])
        
        # Apply additional transforms
        for _ in range(num_transforms - 2):
            # Apply transform
            aug_image = transform(image.unsqueeze(0)).squeeze(0)
            
            # Get prediction
            with torch.no_grad():
                aug_output = model(aug_image.unsqueeze(0).to(device))
                all_outputs += aug_output
    
    # Average predictions
    avg_output = all_outputs / num_transforms
    
    return avg_output

def evaluate_with_tta(model, data_loader, device='cuda', tta=False, tta_transforms=8):
    """Evaluate model with optional test-time augmentation."""
    model.eval()
    model.to(device)
    
    all_preds = []
    all_labels = []
    
    # Create progress bar
    pbar = tqdm(data_loader, desc="Evaluating", unit="batch")
    
    with torch.no_grad():
        for batch in pbar:
            inputs, targets = batch
            
            if tta:
                # Apply test-time augmentation to each image
                outputs = torch.zeros((inputs.size(0), model(inputs[0:1].to(device)).size(1)), device=device)
                
                for i in range(inputs.size(0)):
                    outputs[i] = test_time_augmentation(
                        model, inputs[i], num_transforms=tta_transforms, device=device
                    )
            else:
                # Standard forward pass
                inputs = inputs.to(device)
                outputs = model(inputs)
            
            # Get predictions
            _, preds = torch.max(outputs, 1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(targets.numpy())
    
    return np.array(all_preds), np.array(all_labels)

def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate a trained model with advanced options')
    parser.add_argument('--model-path', type=str, default=None, 
                        help='Path to the model checkpoint file')
    parser.add_argument('--tta', action='store_true',
                        help='Use test-time augmentation')
    parser.add_argument('--tta-transforms', type=int, default=8,
                        help='Number of transforms to use for test-time augmentation')
    parser.add_argument('--use-augment', action='store_true',
                        help='Use augmentations during evaluation (not recommended)')
    parser.add_argument('--ensemble', action='store_true',
                        help='Use model ensemble if multiple models are available')
    parser.add_argument('--output-dir', type=str, default=None,
                        help='Custom output directory for evaluation results')
    return parser.parse_args()

def main():
    # Parse command line arguments
    args = parse_args()
    
    # Find the best model if not specified
    if args.model_path is None:
        best_models_dir = get_best_models_dir(TrainingConfig())
        model_files = [f for f in os.listdir(best_models_dir) if f.endswith('.pth')]
        if not model_files:
            raise ValueError("No model files found in the best_models directory")
        
        # Use the first model file (you might want to implement a better selection strategy)
        args.model_path = os.path.join(best_models_dir, model_files[0])
    
    print(f"Loading model from: {args.model_path}")
    
    # Create evaluation directory
    eval_dir = get_evaluation_dir(args.model_path)
    if args.tta:
        eval_dir = f"{eval_dir}_tta{args.tta_transforms}"
    os.makedirs(eval_dir, exist_ok=True)
    print(f"Evaluation results will be saved to: {eval_dir}")
    
    # Copy model file to evaluation directory
    model_filename = os.path.basename(args.model_path)
    model_copy_path = os.path.join(eval_dir, "model.pth")
    if not os.path.exists(model_copy_path):
        import shutil
        shutil.copy2(args.model_path, model_copy_path)
    
    # Set device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Load model
    model, config = load_model(args.model_path, device)
    
    # Create data module
    data_module = CIFAR10DataModule(
        data_dir=config.data_dir,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
        use_cutmix=False,  # Disable CutMix for evaluation
        cutmix_alpha=config.cutmix_alpha,
        cutmix_prob=0.0,
        use_randaugment=args.use_augment,  # Disable RandAugment by default for evaluation
        randaugment_num_ops=config.randaugment_num_ops,
        randaugment_magnitude=config.randaugment_magnitude
    )
    data_module.setup()
    
    # Generate predictions on test set
    test_loader = data_module.test_dataloader()
    
    print(f"Evaluating model{' with test-time augmentation' if args.tta else ''}...")
    preds, labels = evaluate_with_tta(
        model, test_loader, device, tta=args.tta, tta_transforms=args.tta_transforms
    )
    
    # Calculate accuracy
    accuracy = (preds == labels).mean()
    print(f"Test accuracy: {accuracy:.5f}")
    
    # Save predictions to CSV
    results_df = pd.DataFrame({
        'true_label': labels,
        'predicted_label': preds,
        'correct': preds == labels
    })
    results_df.to_csv(os.path.join(eval_dir, "predictions.csv"), index=False)
    
    # Save summary
    with open(os.path.join(eval_dir, "summary.txt"), 'w') as f:
        f.write(f"Model: {config.model_name}\n")
        f.write(f"Test accuracy: {accuracy:.5f}\n")
        f.write(f"Total samples: {len(labels)}\n")
        f.write(f"Correct predictions: {(preds == labels).sum()}\n")
        f.write(f"Test-time augmentation: {args.tta}\n")
        if args.tta:
            f.write(f"TTA transforms: {args.tta_transforms}\n")
    
    print(f"Evaluation completed. Results saved to {eval_dir}")

if __name__ == '__main__':
    main() 