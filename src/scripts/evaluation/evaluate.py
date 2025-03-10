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

# Add TrainingConfig to safe globals
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

def load_model(model_path):
    """Load a trained model from a checkpoint file."""
    # Load checkpoint
    checkpoint = torch.load(model_path, map_location='cpu')
    
    # Get model configuration
    config = checkpoint.get('config', TrainingConfig())
    model_name = getattr(config, 'model_name', 'resnet_small')
    
    # Create model
    model = ModelFactory.create(model_name)
    
    # Load model weights
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    return model, config

def generate_predictions(model, data_loader, device='cuda'):
    """Generate predictions using the model."""
    model.eval()
    model.to(device)
    
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch in data_loader:
            inputs, targets = batch
            inputs = inputs.to(device)
            
            # Forward pass
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(targets.numpy())
    
    return np.array(all_preds), np.array(all_labels)

def main():
    parser = argparse.ArgumentParser(description='Evaluate a trained model')
    parser.add_argument('--model-path', type=str, default=None, 
                        help='Path to the model checkpoint file')
    args = parser.parse_args()
    
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
    print(f"Evaluation results will be saved to: {eval_dir}")
    
    # Copy model file to evaluation directory
    model_filename = os.path.basename(args.model_path)
    model_copy_path = os.path.join(eval_dir, "model.pth")
    if not os.path.exists(model_copy_path):
        import shutil
        shutil.copy2(args.model_path, model_copy_path)
    
    # Load model
    model, config = load_model(args.model_path)
    
    # Create data module
    data_module = CIFAR10DataModule(
        data_dir=config.data_dir,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory
    )
    data_module.setup()
    
    # Set device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Generate predictions on test set
    test_loader = data_module.test_dataloader()
    preds, labels = generate_predictions(model, test_loader, device)
    
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
    
    print(f"Evaluation completed. Results saved to {eval_dir}")

if __name__ == '__main__':
    main() 