#!/usr/bin/env python
"""
Script to extract validation accuracy from a model checkpoint file.
Usage: python get_model_accuracy.py /path/to/model.pth
"""

import os
import sys
import torch
import argparse
import pandas as pd
from pathlib import Path
from datetime import datetime
from torch.serialization import add_safe_globals

# Add src directory to Python path to ensure imports work correctly
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# Import TrainingConfig and add it to safe globals
from configs.train_config import TrainingConfig
add_safe_globals([TrainingConfig])

from models import ModelFactory
from data.data_module import CIFAR10DataModule

def parse_args():
    parser = argparse.ArgumentParser(description='Extract validation accuracy from a model checkpoint')
    parser.add_argument('model_path', type=str, help='Path to the model checkpoint (.pth file)')
    parser.add_argument('--verbose', '-v', action='store_true', help='Show detailed model information')
    parser.add_argument('--weights-only', action='store_true', help='Use weights_only=True for loading (safer but may fail)')
    return parser.parse_args()

def get_model_info(model_path, weights_only=False):
    """Extract information from a model checkpoint file"""
    if not os.path.exists(model_path):
        print(f"Error: Model file not found at {model_path}")
        return None
    
    try:
        # Load checkpoint with appropriate weights_only setting
        checkpoint = torch.load(model_path, map_location='cpu', weights_only=weights_only)
        
        # Extract information
        info = {
            'model_path': model_path,
            'filename': os.path.basename(model_path),
            'file_size_mb': os.path.getsize(model_path) / (1024 * 1024),
            'best_accuracy': checkpoint.get('best_acc', None),
            'epoch': checkpoint.get('epoch', None),
            'model_name': None,
            'experiment_name': None,
            'num_parameters': None
        }
        
        # Extract config information if available
        if 'config' in checkpoint:
            config = checkpoint['config']
            info['model_name'] = getattr(config, 'model_name', None)
            info['experiment_name'] = getattr(config, 'experiment_name', None)
        
        # Extract model parameters count if state_dict is available
        if 'model_state_dict' in checkpoint:
            param_count = sum(p.numel() for p in checkpoint['model_state_dict'].values())
            info['num_parameters'] = param_count
        
        return info
    
    except Exception as e:
        print(f"Error loading model file: {e}")
        
        # If we failed with weights_only=True, suggest trying with weights_only=False
        if weights_only:
            print("\nTry running again without the --weights-only flag for less restrictive loading.")
        # If we failed with weights_only=False, provide more detailed troubleshooting
        else:
            print("\nTroubleshooting tips:")
            print("1. Make sure you're using the correct PyTorch version")
            print("2. Check if the model was saved with a compatible PyTorch version")
            print("3. Try running with --weights-only flag if you're concerned about security")
        
        return None

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
    
    model = model.to(device)
    model.eval()
    
    return model, config

def evaluate_model(model, data_loader, device='cpu'):
    """Evaluate model accuracy on the given data loader."""
    correct = 0
    total = 0
    
    with torch.no_grad():
        for inputs, targets in data_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += targets.size(0)
            correct += (predicted == targets).sum().item()
    
    accuracy = 100 * correct / total
    return accuracy

def main():
    parser = argparse.ArgumentParser(description='Get model accuracy on test set')
    parser.add_argument('--model-path', type=str, required=True,
                        help='Path to the model checkpoint file')
    parser.add_argument('--data-dir', type=str, default='data/cifar10',
                        help='Directory containing the CIFAR-10 dataset')
    parser.add_argument('--batch-size', type=int, default=128,
                        help='Batch size for evaluation')
    args = parser.parse_args()
    
    # Set device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Load model
    print(f"Loading model from: {args.model_path}")
    model, config = load_model(args.model_path, device)
    
    # Create data module
    data_module = CIFAR10DataModule(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        num_workers=4,
        pin_memory=True
    )
    data_module.setup()
    
    # Evaluate on test set
    test_loader = data_module.test_dataloader()
    test_accuracy = evaluate_model(model, test_loader, device)
    print(f"Test accuracy: {test_accuracy:.2f}%")
    
    # Evaluate on validation set
    val_loader = data_module.val_dataloader()
    val_accuracy = evaluate_model(model, val_loader, device)
    print(f"Validation accuracy: {val_accuracy:.2f}%")
    
    # Save results
    results = {
        'model_path': args.model_path,
        'model_name': config.model_name if hasattr(config, 'model_name') else 'unknown',
        'test_accuracy': test_accuracy,
        'val_accuracy': val_accuracy
    }
    
    # Create results directory
    results_dir = Path('results')
    results_dir.mkdir(exist_ok=True)
    
    # Save to CSV
    results_df = pd.DataFrame([results])
    csv_path = results_dir / 'model_accuracy.csv'
    
    # Append to existing CSV if it exists
    if csv_path.exists():
        existing_df = pd.read_csv(csv_path)
        results_df = pd.concat([existing_df, results_df], ignore_index=True)
    
    results_df.to_csv(csv_path, index=False)
    print(f"Results saved to {csv_path}")

if __name__ == '__main__':
    main() 