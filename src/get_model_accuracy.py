#!/usr/bin/env python
"""
Script to extract validation accuracy from a model checkpoint file.
Usage: python get_model_accuracy.py /path/to/model.pth
"""

import os
import sys
import torch
import argparse
from datetime import datetime
from torch.serialization import add_safe_globals

# Add src directory to Python path to ensure imports work correctly
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import TrainingConfig and add it to safe globals
from configs.train_config import TrainingConfig
add_safe_globals([TrainingConfig])

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

def main():
    args = parse_args()
    
    print(f"\nExtracting information from: {args.model_path}")
    info = get_model_info(args.model_path, weights_only=args.weights_only)
    
    if info:
        # Always show the validation accuracy
        if info['best_accuracy'] is not None:
            print(f"\nBest validation accuracy: {info['best_accuracy']:.2f}%")
        else:
            print("\nValidation accuracy not found in checkpoint")
        
        # Show additional information if verbose mode is enabled
        if args.verbose:
            print("\nDetailed model information:")
            print(f"  Model name: {info['model_name']}")
            print(f"  Experiment: {info['experiment_name']}")
            print(f"  Epoch: {info['epoch']}")
            print(f"  Parameters: {info['num_parameters']:,}" if info['num_parameters'] else "  Parameters: Unknown")
            print(f"  File size: {info['file_size_mb']:.2f} MB")
            print(f"  Filename: {info['filename']}")
    
if __name__ == '__main__':
    main() 