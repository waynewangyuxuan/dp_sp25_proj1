#!/usr/bin/env python3
import os
import torch
import argparse
from pathlib import Path
import sys
import json

# Add the project root to the Python path
sys.path.append(str(Path('/scratch/yw5954/dp_sp25_proj1')))

# Import the TrainingConfig class
from src.configs.train_config import TrainingConfig

# Add TrainingConfig to safe globals for PyTorch 2.6+ compatibility
try:
    from torch.serialization import add_safe_globals
    add_safe_globals([TrainingConfig])
except ImportError:
    pass

def parse_args():
    parser = argparse.ArgumentParser(description='Rename training run directories to include validation accuracy')
    parser.add_argument('--dry-run', action='store_true', help='Only print directories to be renamed without actually renaming them')
    parser.add_argument('--debug', action='store_true', help='Print detailed checkpoint information for debugging')
    return parser.parse_args()

def get_validation_accuracy(checkpoint_path, debug=False):
    """Extract validation accuracy from a checkpoint file."""
    try:
        # Load the checkpoint with weights_only=False to access metadata
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        
        if debug:
            print(f"\nCheckpoint keys: {list(checkpoint.keys())}")
            
            # Print all keys that might contain validation accuracy
            for key in checkpoint.keys():
                if isinstance(checkpoint[key], (int, float)) or (
                    isinstance(checkpoint[key], torch.Tensor) and checkpoint[key].numel() == 1
                ):
                    value = checkpoint[key].item() if isinstance(checkpoint[key], torch.Tensor) else checkpoint[key]
                    print(f"  {key}: {value}")
                elif key == 'config' and isinstance(checkpoint[key], dict):
                    print(f"  config keys: {list(checkpoint['config'].keys())}")
                elif key == 'metrics' and isinstance(checkpoint[key], dict):
                    print(f"  metrics: {checkpoint['metrics']}")
        
        # Extract validation accuracy - in our checkpoints, 'best_acc' is the validation accuracy
        if 'best_acc' in checkpoint:
            return checkpoint['best_acc']
        elif 'best_val_acc' in checkpoint:
            return checkpoint['best_val_acc']
        elif 'val_acc' in checkpoint:
            return checkpoint['val_acc']
        elif 'metrics' in checkpoint and isinstance(checkpoint['metrics'], dict):
            metrics = checkpoint['metrics']
            if 'val_acc' in metrics:
                return metrics['val_acc']
            elif 'validation_accuracy' in metrics:
                return metrics['validation_accuracy']
        elif 'config' in checkpoint and isinstance(checkpoint['config'], dict):
            config = checkpoint['config']
            if 'best_val_acc' in config:
                return config['best_val_acc']
        
        # Try to find any key that might contain validation accuracy
        for key in checkpoint.keys():
            if isinstance(checkpoint[key], (int, float, torch.Tensor)):
                if 'val' in key.lower() and 'acc' in key.lower():
                    if isinstance(checkpoint[key], torch.Tensor):
                        return checkpoint[key].item()
                    return checkpoint[key]
                elif 'best' in key.lower() and 'acc' in key.lower():
                    if isinstance(checkpoint[key], torch.Tensor):
                        return checkpoint[key].item()
                    return checkpoint[key]
        
        # If no validation accuracy found, return None
        return None
    except Exception as e:
        print(f"Error loading checkpoint {checkpoint_path}: {e}")
        return None

def main():
    args = parse_args()
    
    # Path to training runs directory
    training_runs_dir = Path('/scratch/yw5954/dp_sp25_proj1/outputs/training_runs')
    
    # Get all subdirectories
    run_dirs = [d for d in training_runs_dir.iterdir() if d.is_dir()]
    print(f"Found {len(run_dirs)} training run directories")
    
    # Process each directory
    renamed_count = 0
    skipped_count = 0
    
    for run_dir in run_dirs:
        # Check if best.pth exists
        best_model_path = run_dir / 'best.pth'
        if not best_model_path.exists():
            print(f"Skipping {run_dir.name}: No best.pth found")
            skipped_count += 1
            continue
        
        print(f"\nProcessing {run_dir.name}...")
        
        # Get validation accuracy
        val_acc = get_validation_accuracy(best_model_path, debug=args.debug)
        if val_acc is None:
            print(f"Skipping {run_dir.name}: Could not extract validation accuracy")
            skipped_count += 1
            continue
        
        # Format validation accuracy as a string with 2 decimal places
        val_acc_str = f"{val_acc:.2f}".replace('.', '_')
        
        # Check if the directory name already contains the validation accuracy
        if f"val{val_acc_str}" in run_dir.name:
            print(f"Skipping {run_dir.name}: Already contains validation accuracy")
            skipped_count += 1
            continue
        
        # Create new directory name
        new_name = f"{run_dir.name}_val{val_acc_str}"
        new_path = run_dir.parent / new_name
        
        # Print the rename operation
        print(f"Renaming: {run_dir.name} → {new_name}")
        
        # Rename the directory if not in dry run mode
        if not args.dry_run:
            try:
                run_dir.rename(new_path)
                renamed_count += 1
            except Exception as e:
                print(f"  ✗ Error renaming {run_dir.name}: {e}")
                skipped_count += 1
        else:
            renamed_count += 1
    
    # Print summary
    if args.dry_run:
        print(f"\nDRY RUN: {renamed_count} directories would be renamed, {skipped_count} would be skipped")
    else:
        print(f"\nRenamed {renamed_count} directories, skipped {skipped_count} directories")

if __name__ == "__main__":
    main() 