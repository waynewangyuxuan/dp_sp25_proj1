#!/usr/bin/env python3
import os
import torch
import argparse
from pathlib import Path
import sys
import json
import re
from datetime import datetime

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

def parse_timestamp(timestamp_str):
    """Parse timestamp string into datetime object."""
    try:
        # Try different timestamp formats
        formats = [
            "%Y-%m-%d_%H-%M-%S",
            "%Y%m%d_%H%M%S",
            "%Y-%m-%d-%H-%M-%S"
        ]
        
        for fmt in formats:
            try:
                return datetime.strptime(timestamp_str, fmt)
            except ValueError:
                continue
        
        # If none of the formats match, raise an error
        raise ValueError(f"Could not parse timestamp: {timestamp_str}")
    except Exception as e:
        print(f"Error parsing timestamp {timestamp_str}: {e}")
        return None

def extract_run_info(run_dir):
    """Extract information from a training run directory."""
    # Initialize with default values
    info = {
        "timestamp": None,
        "model_name": "unknown",
        "accuracy": None,
        "epoch": None,
        "original_name": run_dir.name
    }
    
    # Try to parse timestamp from directory name
    timestamp_match = re.search(r'(\d{4}[-_]\d{2}[-_]\d{2}[-_]\d{2}[-_]\d{2}[-_]\d{2})', run_dir.name)
    if timestamp_match:
        timestamp_str = timestamp_match.group(1)
        info["timestamp"] = parse_timestamp(timestamp_str)
    
    # Look for config.json
    config_path = run_dir / "config.json"
    if config_path.exists():
        try:
            with open(config_path, 'r') as f:
                config = json.load(f)
                info["model_name"] = config.get("model_name", "unknown")
        except Exception as e:
            print(f"Error reading config.json in {run_dir}: {e}")
    
    # Look for best model checkpoint
    best_model_paths = list(run_dir.glob("*best*.pth"))
    if best_model_paths:
        best_model_path = best_model_paths[0]
        
        # Try to extract accuracy from filename
        acc_match = re.search(r'acc([-_])(\d+\.\d+)', best_model_path.name)
        if acc_match:
            try:
                info["accuracy"] = float(acc_match.group(2))
            except ValueError:
                pass
        
        # Try to extract epoch from filename
        epoch_match = re.search(r'epoch([-_])(\d+)', best_model_path.name)
        if epoch_match:
            try:
                info["epoch"] = int(epoch_match.group(2))
            except ValueError:
                pass
    
    return info

def generate_new_name(info):
    """Generate a new name for the training run based on extracted information."""
    parts = []
    
    # Add model name
    parts.append(info["model_name"])
    
    # Add accuracy if available
    if info["accuracy"] is not None:
        parts.append(f"acc{info['accuracy']:.2f}")
    
    # Add epoch if available
    if info["epoch"] is not None:
        parts.append(f"epoch{info['epoch']}")
    
    # Add timestamp if available
    if info["timestamp"] is not None:
        timestamp_str = info["timestamp"].strftime("%Y%m%d_%H%M%S")
        parts.append(timestamp_str)
    
    # Join parts with underscores
    new_name = "_".join(parts)
    
    # If new name is empty or same as original, use original
    if not new_name or new_name == info["original_name"]:
        return info["original_name"]
    
    return new_name

def rename_training_runs(output_dir, dry_run=True):
    """Rename training run directories based on their contents.
    
    Args:
        output_dir: Directory containing training runs
        dry_run: If True, only print what would be done without actually renaming
    """
    output_path = Path(output_dir)
    
    # Find all training run directories
    run_dirs = [d for d in output_path.iterdir() if d.is_dir() and not d.name.startswith('.')]
    
    if not run_dirs:
        print(f"No training runs found in {output_dir}")
        return
    
    print(f"Found {len(run_dirs)} training runs")
    
    # Process each directory
    renamed_count = 0
    for run_dir in run_dirs:
        # Extract information
        info = extract_run_info(run_dir)
        
        # Generate new name
        new_name = generate_new_name(info)
        
        # Skip if name is the same
        if new_name == run_dir.name:
            print(f"Skipping {run_dir.name} (name would not change)")
            continue
        
        # Create new path
        new_path = run_dir.parent / new_name
        
        # Check if destination already exists
        if new_path.exists():
            print(f"Cannot rename {run_dir.name} to {new_name} (destination already exists)")
            continue
        
        # Rename directory
        print(f"{'Would rename' if dry_run else 'Renaming'} {run_dir.name} -> {new_name}")
        
        if not dry_run:
            try:
                run_dir.rename(new_path)
                renamed_count += 1
            except Exception as e:
                print(f"Error renaming {run_dir}: {e}")
    
    # Print summary
    print(f"\n{'Would rename' if dry_run else 'Renamed'} {renamed_count} directories")
    
    if dry_run:
        print("\nThis was a dry run. No directories were actually renamed.")
        print("Run with --no-dry-run to actually rename the directories.")

def main():
    parser = argparse.ArgumentParser(description='Rename training run directories based on their contents')
    parser.add_argument('--output-dir', type=str, default='outputs',
                        help='Directory containing training runs')
    parser.add_argument('--no-dry-run', action='store_true',
                        help='Actually rename directories (default is dry run)')
    args = parser.parse_args()
    
    rename_training_runs(
        output_dir=args.output_dir,
        dry_run=not args.no_dry_run
    )

if __name__ == '__main__':
    main() 