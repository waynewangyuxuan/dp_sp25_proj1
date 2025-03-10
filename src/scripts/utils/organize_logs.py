#!/usr/bin/env python
"""
Script to organize existing training runs by creating log directories and generating log files.
This is a one-time script to retroactively add logging to existing training runs.
"""

import os
import sys
import json
import torch
import argparse
from datetime import datetime
from pathlib import Path
import numpy as np
import re
import shutil

# Add src directory to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# Import the TrainingConfig class and model factory
from src.configs.train_config import TrainingConfig
from src.models.model_factory import ModelFactory

# Add TrainingConfig to safe globals for PyTorch 2.6+ compatibility
try:
    from torch.serialization import add_safe_globals
    add_safe_globals([TrainingConfig])
except ImportError:
    pass

def parse_args():
    parser = argparse.ArgumentParser(description='Organize training run logs')
    parser.add_argument('--output-dir', type=str, default='outputs',
                        help='Base output directory containing training runs')
    parser.add_argument('--force-weights-only-false', action='store_true',
                        help='Force weights_only=False when loading checkpoints')
    parser.add_argument('--detailed', action='store_true',
                        help='Generate more detailed logs with model parameters')
    parser.add_argument('--force', action='store_true',
                        help='Force overwrite of existing log directories')
    parser.add_argument('--logs-dir', type=str, default='logs',
                        help='Directory containing log files')
    parser.add_argument('--no-dry-run', action='store_true',
                        help='Actually move files (default is dry run)')
    return parser.parse_args()

def create_log_directory(run_dir):
    """Create a logs directory for a training run if it doesn't exist"""
    logs_dir = os.path.join(run_dir, "logs")
    os.makedirs(logs_dir, exist_ok=True)
    return logs_dir

def count_parameters(model):
    """Count total and trainable parameters in a model"""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_params, trainable_params

def get_model_info(checkpoint, config):
    """Extract model information from checkpoint and config"""
    model_info = {
        'model_name': getattr(config, 'model_name', 'unknown'),
        'total_params': None,
        'trainable_params': None,
        'architecture': None
    }
    
    # Try to recreate the model to get parameter counts
    try:
        print(f"  Attempting to recreate model: {getattr(config, 'model_name', 'unknown')}")
        
        # For CIFAR-10, we know the number of classes is 10
        if not hasattr(config, 'num_classes'):
            print(f"  Adding num_classes=10 to config (CIFAR-10)")
            setattr(config, 'num_classes', 10)
        
        if hasattr(config, 'model_name') and hasattr(config, 'num_classes'):
            print(f"  Creating model with name={config.model_name}, num_classes={config.num_classes}")
            model = ModelFactory.create(config.model_name, num_classes=config.num_classes)
            
            # Load state dict to ensure correct architecture
            if 'model_state_dict' in checkpoint:
                print(f"  Loading model state dict")
                model.load_state_dict(checkpoint['model_state_dict'])
            else:
                print(f"  Warning: No model_state_dict found in checkpoint")
            
            # Count parameters
            total_params, trainable_params = count_parameters(model)
            model_info['total_params'] = total_params
            model_info['trainable_params'] = trainable_params
            print(f"  Counted parameters: total={total_params:,}, trainable={trainable_params:,}")
            
            # Get architecture summary
            model_info['architecture'] = str(model)
        else:
            missing_attrs = []
            if not hasattr(config, 'model_name'):
                missing_attrs.append('model_name')
            if not hasattr(config, 'num_classes'):
                missing_attrs.append('num_classes')
            print(f"  Warning: Missing required attributes: {', '.join(missing_attrs)}")
    except Exception as e:
        print(f"  Warning: Could not recreate model for parameter counting: {e}")
        import traceback
        traceback.print_exc()
    
    return model_info

def extract_optimizer_info(checkpoint):
    """Extract optimizer information from checkpoint"""
    optimizer_info = {}
    
    if 'optimizer_state_dict' in checkpoint:
        try:
            opt_state = checkpoint['optimizer_state_dict']
            
            # Extract optimizer type
            if 'name' in opt_state:
                optimizer_info['type'] = opt_state['name']
            elif 'defaults' in opt_state:
                # Try to infer from parameters
                defaults = opt_state['defaults']
                optimizer_info['defaults'] = defaults
                
                # Common optimizer parameters
                if 'lr' in defaults:
                    optimizer_info['learning_rate'] = defaults['lr']
                if 'weight_decay' in defaults:
                    optimizer_info['weight_decay'] = defaults['weight_decay']
                if 'momentum' in defaults:
                    optimizer_info['momentum'] = defaults['momentum']
        except Exception as e:
            print(f"  Warning: Could not extract optimizer info: {e}")
    
    return optimizer_info

def extract_run_info(run_dir, force_weights_only_false=False, detailed=False):
    """Extract information from a training run directory"""
    # Get timestamp from directory name
    run_name = os.path.basename(run_dir)
    
    # Check for best.pth file
    best_path = os.path.join(run_dir, "best.pth")
    checkpoint_info = None
    
    if os.path.exists(best_path):
        try:
            # Try loading with weights_only=False if specified or as a fallback
            try:
                if force_weights_only_false:
                    checkpoint = torch.load(best_path, map_location='cpu', weights_only=False)
                else:
                    checkpoint = torch.load(best_path, map_location='cpu')
            except Exception as e:
                if "Weights only load failed" in str(e) and not force_weights_only_false:
                    print(f"Retrying with weights_only=False for {best_path}")
                    checkpoint = torch.load(best_path, map_location='cpu', weights_only=False)
                else:
                    raise e
            
            # Extract basic checkpoint info
            checkpoint_info = {
                'epoch': checkpoint.get('epoch', None),
                'best_acc': checkpoint.get('best_acc', None),
                'config': checkpoint.get('config', None),
                'train_steps': checkpoint.get('train_steps', None)
            }
            
            # Extract additional info if detailed mode is enabled
            if detailed and checkpoint_info['config']:
                # Get model information
                model_info = get_model_info(checkpoint, checkpoint_info['config'])
                checkpoint_info['model_info'] = model_info
                
                # Get optimizer information
                optimizer_info = extract_optimizer_info(checkpoint)
                checkpoint_info['optimizer_info'] = optimizer_info
                
                # Get scheduler information if available
                if 'scheduler_state_dict' in checkpoint:
                    checkpoint_info['has_scheduler'] = True
            
            print(f"Successfully loaded checkpoint from {best_path}")
            
            # Print some info for verification
            if checkpoint_info['best_acc'] is not None:
                print(f"  Best accuracy: {checkpoint_info['best_acc']:.2f}%")
            if checkpoint_info['epoch'] is not None:
                print(f"  Epoch: {checkpoint_info['epoch']}")
            if detailed and 'model_info' in checkpoint_info and checkpoint_info['model_info']['total_params']:
                print(f"  Model parameters: {checkpoint_info['model_info']['total_params']:,}\n")
                
        except Exception as e:
            print(f"Error loading checkpoint from {best_path}: {e}")
    
    # Get list of checkpoint files
    checkpoints_dir = os.path.join(run_dir, "checkpoints")
    checkpoint_files = []
    
    if os.path.exists(checkpoints_dir):
        checkpoint_files = [f for f in os.listdir(checkpoints_dir) if f.endswith('.pth')]
    
    return {
        'run_name': run_name,
        'checkpoint_info': checkpoint_info,
        'checkpoint_files': checkpoint_files,
        'has_best': os.path.exists(best_path)
    }

def generate_log_files(run_dir, run_info, detailed=False):
    """Generate log files for a training run"""
    logs_dir = create_log_directory(run_dir)
    
    # Create training.log file
    log_file = os.path.join(logs_dir, "training.log")
    with open(log_file, 'w') as f:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        f.write(f"[{timestamp}] Log file created retroactively\n")
        
        if run_info['checkpoint_info']:
            checkpoint_info = run_info['checkpoint_info']
            if checkpoint_info['epoch'] is not None:
                f.write(f"[{timestamp}] Training completed at epoch {checkpoint_info['epoch']}\n")
            
            if checkpoint_info['best_acc'] is not None:
                f.write(f"[{timestamp}] Best validation accuracy: {checkpoint_info['best_acc']:.2f}%\n")
            
            if checkpoint_info['train_steps'] is not None:
                f.write(f"[{timestamp}] Total training steps: {checkpoint_info['train_steps']}\n")
            
            if checkpoint_info['config']:
                config = checkpoint_info['config']
                model_name = getattr(config, 'model_name', 'unknown')
                f.write(f"[{timestamp}] Model: {model_name}\n")
                
                # Add more config details
                batch_size = getattr(config, 'batch_size', 'unknown')
                f.write(f"[{timestamp}] Batch size: {batch_size}\n")
                
                lr = getattr(config, 'learning_rate', 'unknown')
                f.write(f"[{timestamp}] Learning rate: {lr}\n")
                
                weight_decay = getattr(config, 'weight_decay', 'unknown')
                f.write(f"[{timestamp}] Weight decay: {weight_decay}\n")
                
                # Add detailed model information if available
                if detailed and 'model_info' in checkpoint_info:
                    model_info = checkpoint_info['model_info']
                    if model_info['total_params']:
                        f.write(f"[{timestamp}] Total parameters: {model_info['total_params']:,}\n")
                    if model_info['trainable_params']:
                        f.write(f"[{timestamp}] Trainable parameters: {model_info['trainable_params']:,}\n")
                        non_trainable = model_info['total_params'] - model_info['trainable_params']
                        f.write(f"[{timestamp}] Non-trainable parameters: {non_trainable:,}\n")
                
                # Add optimizer information if available
                if detailed and 'optimizer_info' in checkpoint_info:
                    optimizer_info = checkpoint_info['optimizer_info']
                    if 'type' in optimizer_info:
                        f.write(f"[{timestamp}] Optimizer: {optimizer_info['type']}\n")
                    if 'learning_rate' in optimizer_info:
                        f.write(f"[{timestamp}] Optimizer learning rate: {optimizer_info['learning_rate']}\n")
                    if 'weight_decay' in optimizer_info:
                        f.write(f"[{timestamp}] Optimizer weight decay: {optimizer_info['weight_decay']}\n")
                    if 'momentum' in optimizer_info:
                        f.write(f"[{timestamp}] Optimizer momentum: {optimizer_info['momentum']}\n")
        
        f.write(f"[{timestamp}] Checkpoint files: {len(run_info['checkpoint_files'])}\n")
        if run_info['has_best']:
            f.write(f"[{timestamp}] Best model saved: Yes\n")
    
    # Create config.json file if we have config information
    if run_info['checkpoint_info'] and run_info['checkpoint_info']['config']:
        config = run_info['checkpoint_info']['config']
        config_file = os.path.join(logs_dir, "config.json")
        
        # Convert config to dict
        config_dict = {}
        for key, value in vars(config).items():
            if not key.startswith('_'):  # Skip private attributes
                # Handle non-serializable types
                if isinstance(value, (int, float, str, bool, list, dict, tuple)) or value is None:
                    config_dict[key] = value
                else:
                    config_dict[key] = str(value)
        
        # Save to file
        with open(config_file, 'w') as f:
            json.dump(config_dict, f, indent=4)
    
    # Create metrics.csv file with placeholder
    metrics_file = os.path.join(logs_dir, "metrics.csv")
    if not os.path.exists(metrics_file):
        with open(metrics_file, 'w') as f:
            f.write("epoch,train_loss,train_acc,val_loss,val_acc,lr\n")
            
            # If we have final epoch and best accuracy, add a row
            if (run_info['checkpoint_info'] and 
                run_info['checkpoint_info']['epoch'] is not None and 
                run_info['checkpoint_info']['best_acc'] is not None):
                
                epoch = run_info['checkpoint_info']['epoch']
                best_acc = run_info['checkpoint_info']['best_acc']
                f.write(f"{epoch},N/A,N/A,N/A,{best_acc},N/A\n")
    
    # Create model_architecture.txt file if detailed mode is enabled
    if detailed and run_info['checkpoint_info'] and 'model_info' in run_info['checkpoint_info']:
        model_info = run_info['checkpoint_info']['model_info']
        if model_info['architecture']:
            arch_file = os.path.join(logs_dir, "model_architecture.txt")
            with open(arch_file, 'w') as f:
                f.write(f"Model: {model_info['model_name']}\n")
                f.write(f"Total parameters: {model_info['total_params']:,}\n")
                f.write(f"Trainable parameters: {model_info['trainable_params']:,}\n")
                f.write("\nArchitecture:\n")
                f.write(model_info['architecture'])
    
    print(f"Created log files in {logs_dir}")

def organize_training_runs(output_dir, force_weights_only_false=False, detailed=False, force=False):
    """Organize all training runs in the output directory"""
    training_runs_dir = os.path.join(output_dir, "training_runs")
    
    if not os.path.exists(training_runs_dir):
        print(f"Training runs directory not found: {training_runs_dir}")
        return
    
    # Get all training run directories
    run_dirs = [os.path.join(training_runs_dir, d) for d in os.listdir(training_runs_dir)
                if os.path.isdir(os.path.join(training_runs_dir, d))]
    
    print(f"Found {len(run_dirs)} training runs")
    
    # Process each run directory
    for run_dir in run_dirs:
        print(f"\nProcessing {run_dir}...")
        
        # Check if logs directory already exists
        logs_dir = os.path.join(run_dir, "logs")
        if os.path.exists(logs_dir):
            if force:
                print(f"Overwriting existing logs directory: {logs_dir}")
                import shutil
                shutil.rmtree(logs_dir)
            else:
                print(f"Logs directory already exists: {logs_dir}")
                print("Use --force to overwrite existing logs")
                continue
        
        # Extract run information
        run_info = extract_run_info(run_dir, force_weights_only_false, detailed)
        
        # Generate log files
        generate_log_files(run_dir, run_info, detailed)

def organize_logs(logs_dir, output_dir, dry_run=True):
    """Organize log files into a structured directory hierarchy.
    
    Args:
        logs_dir: Directory containing log files
        output_dir: Directory to store organized logs
        dry_run: If True, only print what would be done without actually moving files
    """
    logs_path = Path(logs_dir)
    output_path = Path(output_dir)
    
    # Create output directory if it doesn't exist
    if not dry_run:
        output_path.mkdir(parents=True, exist_ok=True)
    
    # Find all log files
    log_files = list(logs_path.glob("*.log"))
    
    if not log_files:
        print(f"No log files found in {logs_dir}")
        return
    
    print(f"Found {len(log_files)} log files")
    
    # Process each log file
    organized_count = 0
    for log_file in log_files:
        # Extract information from filename
        model_name = "unknown"
        timestamp = None
        
        # Try to extract model name and timestamp from filename
        filename = log_file.name
        
        # Extract model name
        model_match = re.search(r'^([a-zA-Z0-9_]+)_', filename)
        if model_match:
            model_name = model_match.group(1)
        
        # Extract timestamp
        timestamp_match = re.search(r'(\d{4}[-_]\d{2}[-_]\d{2}[-_]\d{2}[-_]\d{2}[-_]\d{2})', filename)
        if timestamp_match:
            timestamp_str = timestamp_match.group(1)
            try:
                # Try different timestamp formats
                formats = [
                    "%Y-%m-%d_%H-%M-%S",
                    "%Y%m%d_%H%M%S",
                    "%Y-%m-%d-%H-%M-%S"
                ]
                
                for fmt in formats:
                    try:
                        timestamp = datetime.strptime(timestamp_str, fmt)
                        break
                    except ValueError:
                        continue
            except Exception as e:
                print(f"Error parsing timestamp in {filename}: {e}")
        
        # If no timestamp found, use file modification time
        if timestamp is None:
            timestamp = datetime.fromtimestamp(log_file.stat().st_mtime)
        
        # Create directory structure
        year_month = timestamp.strftime("%Y-%m")
        model_dir = output_path / model_name / year_month
        
        # Create target path
        target_path = model_dir / log_file.name
        
        # Print the move operation
        print(f"{'Would move' if dry_run else 'Moving'} {log_file} -> {target_path}")
        
        # Move the file if not in dry run mode
        if not dry_run:
            try:
                # Create directory if it doesn't exist
                model_dir.mkdir(parents=True, exist_ok=True)
                
                # Copy the file
                shutil.copy2(log_file, target_path)
                
                # Remove the original file
                log_file.unlink()
                
                organized_count += 1
            except Exception as e:
                print(f"Error moving {log_file}: {e}")
        else:
            organized_count += 1
    
    # Print summary
    print(f"\n{'Would organize' if dry_run else 'Organized'} {organized_count} log files")
    
    if dry_run:
        print("\nThis was a dry run. No files were actually moved.")
        print("Run with --no-dry-run to actually move the files.")

def main():
    args = parse_args()
    organize_training_runs(args.output_dir, args.force_weights_only_false, args.detailed, args.force)
    print("\nLog organization complete!")

    organize_logs(
        logs_dir=args.logs_dir,
        output_dir=args.output_dir,
        dry_run=not args.no_dry_run
    )

if __name__ == '__main__':
    main() 