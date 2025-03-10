#!/usr/bin/env python3
import os
import sys
import argparse
import pandas as pd
import matplotlib.pyplot as plt
import glob
from pathlib import Path

# Add src directory to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

"""
generate_training_graphs.py

This script generates training graphs from metrics CSV files in the training logs directory.
It creates three graphs:
1. Training Loss and Validation Loss
2. Validation Loss and Validation Accuracy
3. Learning Rate

The graphs are saved in the same directory as the metrics CSV file.
"""

def parse_args():
    parser = argparse.ArgumentParser(description='Generate training graphs from metrics CSV files')
    parser.add_argument('--metrics-file', type=str, default=None,
                        help='Path to the metrics CSV file')
    parser.add_argument('--output-dir', type=str, default=None,
                        help='Directory to save the graphs (defaults to same directory as metrics file)')
    parser.add_argument('--all-runs', action='store_true',
                        help='Generate graphs for all training runs')
    parser.add_argument('--dpi', type=int, default=300,
                        help='DPI for saved images')
    parser.add_argument('--figsize', type=str, default='10,6',
                        help='Figure size in inches, format: width,height')
    parser.add_argument('--style', type=str, default='seaborn-v0_8-darkgrid',
                        help='Matplotlib style')
    return parser.parse_args()

def generate_graphs(metrics_file, output_dir=None, dpi=300, figsize=(10, 6), style='seaborn-v0_8-darkgrid'):
    """Generate training graphs from a metrics CSV file."""
    # Set matplotlib style
    plt.style.use(style)
    
    # Load metrics
    try:
        metrics = pd.read_csv(metrics_file)
    except Exception as e:
        print(f"Error loading metrics file {metrics_file}: {e}")
        return False
    
    # Check if metrics file has required columns
    required_columns = ['epoch', 'train_loss', 'val_loss', 'val_acc', 'learning_rate']
    if not all(col in metrics.columns for col in required_columns):
        print(f"Metrics file {metrics_file} does not have all required columns: {required_columns}")
        print(f"Available columns: {metrics.columns.tolist()}")
        return False
    
    # Set output directory
    if output_dir is None:
        output_dir = os.path.dirname(metrics_file)
    os.makedirs(output_dir, exist_ok=True)
    
    # Get experiment name from directory structure
    try:
        # Extract timestamp from path
        timestamp = Path(metrics_file).parent.parent.name
        # Try to find config.json to get experiment name
        config_file = os.path.join(Path(metrics_file).parent, "config.json")
        if os.path.exists(config_file):
            import json
            with open(config_file, 'r') as f:
                config = json.load(f)
                experiment_name = config.get('experiment_name', timestamp)
        else:
            experiment_name = timestamp
    except:
        experiment_name = "Training Run"
    
    # 1. Training Loss and Validation Loss
    plt.figure(figsize=figsize)
    plt.plot(metrics['epoch'], metrics['train_loss'], label='Training Loss', marker='o', linestyle='-', color='#1f77b4')
    plt.plot(metrics['epoch'], metrics['val_loss'], label='Validation Loss', marker='s', linestyle='-', color='#ff7f0e')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(f'Training and Validation Loss\n{experiment_name}')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    loss_plot_path = os.path.join(output_dir, 'loss_plot.png')
    plt.savefig(loss_plot_path, dpi=dpi)
    plt.close()
    
    # 2. Validation Loss and Validation Accuracy
    fig, ax1 = plt.subplots(figsize=figsize)
    
    # Validation Loss (left y-axis)
    color = '#1f77b4'
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Validation Loss', color=color)
    ax1.plot(metrics['epoch'], metrics['val_loss'], label='Validation Loss', marker='o', linestyle='-', color=color)
    ax1.tick_params(axis='y', labelcolor=color)
    
    # Validation Accuracy (right y-axis)
    ax2 = ax1.twinx()
    color = '#ff7f0e'
    ax2.set_ylabel('Validation Accuracy (%)', color=color)
    ax2.plot(metrics['epoch'], metrics['val_acc'], label='Validation Accuracy', marker='s', linestyle='-', color=color)
    ax2.tick_params(axis='y', labelcolor=color)
    
    # Add legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='center right')
    
    plt.title(f'Validation Loss and Accuracy\n{experiment_name}')
    plt.grid(True)
    plt.tight_layout()
    val_plot_path = os.path.join(output_dir, 'validation_plot.png')
    plt.savefig(val_plot_path, dpi=dpi)
    plt.close()
    
    # 3. Learning Rate
    plt.figure(figsize=figsize)
    plt.plot(metrics['epoch'], metrics['learning_rate'], label='Learning Rate', marker='o', linestyle='-', color='#2ca02c')
    plt.xlabel('Epoch')
    plt.ylabel('Learning Rate')
    plt.title(f'Learning Rate Schedule\n{experiment_name}')
    plt.grid(True)
    plt.tight_layout()
    lr_plot_path = os.path.join(output_dir, 'learning_rate_plot.png')
    plt.savefig(lr_plot_path, dpi=dpi)
    plt.close()
    
    # 4. Combined plot with all metrics
    plt.figure(figsize=(figsize[0], figsize[1] * 1.5))
    
    # Create 3 subplots
    plt.subplot(3, 1, 1)
    plt.plot(metrics['epoch'], metrics['train_loss'], label='Training Loss', marker='o', linestyle='-', color='#1f77b4')
    plt.plot(metrics['epoch'], metrics['val_loss'], label='Validation Loss', marker='s', linestyle='-', color='#ff7f0e')
    plt.ylabel('Loss')
    plt.title(f'Training Metrics\n{experiment_name}')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(3, 1, 2)
    plt.plot(metrics['epoch'], metrics['val_acc'], label='Validation Accuracy', marker='o', linestyle='-', color='#2ca02c')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(3, 1, 3)
    plt.plot(metrics['epoch'], metrics['learning_rate'], label='Learning Rate', marker='o', linestyle='-', color='#d62728')
    plt.xlabel('Epoch')
    plt.ylabel('Learning Rate')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    combined_plot_path = os.path.join(output_dir, 'combined_plot.png')
    plt.savefig(combined_plot_path, dpi=dpi)
    plt.close()
    
    print(f"Generated graphs for {experiment_name}:")
    print(f"  - Loss plot: {loss_plot_path}")
    print(f"  - Validation plot: {val_plot_path}")
    print(f"  - Learning rate plot: {lr_plot_path}")
    print(f"  - Combined plot: {combined_plot_path}")
    
    return True

def find_all_metrics_files(base_dir='outputs'):
    """Find all metrics.csv files in the training runs directory."""
    # Find all metrics.csv files in the training runs directory
    metrics_files = glob.glob(os.path.join(base_dir, 'training_runs', '*', 'logs', 'metrics.csv'))
    return metrics_files

def main():
    args = parse_args()
    
    # Parse figsize
    figsize = tuple(map(float, args.figsize.split(',')))
    
    if args.all_runs:
        # Find all metrics files
        metrics_files = find_all_metrics_files()
        if not metrics_files:
            print("No metrics files found in outputs/training_runs/*/logs/")
            return
        
        print(f"Found {len(metrics_files)} metrics files")
        success_count = 0
        
        for metrics_file in metrics_files:
            print(f"\nProcessing {metrics_file}...")
            if generate_graphs(metrics_file, args.output_dir, args.dpi, figsize, args.style):
                success_count += 1
        
        print(f"\nGenerated graphs for {success_count} out of {len(metrics_files)} metrics files")
    
    elif args.metrics_file:
        # Generate graphs for a single metrics file
        if not os.path.exists(args.metrics_file):
            print(f"Metrics file {args.metrics_file} does not exist")
            return
        
        generate_graphs(args.metrics_file, args.output_dir, args.dpi, figsize, args.style)
    
    else:
        # Try to find the most recent metrics file
        metrics_files = find_all_metrics_files()
        if not metrics_files:
            print("No metrics files found and no metrics file specified")
            print("Please specify a metrics file with --metrics-file or use --all-runs to process all metrics files")
            return
        
        # Sort by modification time (most recent first)
        metrics_files.sort(key=os.path.getmtime, reverse=True)
        most_recent = metrics_files[0]
        print(f"Using most recent metrics file: {most_recent}")
        generate_graphs(most_recent, args.output_dir, args.dpi, figsize, args.style)

if __name__ == '__main__':
    main() 