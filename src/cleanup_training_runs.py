#!/usr/bin/env python3
import os
import shutil
import argparse
from pathlib import Path

def parse_args():
    parser = argparse.ArgumentParser(description='Delete training run directories without best.pth files')
    parser.add_argument('--dry-run', action='store_true', help='Only print directories to be deleted without actually deleting them')
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Path to training runs directory
    training_runs_dir = Path('/scratch/yw5954/dp_sp25_proj1/outputs/training_runs')
    
    # Get all subdirectories
    run_dirs = [d for d in training_runs_dir.iterdir() if d.is_dir()]
    print(f"Found {len(run_dirs)} training run directories")
    
    # Count directories to delete and keep
    to_delete = []
    to_keep = []
    
    # Check each directory for best.pth
    for run_dir in run_dirs:
        best_model_path = run_dir / 'best.pth'
        if best_model_path.exists():
            to_keep.append(run_dir)
        else:
            to_delete.append(run_dir)
    
    print(f"Directories with best.pth (to keep): {len(to_keep)}")
    print(f"Directories without best.pth (to delete): {len(to_delete)}")
    
    # Print directories to delete
    if to_delete:
        print("\nDirectories to be deleted:")
        for dir_path in to_delete:
            print(f"  - {dir_path}")
    
    # Delete directories if not in dry run mode
    if not args.dry_run and to_delete:
        print("\nDeleting directories...")
        for dir_path in to_delete:
            try:
                shutil.rmtree(dir_path)
                print(f"  ✓ Deleted: {dir_path}")
            except Exception as e:
                print(f"  ✗ Error deleting {dir_path}: {e}")
        
        print(f"\nDeleted {len(to_delete)} directories")
    elif args.dry_run:
        print("\nDRY RUN: No directories were deleted")
    
    print(f"\nRemaining training run directories: {len(to_keep)}")

if __name__ == "__main__":
    main() 