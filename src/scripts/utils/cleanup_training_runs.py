#!/usr/bin/env python3
import os
import sys
import argparse
import shutil
from pathlib import Path

# Add src directory to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

def cleanup_training_runs(output_dir, keep_best=True, dry_run=True):
    """Clean up training runs by removing checkpoints except the best ones.
    
    Args:
        output_dir: Directory containing training runs
        keep_best: Whether to keep the best model for each run
        dry_run: If True, only print what would be done without actually deleting
    """
    output_path = Path(output_dir)
    
    # Find all training run directories
    run_dirs = [d for d in output_path.iterdir() if d.is_dir() and not d.name.startswith('.')]
    
    if not run_dirs:
        print(f"No training runs found in {output_dir}")
        return
    
    print(f"Found {len(run_dirs)} training runs")
    
    total_space_saved = 0
    files_removed = 0
    
    for run_dir in run_dirs:
        # Find all checkpoint files
        checkpoint_files = list(run_dir.glob("*.pth"))
        
        if not checkpoint_files:
            print(f"No checkpoint files found in {run_dir}")
            continue
        
        # Find the best model if it exists
        best_model = None
        if keep_best:
            best_candidates = [f for f in checkpoint_files if "best" in f.name]
            if best_candidates:
                best_model = best_candidates[0]
        
        # Calculate space that would be saved
        space_to_save = 0
        files_to_remove = []
        
        for checkpoint in checkpoint_files:
            if best_model and checkpoint == best_model:
                continue
            
            size = checkpoint.stat().st_size
            space_to_save += size
            files_to_remove.append((checkpoint, size))
        
        # Print summary for this run
        print(f"\nRun: {run_dir.name}")
        print(f"  Total checkpoints: {len(checkpoint_files)}")
        print(f"  Checkpoints to remove: {len(files_to_remove)}")
        print(f"  Space to save: {space_to_save / (1024 * 1024):.2f} MB")
        
        if best_model:
            print(f"  Keeping best model: {best_model.name}")
        
        # Remove files if not a dry run
        if not dry_run and files_to_remove:
            for file_path, _ in files_to_remove:
                file_path.unlink()
                files_removed += 1
            
            print(f"  Removed {len(files_to_remove)} files")
        
        total_space_saved += space_to_save
    
    # Print overall summary
    print("\nSummary:")
    print(f"  Total space saved: {total_space_saved / (1024 * 1024):.2f} MB")
    print(f"  Total files {'that would be' if dry_run else ''} removed: {files_removed if not dry_run else sum(len(files) for _, files in [(run_dir, [f for f in run_dir.glob('*.pth') if not (keep_best and 'best' in f.name)]) for run_dir in run_dirs])}")
    
    if dry_run:
        print("\nThis was a dry run. No files were actually deleted.")
        print("Run with --no-dry-run to actually delete the files.")

def main():
    parser = argparse.ArgumentParser(description='Clean up training runs by removing checkpoints')
    parser.add_argument('--output-dir', type=str, default='outputs',
                        help='Directory containing training runs')
    parser.add_argument('--no-keep-best', action='store_true',
                        help='Do not keep the best model for each run')
    parser.add_argument('--no-dry-run', action='store_true',
                        help='Actually delete files (default is dry run)')
    args = parser.parse_args()
    
    cleanup_training_runs(
        output_dir=args.output_dir,
        keep_best=not args.no_keep_best,
        dry_run=not args.no_dry_run
    )

if __name__ == '__main__':
    main() 