import os
import sys
import torch
import pandas as pd
from tqdm import tqdm
from datetime import datetime
from torch.serialization import add_safe_globals

# Add src directory to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models import ModelFactory
from data.data_module import CIFAR10DataModule
from configs.train_config import TrainingConfig

# Add TrainingConfig to safe globals
add_safe_globals([TrainingConfig])

def get_run_dir(config: TrainingConfig) -> str:
    """Get the directory for the current run based on timestamp"""
    timestamp = datetime.now().strftime("%Y_%m_%d_%H_%M")
    return os.path.join(config.output_dir, "training_runs", timestamp)

def get_best_models_dir(config: TrainingConfig) -> str:
    """Get the directory for storing best performing models"""
    return os.path.join(config.output_dir, "best_models")

def setup_output_dirs(config: TrainingConfig) -> tuple:
    """Setup output directories for the current run"""
    # Create main directories
    run_dir = get_run_dir(config)
    best_models_dir = get_best_models_dir(config)
    checkpoints_dir = os.path.join(run_dir, "checkpoints")
    
    # Create all directories
    os.makedirs(run_dir, exist_ok=True)
    os.makedirs(best_models_dir, exist_ok=True)
    os.makedirs(checkpoints_dir, exist_ok=True)
    
    return run_dir, best_models_dir, checkpoints_dir

def load_model(checkpoint_path: str, device: str = 'cuda'):
    """Load trained model from checkpoint"""
    # Load checkpoint with weights_only=False since we need the full state
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = checkpoint['config']
    
    # Create model
    model = ModelFactory.create(config.model_name)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    print(f"\nLoaded model from {checkpoint_path}")
    print(f"Best validation accuracy: {checkpoint['best_acc']:.2f}%")
    return model

@torch.no_grad()
def generate_predictions(model: torch.nn.Module,
                       data_loader: torch.utils.data.DataLoader,
                       device: str = 'cuda') -> pd.DataFrame:
    """Generate predictions for test set"""
    predictions = []
    image_ids = []
    
    # Create progress bar
    pbar = tqdm(data_loader, desc="Generating predictions", unit="batch")
    
    for images, ids in pbar:
        # Move to device
        images = images.to(device)
        
        # Forward pass
        outputs = model(images)
        _, predicted = outputs.max(1)
        
        # Store predictions
        predictions.extend(predicted.cpu().numpy())
        image_ids.extend(ids.numpy())
    
    # Create DataFrame with exact column names
    df = pd.DataFrame({
        '"ID"': image_ids,
        '"Labels"': predictions
    })
    
    # Sort by ID to ensure correct order
    df = df.sort_values('"ID"')
    
    return df

def main():
    # Set device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Load config and update paths
    config = TrainingConfig()
    
    # Setup output directories
    run_dir, best_models_dir, _ = setup_output_dirs(config)
    
    # Load best model from best_models directory
    best_model_path = os.path.join(best_models_dir, f"{config.experiment_name}_best.pth")
    model = load_model(best_model_path, device)
    
    # Create data module and setup
    data_module = CIFAR10DataModule(
        data_dir=config.data_dir,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory
    )
    data_module.setup()
    
    # Generate predictions
    print("\nGenerating predictions for test set...")
    predictions_df = generate_predictions(
        model=model,
        data_loader=data_module.test_dataloader(),
        device=device
    )
    
    # Save predictions in the run directory with quoted column names
    output_file = os.path.join(run_dir, "predictions.csv")
    predictions_df.to_csv(output_file, index=False, quoting=1)  # QUOTE_ALL mode
    print(f"\nSaved predictions to {output_file}")
    
    # Print sample predictions
    print("\nSample predictions:")
    print(predictions_df.head(10))
    print(f"\nTotal predictions: {len(predictions_df)}")

if __name__ == '__main__':
    main() 