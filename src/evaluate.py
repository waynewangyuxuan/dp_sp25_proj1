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

def get_evaluation_dir(config: TrainingConfig) -> str:
    """Get the directory for storing evaluation results"""
    return os.path.join(config.output_dir, "evaluations")

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
    folder_name = f"{config.experiment_name}_val_acc_{val_acc:.2f}_{timestamp}"
    
    # Create evaluation directory if it doesn't exist
    eval_dir = get_evaluation_dir(config)
    os.makedirs(eval_dir, exist_ok=True)
    
    # Create specific evaluation folder
    eval_folder = os.path.join(eval_dir, folder_name)
    os.makedirs(eval_folder, exist_ok=True)
    
    return eval_folder

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
    return model, checkpoint['best_acc']

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
        'ID': image_ids,
        'Labels': predictions
    })
    
    # Sort by ID to ensure correct order
    df = df.sort_values('ID').reset_index(drop=True)
    
    return df

def main():
    # Set device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Load config and update paths
    config = TrainingConfig()
    
    # Load best model from best_models directory
    best_model_path = os.path.join(get_best_models_dir(config), f"{config.experiment_name}_best.pth")
    model, val_acc = load_model(best_model_path, device)
    
    # Create evaluation folder
    eval_folder = create_evaluation_folder(config, best_model_path)
    
    # Create symbolic link to the model
    model_link_path = os.path.join(eval_folder, "model.pth")
    if not os.path.exists(model_link_path):
        os.symlink(best_model_path, model_link_path)
    
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
    
    # Save predictions with exact format
    output_file = os.path.join(eval_folder, "predictions.csv")
    predictions_df.to_csv(output_file, index=False)
    print(f"\nSaved predictions to {output_file}")
    
    # Print sample predictions
    print("\nSample predictions:")
    print(predictions_df.head(10).to_string())
    print(f"\nTotal predictions: {len(predictions_df)}")
    print(f"\nEvaluation results stored in: {eval_folder}")

if __name__ == '__main__':
    main() 