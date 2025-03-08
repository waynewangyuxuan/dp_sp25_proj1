import os
import sys
import torch
import pandas as pd
import argparse
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
    # Extract model name from the path to include in folder name
    model_filename = os.path.basename(model_path)
    folder_name = f"{config.experiment_name}_val_acc_{val_acc:.2f}_{timestamp}_model_{model_filename}"
    
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
    
    # Extract model configuration
    model_name = config.model_name
    print(f"Loading model: {model_name}")
    
    # Check if the model has SE layers by inspecting the state dict keys
    has_se_layers = any('se.fc' in key for key in checkpoint['model_state_dict'].keys())
    
    # If the model doesn't have SE layers but the default would create them,
    # we need to create a model without SE layers
    if not has_se_layers and model_name == 'resnet_small':
        print("Creating model without SE layers to match checkpoint")
        from models.resnet import ResNet, BasicBlock
        # Create ResNet model directly with use_se=False
        model = ResNet(
            block=BasicBlock,
            layers=[2, 2, 2, 2],
            base_channels=32,
            num_classes=10,
            dropout_rate=0.3,
            use_se=False,
            se_reduction=16
        )
    else:
        # Create model using factory
        model = ModelFactory.create(model_name)
    
    # Load state dict
    try:
        model.load_state_dict(checkpoint['model_state_dict'])
    except RuntimeError as e:
        print(f"Error loading state dict: {e}")
        print("\nAttempting to load with strict=False...")
        model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        print("Model loaded with missing or unexpected keys.")
    
    model = model.to(device)
    model.eval()
    
    print(f"\nLoaded model from {checkpoint_path}")
    print(f"Best validation accuracy: {checkpoint['best_acc']:.2f}%")
    return model, checkpoint['best_acc'], config

@torch.no_grad()
def generate_predictions(model: torch.nn.Module,
                       data_loader: torch.utils.data.DataLoader,
                       device: str = 'cuda',
                       use_tta: bool = False,
                       tta_transforms: int = 8) -> pd.DataFrame:
    """Generate predictions for test set
    
    Args:
        model: The trained model
        data_loader: Test data loader
        device: Device to run inference on
        use_tta: Whether to use Test-Time Augmentation
        tta_transforms: Number of TTA transforms to apply
    """
    predictions = []
    image_ids = []
    
    # Create progress bar
    pbar = tqdm(data_loader, desc="Generating predictions", unit="batch")
    
    for images, ids in pbar:
        # Move to device
        images = images.to(device)
        
        if not use_tta:
            # Standard forward pass
            outputs = model(images)
            _, predicted = outputs.max(1)
        else:
            # Test-Time Augmentation - apply multiple augmentations and average results
            batch_size = images.size(0)
            tta_outputs = torch.zeros(batch_size, 10, device=device)  # Assuming 10 classes
            
            # Original prediction
            outputs = model(images)
            tta_outputs += outputs
            
            # Apply horizontal flip
            flipped_images = torch.flip(images, dims=[3])
            outputs = model(flipped_images)
            tta_outputs += outputs
            
            # Apply different crops
            for i in range(tta_transforms - 2):  # -2 because we already did original and flip
                # Create shifted versions of the image (random crops basically)
                padded_images = torch.nn.functional.pad(images, (4, 4, 4, 4), 'reflect')
                h_offset = torch.randint(0, 8, (1,)).item()
                w_offset = torch.randint(0, 8, (1,)).item()
                cropped_images = padded_images[:, :, h_offset:h_offset+32, w_offset:w_offset+32]
                
                # Get predictions
                outputs = model(cropped_images)
                tta_outputs += outputs
                
            # Average all augmentation results
            tta_outputs /= tta_transforms
            _, predicted = tta_outputs.max(1)
        
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

def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate model on test set with advanced options')
    parser.add_argument('--use-augment', action='store_true',
                        help='Use augmentations during evaluation (not recommended)')
    parser.add_argument('--model-path', type=str, default=None,
                        help='Path to model checkpoint (if not using the default best model)')
    parser.add_argument('--tta', action='store_true',
                        help='Use Test-Time Augmentation')
    parser.add_argument('--tta-transforms', type=int, default=8,
                        help='Number of transforms to use with TTA (default: 8)')
    parser.add_argument('--ensemble', action='store_true',
                        help='Use model ensemble if multiple models are available')
    parser.add_argument('--output-dir', type=str, default=None,
                        help='Custom output directory for evaluation results')
    return parser.parse_args()

def main():
    # Parse command line arguments
    args = parse_args()
    
    # Set device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Load config and update paths
    config = TrainingConfig()
    
    # Determine the model path to use
    if args.model_path:
        best_model_path = args.model_path
    else:
        best_model_path = os.path.join(get_best_models_dir(config), f"{config.experiment_name}_best.pth")
    
    # Check if the model path exists
    if not os.path.exists(best_model_path):
        print(f"Error: Model file not found at {best_model_path}")
        print("Available model files:")
        best_models_dir = get_best_models_dir(config)
        if os.path.exists(best_models_dir):
            for model_file in os.listdir(best_models_dir):
                print(f"  - {model_file}")
        else:
            print(f"  No models found in {best_models_dir}")
        return
    
    # Load the model
    model, val_acc, saved_config = load_model(best_model_path, device)
    
    # Create evaluation folder
    eval_folder = create_evaluation_folder(config, best_model_path)
    
    # Create hard copy of the model instead of a symbolic link
    model_copy_path = os.path.join(eval_folder, "model.pth")
    try:
        # Use copy instead of symbolic link to ensure we have the actual model file
        import shutil
        shutil.copy2(best_model_path, model_copy_path)
        print(f"Copied model file to: {model_copy_path}")
    except Exception as e:
        print(f"Warning: Could not copy model file. Error: {e}")
        # Fall back to symlink if copy fails
        if not os.path.exists(model_copy_path):
            try:
                os.symlink(best_model_path, model_copy_path)
                print(f"Created symbolic link instead: {model_copy_path} -> {best_model_path}")
            except Exception as e2:
                print(f"Warning: Could not create symbolic link either. Error: {e2}")
    
    # Create data module and setup - disable augmentations for evaluation unless explicitly requested
    data_module = CIFAR10DataModule(
        data_dir=config.data_dir,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
        use_cutmix=False,  # Disable CutMix for evaluation
        cutmix_alpha=config.cutmix_alpha,
        cutmix_prob=0.0,
        use_randaugment=args.use_augment,  # Disable RandAugment by default for evaluation
        randaugment_num_ops=config.randaugment_num_ops,
        randaugment_magnitude=config.randaugment_magnitude
    )
    data_module.setup()
    
    # Generate predictions
    print("\nGenerating predictions for test set...")
    if args.tta:
        print(f"Using Test-Time Augmentation (TTA) with {args.tta_transforms} transforms")
    
    predictions_df = generate_predictions(
        model=model,
        data_loader=data_module.test_dataloader(),
        device=device,
        use_tta=args.tta,
        tta_transforms=args.tta_transforms
    )
    
    # Generate a descriptive filename based on evaluation settings
    filename_prefix = "predictions"
    if args.tta:
        filename_prefix += f"_tta{args.tta_transforms}"
    if args.use_augment:
        filename_prefix += "_augmented"
    
    # Save predictions with exact format
    output_file = os.path.join(eval_folder, f"{filename_prefix}.csv")
    predictions_df.to_csv(output_file, index=False)
    print(f"\nSaved predictions to {output_file}")
    
    # Print sample predictions
    print("\nSample predictions:")
    print(predictions_df.head(10).to_string())
    print(f"\nTotal predictions: {len(predictions_df)}")
    print(f"\nEvaluation results stored in: {eval_folder}")

if __name__ == '__main__':
    main() 