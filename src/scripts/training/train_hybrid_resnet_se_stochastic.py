import os
import sys
import torch
import numpy as np
import random
from datetime import datetime
from tqdm import tqdm

# Add src directory to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from models.hybrid_model_factory import HybridModelFactory
from data.data_module import CIFAR10DataModule
from training.trainer import Trainer
from configs.hybrid_config import HybridResNetConfig

"""
train_hybrid_resnet_se_stochastic.py

This script trains a Hybrid ResNet model with Squeeze-and-Excitation (SE) blocks and 
stochastic depth on the CIFAR-10 dataset. The hybrid model combines SE blocks for 
channel attention and stochastic depth for regularization.
"""

def set_seed(seed=42):
    """Set random seeds for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def main():
    # Set random seeds for reproducibility
    set_seed(42)
    
    # Create config
    config = HybridResNetConfig()
    
    # Add timestamp to experiment name for unique identification
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    config.experiment_name = f"{config.experiment_name}_{timestamp}"
    
    # Enable SE and stochastic depth for hybrid model
    config.use_se = True
    config.use_stochastic_depth = True
    config.stochastic_depth_prob = 0.2
    
    print(f"\n{'='*80}")
    print(f"Starting Hybrid ResNet with SE and Stochastic Depth training: {config.experiment_name}")
    print(f"Model: {config.model_name} with SE={config.use_se}, Stochastic Depth={config.use_stochastic_depth} (prob={config.stochastic_depth_prob})")
    print(f"{'='*80}\n")
    
    # Set device
    if torch.cuda.is_available():
        config.device = 'cuda'
        torch.backends.cudnn.benchmark = True
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        config.device = 'cpu'
        print("WARNING: Using CPU for training")
    
    # Create data module
    data_module = CIFAR10DataModule(
        data_dir=config.data_dir,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
        use_cutmix=config.use_cutmix,
        cutmix_alpha=config.cutmix_alpha,
        cutmix_prob=config.cutmix_prob
    )
    data_module.setup()
    
    # Print dataset information
    print(f"\nDataset Information:")
    print(f"Training samples: {len(data_module.train_dataset)}")
    print(f"Validation samples: {len(data_module.val_dataset)}")
    print(f"Test samples: {len(data_module.test_dataset)}")
    print(f"Batch size: {config.batch_size}")
    print(f"Steps per epoch: {len(data_module.train_dataloader())}")
    
    # Create model
    model = HybridModelFactory.create(
        config.model_name, 
        use_se=config.use_se, 
        use_stochastic_depth=config.use_stochastic_depth,
        stochastic_depth_prob=config.stochastic_depth_prob
    )
    
    # Print model configuration
    model_config = model.get_config()
    model_config['num_parameters'] = sum(p.numel() for p in model.parameters())
    model_config['trainable_parameters'] = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print("\nModel Configuration:")
    for key, value in model_config.items():
        print(f"{key}: {value}")
    
    # Create trainer
    trainer = Trainer(
        model=model,
        train_loader=data_module.train_dataloader(),
        val_loader=data_module.val_dataloader(),
        config=config
    )
    
    # Train model
    print("\nStarting training...")
    best_acc = trainer.train()
    print(f"Training completed! Best validation accuracy: {best_acc:.4f}%")
    
    # Test model
    print("\nEvaluating on test set...")
    test_metrics = trainer.evaluate(data_module.test_dataloader())
    print(f"Test metrics:")
    for key, value in test_metrics.items():
        print(f"{key}: {value:.4f}")
    
    # Save final model
    final_model_path = os.path.join(
        config.output_dir, 
        "final_models", 
        f"{config.experiment_name}_final.pth"
    )
    os.makedirs(os.path.dirname(final_model_path), exist_ok=True)
    
    # Save model with full state
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'config': config,
        'test_accuracy': test_metrics['accuracy'],
        'best_val_accuracy': best_acc
    }
    torch.save(checkpoint, final_model_path)
    print(f"\nFinal model saved to {final_model_path}")
    
    print(f"\n{'='*80}")
    print(f"Hybrid training completed: {config.experiment_name}")
    print(f"{'='*80}\n")

if __name__ == "__main__":
    main() 