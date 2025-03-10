import os
import sys
import torch
import numpy as np
import random
from datetime import datetime

# Add src directory to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from models import ModelFactory
from data.data_module import CIFAR10DataModule
from training.trainer import Trainer
from configs.resnet_randaugment_config import ResNetRandAugmentConfig

"""
train_resnet_randaugment.py

This script trains a ResNet model with RandAugment data augmentation on the CIFAR-10 dataset.
It uses the standard ResNet architecture with RandAugment for improved generalization.
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
    config = ResNetRandAugmentConfig()
    
    # Add timestamp to experiment name for unique identification
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    config.experiment_name = f"{config.experiment_name}_{timestamp}"
    
    print(f"\n{'='*80}")
    print(f"Starting ResNet with RandAugment training: {config.experiment_name}")
    print(f"Model: {config.model_name} with RandAugment (ops={config.randaugment_num_ops}, magnitude={config.randaugment_magnitude})")
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
        cutmix_prob=config.cutmix_prob,
        use_randaugment=config.use_randaugment,
        randaugment_num_ops=config.randaugment_num_ops,
        randaugment_magnitude=config.randaugment_magnitude
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
    model = ModelFactory.create(config.model_name)
    
    # Print model summary
    print("\nModel Configuration:")
    model_config = model.get_config()
    for key, value in model_config.items():
        print(f"{key}: {value}")
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nTotal parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
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
    
    # Evaluate on test set
    print("\nEvaluating on test set...")
    test_metrics = trainer.evaluate(data_module.test_dataloader())
    print(f"Test accuracy: {test_metrics['accuracy']:.4f}%")
    
    print(f"\n{'='*80}")
    print(f"Training completed: {config.experiment_name}")
    print(f"{'='*80}\n")

if __name__ == '__main__':
    main() 