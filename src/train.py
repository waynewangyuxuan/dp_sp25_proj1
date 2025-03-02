import os
import sys

# Add src directory to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from models import ModelFactory
from data.data_module import CIFAR10DataModule
from training.trainer import Trainer
from configs.train_config import TrainingConfig

def main():
    # Create config
    config = TrainingConfig()
    
    # Set device
    if torch.cuda.is_available():
        config.device = 'cuda'
        torch.backends.cudnn.benchmark = True
    else:
        config.device = 'cpu'
        print("WARNING: Using CPU for training")
    
    # Create data module
    data_module = CIFAR10DataModule(
        data_dir=config.data_dir,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory
    )
    data_module.setup()
    
    # Create model
    model = ModelFactory.create(config.model_name)
    
    # Print model summary
    print("\nModel Configuration:")
    for key, value in model.get_config().items():
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
    trainer.train()
    print("Training completed!")

if __name__ == '__main__':
    main() 