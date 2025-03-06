import os
import torch
import numpy as np
from tqdm import tqdm

from models.stochastic_model_factory import StochasticModelFactory
from data import CIFAR10DataModule
from training import Trainer
from configs.stochastic_config import StochasticTrainingConfig

def main():
    # Set random seeds for reproducibility
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    np.random.seed(42)
    
    # Load configuration
    config = StochasticTrainingConfig()
    
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
    
    # Create model
    model = StochasticModelFactory.create(config.model_name)
    
    # Print model configuration
    model_config = model.get_config()
    model_config['num_parameters'] = sum(p.numel() for p in model.parameters())
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
    trainer.train()
    
    # Test model
    print("\nEvaluating on test set...")
    test_loader = data_module.test_dataloader()
    test_metrics = trainer.evaluate(test_loader)
    
    print(f"\nTest metrics:")
    for key, value in test_metrics.items():
        print(f"{key}: {value:.4f}")
    
    # Save final model
    final_model_path = os.path.join(
        config.output_dir, 
        "final_models", 
        f"{config.experiment_name}_final.pth"
    )
    os.makedirs(os.path.dirname(final_model_path), exist_ok=True)
    torch.save(model.state_dict(), final_model_path)
    print(f"\nFinal model saved to {final_model_path}")

if __name__ == "__main__":
    main() 