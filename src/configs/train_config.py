from dataclasses import dataclass
from typing import Optional

@dataclass
class TrainingConfig:
    # Model
    model_name: str = 'resnet_small'
    
    # Data
    data_dir: str = 'data/cifar10'
    batch_size: int = 128
    num_workers: int = 4
    pin_memory: bool = True
    
    # Training
    num_epochs: int = 200
    learning_rate: float = 0.1
    weight_decay: float = 5e-4
    momentum: float = 0.9
    
    # Learning rate schedule
    lr_schedule: str = 'cosine'  # ['step', 'cosine']
    lr_step_size: int = 60
    lr_gamma: float = 0.1
    warmup_epochs: int = 5
    
    # Augmentation
    random_crop_padding: int = 4
    random_horizontal_flip: bool = True
    
    # Regularization
    label_smoothing: float = 0.1
    
    # Device
    device: str = 'cuda'
    
    # Logging and saving
    log_interval: int = 100  # Print every N batches
    save_interval: int = 10  # Save checkpoint every N epochs
    output_dir: str = 'outputs'
    experiment_name: str = 'cifar10_resnet'
    resume_from: Optional[str] = None  # Path to checkpoint to resume from 