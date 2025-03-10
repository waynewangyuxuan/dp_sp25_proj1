from dataclasses import dataclass, field
from typing import Optional

"""
Configuration for ResNet model with RandAugment data augmentation.

This configuration is used by the train_resnet_randaugment.py script to train
a ResNet model with RandAugment data augmentation on the CIFAR-10 dataset.
"""

@dataclass
class ResNetRandAugmentConfig:
    # Model
    model_name: str = 'resnet_small'
    
    # Data
    data_dir: str = 'data/cifar10'
    batch_size: int = 128
    num_workers: int = 4
    pin_memory: bool = True
    
    # Training
    num_epochs: int = 200  # Increased epochs for better convergence
    learning_rate: float = 0.05  # Slightly lower initial learning rate
    weight_decay: float = 1e-4  # Adjusted weight decay
    momentum: float = 0.9
    
    # Learning rate schedule
    lr_schedule: str = 'onecycle'  # Keep OneCycle policy
    max_lr: float = 0.1
    
    # Augmentation - Enhanced
    random_crop_padding: int = 4
    random_horizontal_flip: bool = True
    mixup_alpha: float = 1.0
    cutout_holes: int = 1
    cutout_length: int = 16
    random_rotation: int = 15
    color_jitter: dict = field(default_factory=lambda: {
        'brightness': 0.3,  # Increased
        'contrast': 0.3,    # Increased
        'saturation': 0.3,  # Increased
        'hue': 0.1          # Added hue jitter
    })
    
    # RandAugment parameters
    use_randaugment: bool = True
    randaugment_num_ops: int = 2  # Number of operations to apply
    randaugment_magnitude: int = 9  # Magnitude of augmentation (0-10)
    
    # CutMix - Increased probability
    use_cutmix: bool = True
    cutmix_alpha: float = 1.0
    cutmix_prob: float = 0.7  # Increased from 0.5
    
    # Regularization
    label_smoothing: float = 0.1
    dropout_rate: float = 0.3  # Increased dropout for better regularization
    grad_clip: float = 1.0
    
    # Device
    device: str = 'cuda'
    
    # Logging and saving
    log_interval: int = 100
    save_interval: int = 10
    output_dir: str = 'outputs'
    experiment_name: str = 'cifar10_resnet_randaug'  # Updated name to reflect RandAugment
    resume_from: Optional[str] = None 