from dataclasses import dataclass, field
from typing import Optional

"""
Configuration for Hybrid ResNet model with SE blocks and stochastic depth.

This configuration is used by the train_hybrid_resnet_se_stochastic.py script to train
a Hybrid ResNet model that combines Squeeze-and-Excitation (SE) blocks for channel attention
and stochastic depth for regularization on the CIFAR-10 dataset.
"""

@dataclass
class HybridResNetConfig:
    # Model
    model_name: str = 'hybrid_resnet'
    
    # Hybrid model specific parameters
    use_se: bool = True  # Use Squeeze-and-Excitation blocks
    use_stochastic_depth: bool = True  # Use stochastic depth
    stochastic_depth_prob: float = 0.2  # Probability of dropping a layer
    
    # Data
    data_dir: str = 'data/cifar10'
    batch_size: int = 128
    num_workers: int = 4
    pin_memory: bool = True
    
    # Training
    num_epochs: int = 200
    learning_rate: float = 0.05
    weight_decay: float = 1e-4
    momentum: float = 0.9
    
    # Learning rate schedule
    lr_schedule: str = 'cosine'  # Use cosine annealing
    max_lr: float = 0.1
    
    # Augmentation - Enhanced
    random_crop_padding: int = 4
    random_horizontal_flip: bool = True
    mixup_alpha: float = 1.0
    cutout_holes: int = 1
    cutout_length: int = 16
    random_rotation: int = 15
    color_jitter: dict = field(default_factory=lambda: {
        'brightness': 0.3,
        'contrast': 0.3,
        'saturation': 0.3,
        'hue': 0.1
    })
    
    # CutMix - Increased probability
    use_cutmix: bool = True
    cutmix_alpha: float = 1.0
    cutmix_prob: float = 0.7
    
    # Regularization
    label_smoothing: float = 0.1
    dropout_rate: float = 0.3
    grad_clip: float = 1.0
    
    # Device
    device: str = 'cuda'
    
    # Logging and saving
    log_interval: int = 100
    save_interval: int = 10
    output_dir: str = 'outputs'
    experiment_name: str = 'cifar10_hybrid_resnet'
    resume_from: Optional[str] = None 