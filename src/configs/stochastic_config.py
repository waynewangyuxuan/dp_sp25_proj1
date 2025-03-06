from dataclasses import dataclass, field
from typing import Optional

@dataclass
class StochasticTrainingConfig:
    # Model
    model_name: str = 'stochastic_resnet'
    
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
    experiment_name: str = 'cifar10_stochastic_resnet'
    resume_from: Optional[str] = None 