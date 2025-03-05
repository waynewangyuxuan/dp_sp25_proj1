from dataclasses import dataclass, field
from typing import Optional

@dataclass
class TrainingConfig:
    # Model
    model_name: str = 'resnet_small'
    
    # Data
    data_dir: str = 'data/cifar10'
    batch_size: int = 128  # Slightly smaller batch size for better generalization
    num_workers: int = 4
    pin_memory: bool = True
    
    # Training
    num_epochs: int = 100  # Reduced from 300 to 100 epochs
    learning_rate: float = 0.1
    weight_decay: float = 5e-4  # Increased weight decay
    momentum: float = 0.9
    
    # Learning rate schedule
    lr_schedule: str = 'onecycle'
    max_lr: float = 0.1
    
    # Augmentation
    random_crop_padding: int = 4
    random_horizontal_flip: bool = True
    mixup_alpha: float = 1.0
    cutout_holes: int = 1
    cutout_length: int = 16
    random_rotation: int = 15
    color_jitter: dict = field(default_factory=lambda: {
        'brightness': 0.2,
        'contrast': 0.2,
        'saturation': 0.2
    })
    
    # CutMix
    use_cutmix: bool = True
    cutmix_alpha: float = 1.0
    cutmix_prob: float = 0.5
    
    # Regularization
    label_smoothing: float = 0.1
    dropout_rate: float = 0.2
    grad_clip: float = 1.0
    
    # Device
    device: str = 'cuda'
    
    # Logging and saving
    log_interval: int = 100
    save_interval: int = 10
    output_dir: str = 'outputs'
    experiment_name: str = 'cifar10_resnet_cutmix'  # Updated experiment name to reflect CutMix usage
    resume_from: Optional[str] = None 