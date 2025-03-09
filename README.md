# Deep Learning Spring 2025 Project 1: CIFAR-10 Classification

This project implements a ResNet model for CIFAR-10 image classification. The implementation includes a custom training pipeline with learning rate scheduling, model checkpointing, and comprehensive evaluation metrics.

## Project Structure

```
dp_sp25_proj1/
├── src/
│   ├── configs/
│   │   ├── train_config.py      # Training configuration
│   │   └── stochastic_config.py # Stochastic model configuration
│   ├── data/
│   │   ├── cifar10_dataset.py   # Custom CIFAR-10 dataset implementation
│   │   ├── data_module.py       # PyTorch Lightning data module
│   │   └── augmentations.py     # Data augmentation implementations
│   ├── models/
│   │   ├── resnet.py            # ResNet model implementation
│   │   ├── stochastic_resnet.py # Stochastic depth ResNet implementation
│   │   ├── model_factory.py     # Factory for creating models
│   │   └── stochastic_model_factory.py # Factory for stochastic models
│   ├── training/
│   │   └── trainer.py           # Training loop implementation
│   ├── train.py                 # Main training script
│   ├── train_stochastic.py      # Stochastic model training script
│   ├── evaluate.py              # Basic model evaluation script
│   ├── evaluate_advanced.py     # Advanced evaluation with TTA
│   └── get_model_accuracy.py    # Script to extract model accuracy
├── data/
│   └── cifar10/                 # CIFAR-10 dataset files
├── outputs/
│   ├── best_models/             # Best performing model checkpoints
│   ├── evaluations/             # Evaluation results
│   │   └── {experiment_name}_val_acc_{val_acc}_{timestamp}_model_{model_file}/
│   │       ├── predictions.csv  # Test set predictions
│   │       └── model.pth        # Copy of model checkpoint
│   └── training_runs/           # Training run outputs
│       └── {timestamp}/
│           ├── checkpoints/     # Regular training checkpoints
│           └── logs/            # Training logs
├── requirements.txt             # Project dependencies
└── activate.sh                  # Environment activation script
```

## Model Architectures

### 1. Enhanced ResNet with SE Blocks

The project implements a small ResNet model with Squeeze-and-Excitation blocks:

- Input: 32x32 RGB images
- Initial convolution: 3x3, 32 channels
- 4 ResNet blocks with BasicBlock:
  - Layer 1: 2 blocks (32 → 32 channels)
  - Layer 2: 2 blocks (32 → 64 channels)
  - Layer 3: 2 blocks (64 → 128 channels)
  - Layer 4: 2 blocks (128 → 256 channels)
- Each BasicBlock contains:
  - 2 convolutional layers (3x3)
  - Batch normalization
  - ReLU activation
  - Skip connections
  - Squeeze-and-Excitation block
- Global average pooling
- Dropout (rate=0.3)
- Fully connected layer (256 → 10)
- Output: 10 class probabilities

#### Performance
- Test Accuracy: 83.14%
- Model Size: ~2.8M parameters
- Training Time: ~2 hours on a single GPU

### 2. Stochastic Depth ResNet

An alternative implementation using stochastic depth for regularization:

- Similar architecture to the enhanced ResNet
- Stochastic depth: randomly drops entire residual blocks during training
- Probability of dropping increases linearly with depth
- Maximum drop probability: 0.2
- No SE blocks

#### Performance
- Test Accuracy: 78.57%
- Model Size: ~2.8M parameters
- Training Time: ~2 hours on a single GPU

### 3. Enhanced ResNet with RandAugment

An improved version of the enhanced ResNet with advanced data augmentation:

- Same architecture as the enhanced ResNet with SE blocks
- Trained with RandAugment data augmentation
- RandAugment applies a sequence of random transformations with configurable magnitude

#### Performance
- Test Accuracy: 83.64% (with Test-Time Augmentation)
- Model Size: ~2.8M parameters
- Training Time: ~2.5 hours on a single GPU

## Training Process

The training process includes:

1. Data augmentation:
   - Random horizontal flips (p=0.5)
   - Random rotations (±15 degrees)
   - Color jittering (brightness, contrast, saturation, hue)
   - Random perspective (p=0.5)
   - Random erasing (p=0.3)
   - CutMix augmentation (p=0.7)
   - RandAugment (optional, 2 operations with magnitude 9)
   - Normalization (mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010])

2. Training features:
   - Cross-entropy loss with label smoothing (0.1)
   - SGD optimizer (lr=0.05, momentum=0.9, weight_decay=1e-4)
   - Learning rate scheduling (OneCycle or Cosine Annealing)
   - Model checkpointing (every 10 epochs)
   - Gradient clipping (1.0)

3. Evaluation metrics:
   - Training accuracy
   - Validation accuracy
   - Test accuracy
   - Test accuracy with Test-Time Augmentation (TTA)

## Usage

### Environment Setup

1. Create and activate the virtual environment:
```bash
source activate.sh
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

### Training

To train the enhanced ResNet model with SE blocks:
```bash
python src/train.py
```

To train the stochastic depth ResNet model:
```bash
python src/train_stochastic.py
```

The training scripts will:
- Create necessary output directories
- Train the model with the specified configuration
- Save checkpoints and logs
- Display training progress and metrics

### Evaluation

#### Basic Evaluation

To evaluate a trained model and generate predictions:
```bash
python src/evaluate.py
```

The evaluation script will:
- Load the best model checkpoint
- Generate predictions for the test set
- Create an evaluation folder with:
  - `predictions.csv`: Test set predictions
  - `model.pth`: Copy of the model used
- Display sample predictions and statistics

#### Advanced Evaluation with Test-Time Augmentation

To evaluate a model with Test-Time Augmentation (TTA):
```bash
python src/evaluate_advanced.py --tta
```

Additional options:
```bash
# Specify a custom model path
python src/evaluate_advanced.py --model-path /path/to/model.pth --tta

# Adjust the number of TTA transforms
python src/evaluate_advanced.py --tta --tta-transforms 16

# Use data augmentation during evaluation (not recommended)
python src/evaluate_advanced.py --use-augment
```

#### Quick Model Information

To quickly check a model's validation accuracy and other information:
```bash
python src/get_model_accuracy.py /path/to/model.pth
```

For detailed model information:
```bash
python src/get_model_accuracy.py /path/to/model.pth --verbose
```

## Experimental Results

We conducted several experiments to improve the model's performance:

| Model Configuration | Test Accuracy | Notes |
|---------------------|---------------|-------|
| ResNet with SE Blocks | 83.14% | Baseline model |
| ResNet with SE Blocks + TTA (8 transforms) | 83.579% | Significant improvement with TTA |
| ResNet with Stochastic Depth | 78.57% | Less effective than SE blocks |
| ResNet with RandAugment + TTA | 83.64% | Best overall performance |
| ResNet with TTA (16 transforms) | 83.38% | More transforms reduced accuracy |

### Key Findings:

1. **Squeeze-and-Excitation Blocks** significantly improve performance over the base ResNet.
2. **Test-Time Augmentation (TTA)** consistently improves model performance, with gains of 0.439% for the baseline model.
3. **RandAugment** provides a small but consistent improvement in accuracy.
4. **The optimal number of TTA transforms** is around 8, as using 16 transforms actually reduced accuracy.

## Output Organization

### Training Outputs

Training outputs are organized in the `outputs/training_runs/{timestamp}/` directory:
- `checkpoints/`: Regular training checkpoints
- `logs/`: Training logs and metrics

### Best Models

The best performing models are stored in `outputs/best_models/`:
- `{experiment_name}_best.pth`: Best model checkpoint
- `best_test_83140.pth`: Model with best test accuracy (83.14%)
- `cifar10_resnet_bestTest_83140.pth`: Symbolic link to the best test model

### Evaluation Results

Evaluation results are stored in `outputs/evaluations/`:
- Each evaluation run gets its own folder named with validation accuracy and model filename
- Folder format: `{experiment_name}_val_acc_{val_acc}_{timestamp}_model_{model_file}/`
- Contains predictions CSV and model file

## Configuration

Training parameters can be modified in:
- `src/configs/train_config.py`: Enhanced ResNet configuration
- `src/configs/stochastic_config.py`: Stochastic depth ResNet configuration

Key configurable parameters:
- `use_randaugment`: Enable/disable RandAugment
- `randaugment_num_ops`: Number of operations to apply in RandAugment (default: 2)
- `randaugment_magnitude`: Magnitude of operations in RandAugment (default: 9)
- `use_cutmix`: Enable/disable CutMix augmentation
- `cutmix_prob`: Probability of applying CutMix (default: 0.7)

## Dependencies

- Python 3.9+
- PyTorch
- torchvision
- pandas
- tqdm
- scikit-learn
- matplotlib
