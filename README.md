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
│   │   └── data_module.py       # PyTorch Lightning data module
│   ├── models/
│   │   ├── resnet.py            # ResNet model implementation
│   │   ├── stochastic_resnet.py # Stochastic depth ResNet implementation
│   │   ├── model_factory.py     # Factory for creating models
│   │   └── stochastic_model_factory.py # Factory for stochastic models
│   ├── training/
│   │   └── trainer.py           # Training loop implementation
│   ├── train.py                 # Main training script
│   ├── train_stochastic.py      # Stochastic model training script
│   └── evaluate.py              # Model evaluation script
├── data/
│   └── cifar10/                 # CIFAR-10 dataset files
├── outputs/
│   ├── best_models/             # Best performing model checkpoints
│   ├── evaluations/             # Evaluation results
│   │   └── {experiment_name}_val_acc_{val_acc}_{timestamp}/
│   │       ├── predictions.csv  # Test set predictions
│   │       └── model.pth        # Symbolic link to model checkpoint
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

## Training Process

The training process includes:

1. Data augmentation:
   - Random horizontal flips (p=0.5)
   - Random rotations (±15 degrees)
   - Color jittering (brightness, contrast, saturation, hue)
   - Random perspective (p=0.5)
   - Random erasing (p=0.3)
   - CutMix augmentation (p=0.7)
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

To evaluate a trained model and generate predictions:
```bash
python src/evaluate.py
```

The evaluation script will:
- Load the best model checkpoint
- Generate predictions for the test set
- Create an evaluation folder with:
  - `predictions.csv`: Test set predictions
  - `model.pth`: Symbolic link to the model used
- Display sample predictions and statistics

## Output Organization

### Training Outputs

Training outputs are organized in the `outputs/training_runs/{timestamp}/` directory:
- `checkpoints/`: Regular training checkpoints
- `logs/`: Training logs and metrics

### Best Models

The best performing models are stored in `outputs/best_models/`:
- `{experiment_name}_best.pth`: Best model checkpoint

### Evaluation Results

Evaluation results are stored in `outputs/evaluations/`:
- Each evaluation run gets its own folder named with validation accuracy
- Folder format: `{experiment_name}_val_acc_{val_acc}_{timestamp}/`
- Contains predictions CSV and model symbolic link

## Configuration

Training parameters can be modified in:
- `src/configs/train_config.py`: Enhanced ResNet configuration
- `src/configs/stochastic_config.py`: Stochastic depth ResNet configuration

## Dependencies

- Python 3.9+
- PyTorch
- torchvision
- pandas
- tqdm
- scikit-learn
- matplotlib
