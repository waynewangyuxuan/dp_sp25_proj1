# Deep Learning Spring 2025 Project 1: CIFAR-10 Classification

This project implements a ResNet model for CIFAR-10 image classification. The implementation includes a custom training pipeline with learning rate scheduling, model checkpointing, and comprehensive evaluation metrics.

## Project Structure

```
dp_sp25_proj1/
├── src/
│   ├── configs/
│   │   └── train_config.py      # Training configuration
│   ├── data/
│   │   ├── cifar10_dataset.py   # Custom CIFAR-10 dataset implementation
│   │   └── data_module.py       # PyTorch Lightning data module
│   ├── models/
│   │   └── resnet.py           # ResNet model implementation
│   ├── training/
│   │   └── trainer.py          # Training loop implementation
│   ├── train.py                # Main training script
│   └── evaluate.py             # Model evaluation script
├── data/
│   └── cifar10/                # CIFAR-10 dataset files
├── outputs/
│   ├── best_models/            # Best performing model checkpoints
│   ├── evaluations/            # Evaluation results
│   │   └── {experiment_name}_val_acc_{val_acc}_{timestamp}/
│   │       ├── predictions.csv # Test set predictions
│   │       └── model.pth       # Symbolic link to model checkpoint
│   └── training_runs/          # Training run outputs
│       └── {timestamp}/
│           ├── checkpoints/    # Regular training checkpoints
│           └── logs/           # Training logs
├── requirements.txt            # Project dependencies
└── activate.sh                 # Environment activation script
```

## Model Architecture

The project implements a small ResNet model with the following architecture:

- Input: 32x32 RGB images
- Initial convolution: 3x3, 48 channels
- 4 ResNet blocks with BasicBlock:
  - Layer 1: 3 blocks (48 → 48 channels)
  - Layer 2: 4 blocks (48 → 96 channels)
  - Layer 3: 23 blocks (96 → 192 channels)
  - Layer 4: 3 blocks (192 → 384 channels)
- Each BasicBlock contains:
  - 2 convolutional layers (3x3)
  - Batch normalization
  - ReLU activation
  - Skip connections
- Global average pooling
- Dropout (rate=0.2)
- Fully connected layer (384 → 10)
- Output: 10 class probabilities

### Performance
- Test Accuracy: 82.422%
- Model Size: ~4.2M parameters
- Training Time: ~2 hours on a single GPU

## Training Process

The training process includes:

1. Data augmentation:
   - Random horizontal flips (p=0.5)
   - Random rotations (±15 degrees)
   - Color jittering (brightness, contrast, saturation)
   - Normalization (mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010])

2. Training features:
   - Cross-entropy loss
   - Adam optimizer (lr=0.001, weight_decay=0.0001)
   - Cosine learning rate schedule with warmup
   - Model checkpointing (every 5 epochs)
   - Early stopping (patience=10)
   - Progress bars for monitoring

3. Evaluation metrics:
   - Training accuracy
   - Validation accuracy
   - Per-class accuracy
   - Confusion matrix

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

To train the model:
```bash
python src/train.py
```

The training script will:
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

Training parameters can be modified in `src/configs/train_config.py`:
- Model architecture
- Training hyperparameters
- Data augmentation settings
- Output paths
- Logging settings

## Dependencies

- Python 3.9+
- PyTorch
- torchvision
- pandas
- tqdm
- scikit-learn
- matplotlib
