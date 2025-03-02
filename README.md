# Deep Learning Project 1 - CIFAR-10 Classification

This project implements a deep learning model for image classification on the CIFAR-10 dataset.

## Project Structure

```
dp_sp25_proj1/
├── data/                    # Data directory
│   └── cifar10/            # CIFAR-10 dataset files
│       ├── cifar-10-python/  # Training and validation data
│       └── cifar_test_nolabel.pkl  # Unlabeled test data
├── src/                    # Source code
│   ├── data/              # Data loading and processing
│   │   ├── __init__.py
│   │   ├── cifar10_dataset.py
│   │   └── data_module.py
│   ├── models/            # Model architectures
│   │   ├── __init__.py
│   │   ├── base_model.py
│   │   ├── model_factory.py
│   │   └── resnet.py
│   ├── training/          # Training scripts and trainer class
│   │   ├── __init__.py
│   │   └── trainer.py
│   ├── configs/           # Configuration classes
│   │   ├── __init__.py
│   │   └── train_config.py
│   ├── evaluate.py        # Evaluation and prediction script
│   ├── train.py          # Main training script
│   └── utils/            # Utility functions
├── outputs/              # Model outputs
│   ├── best_models/     # Best performing models (tracked in git)
│   └── training_runs/   # Training runs organized by timestamp (not tracked)
│       └── YYYY_MM_DD_HH_MM/  # Individual run directory
│           ├── checkpoints/   # Epoch checkpoints
│           ├── best.pth      # Best model for this run
│           └── predictions.csv  # Model predictions
├── requirements.txt      # Python dependencies
├── activate.sh          # Environment activation script
└── README.md            # This file
```

## Setup

1. Create and activate the virtual environment:
```bash
python -m venv venv
source activate.sh
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Download the CIFAR-10 dataset:
```bash
# Configure Kaggle credentials first
mkdir -p ~/.config/kaggle
# Place your kaggle.json in ~/.config/kaggle/
chmod 600 ~/.config/kaggle/kaggle.json

# Download the dataset
kaggle competitions download -c deep-learning-spring-2025-project-1
```

4. Extract the dataset:
```bash
unzip deep-learning-spring-2025-project-1.zip -d data/cifar10/
```

## Data Loading

The project uses a custom data loading pipeline with the following components:

- `CIFAR10Dataset`: A PyTorch Dataset class that handles loading and preprocessing of CIFAR-10 data
- `CIFAR10DataModule`: A data module that manages data loading, including:
  - Training/validation split (4 batches for training, 1 for validation)
  - Data augmentation (random crop, horizontal flip)
  - Normalization
  - DataLoader configuration
  - Test dataset handling (cifar_test_nolabel.pkl)

The data is organized as follows:
- Training data: 40,000 images (batches 1-4)
- Validation data: 10,000 images (batch 5)
- Test data: Unlabeled images from cifar_test_nolabel.pkl

To test the data loading pipeline:
```bash
python src/test_dataloader.py
```

## Model Architecture

We implement a small ResNet variant (~700K parameters) specifically designed for CIFAR-10 classification. The architecture follows the ResNet design principles while maintaining a small parameter count:

### Network Structure
- Initial Conv Layer: 3 → 16 channels (3×3 conv, stride 1)
- Stage 1: 16 → 16 channels (2 BasicBlocks)
- Stage 2: 16 → 32 channels (2 BasicBlocks)
- Stage 3: 32 → 64 channels (2 BasicBlocks)
- Stage 4: 64 → 128 channels (2 BasicBlocks)
- Global Average Pooling
- Fully Connected: 128 → 10 classes

### Key Features
- BasicBlock with two 3×3 convolutions and residual connection
- Batch Normalization after each convolution
- ReLU activation

## Training

The training pipeline is managed by the `Trainer` class in `src/training/trainer.py`. Key features include:

- Cosine learning rate schedule with warmup
- Label smoothing for better generalization
- Progress bars for both epoch and batch-level progress
- Regular checkpointing and best model saving
- Validation after each epoch

To start training:
```bash
python src/train.py
```

## Evaluation and Predictions

The evaluation pipeline (`src/evaluate.py`) generates predictions for the test set:

1. Loads the best performing model from `outputs/best_models/`
2. Runs inference on the test set
3. Generates predictions in the required format:
   - CSV file with columns "ID" and "Labels"
   - Sorted by ID for consistency

To generate predictions:
```bash
python src/evaluate.py
```

## Output Organization

The project organizes outputs in a structured way:

- `outputs/best_models/`: Stores the best performing models (tracked in git)
  - `cifar10_resnet_best.pth`: Best model checkpoint with highest validation accuracy

- `outputs/training_runs/`: Training runs organized by timestamp (not tracked in git)
  - Each run has its own directory (YYYY_MM_DD_HH_MM format)
  - Contains checkpoints, best model for the run, and predictions

This structure ensures:
- Best models are version controlled
- Training runs are organized and easily identifiable
- Clear separation between tracked and untracked files
