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
│   ├── training/          # Training scripts
│   └── utils/             # Utility functions
├── configs/               # Configuration files
├── notebooks/            # Jupyter notebooks
├── outputs/              # Model checkpoints and logs
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
- Spatial reduction (stride=2) at stages 2, 3, and 4
- Kaiming initialization for better training dynamics

To test the model:
```bash
python src/test_model.py
```

## Training

[Training documentation will be added as the project progresses]

## Submission Format

The model predictions on the test set will be saved in a CSV file with the following format:
- Each row corresponds to an image in the test set
- Two columns: 'id' (image index) and 'label' (predicted class, 0-9)
- The order of predictions should match the order of images in cifar_test_nolabel.pkl

## Results

[Results and evaluation metrics will be added as the project progresses]
