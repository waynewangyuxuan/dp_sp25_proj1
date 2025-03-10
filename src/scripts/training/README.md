# Training Scripts

This directory contains scripts for training different variants of ResNet models on the CIFAR-10 dataset.

## Available Training Scripts

### 1. `train_resnet_randaugment.py`

Trains a standard ResNet model with RandAugment data augmentation.

- **Model**: ResNet with standard architecture
- **Key Feature**: Uses RandAugment for data augmentation
- **Configuration**: Uses `TrainingConfig` from `configs/train_config.py`
- **Expected Test Accuracy**: ~83.6% (with Test-Time Augmentation)

```bash
python src/scripts/training/train_resnet_randaugment.py
```

### 2. `train_hybrid_resnet_se_stochastic.py`

Trains a hybrid ResNet model that combines Squeeze-and-Excitation (SE) blocks and stochastic depth.

- **Model**: Hybrid ResNet with SE blocks and stochastic depth
- **Key Features**: 
  - SE blocks for channel attention
  - Stochastic depth for regularization
- **Configuration**: Uses `HybridTrainingConfig` from `configs/hybrid_config.py`
- **Expected Test Accuracy**: ~84.2%

```bash
python src/scripts/training/train_hybrid_resnet_se_stochastic.py
```

### 3. `train_resnet_stochastic.py`

Trains a ResNet model with stochastic depth regularization.

- **Model**: ResNet with stochastic depth
- **Key Feature**: Uses stochastic depth (randomly drops layers during training)
- **Configuration**: Uses modified `TrainingConfig` from `configs/train_config.py`
- **Expected Test Accuracy**: ~78.6%

```bash
python src/scripts/training/train_resnet_stochastic.py
```

## Training Output

All training scripts produce detailed logs and save models in the following locations:

- **Training logs**: `outputs/training_runs/{timestamp}/logs/training.log`
- **Metrics CSV**: `outputs/training_runs/{timestamp}/logs/metrics.csv`
- **Best model**: `outputs/best_models/{experiment_name}_best.pth`
- **Final model**: `outputs/final_models/{experiment_name}_final.pth`

## Choosing the Right Model

- For best overall performance, use the **Hybrid ResNet** with SE blocks and stochastic depth.
- For a good balance of performance and simplicity, use the **ResNet with RandAugment**.
- For experimental purposes or to understand the impact of stochastic depth alone, use the **ResNet with stochastic depth**.

All models can benefit from Test-Time Augmentation (TTA) during evaluation, which can be enabled using the `--tta` flag with the evaluation scripts. 