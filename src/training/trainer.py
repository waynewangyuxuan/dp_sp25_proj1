import os
import sys
import time
import numpy as np
from typing import Dict, Optional, Tuple, Any
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import _LRScheduler, OneCycleLR
from tqdm import tqdm
import json
import csv
import matplotlib.pyplot as plt

# Add src directory to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models import BaseModel
# Use a more generic import for configuration
from configs.resnet_randaugment_config import ResNetRandAugmentConfig
from configs.hybrid_config import HybridResNetConfig
from configs.stochastic_config import ResNetStochasticConfig
from data.augmentations import cutmix_data

# Define a generic config type for type hints
ConfigType = Any  # This could be any of the config classes

class WarmupCosineScheduler(_LRScheduler):
    """Cosine LR scheduler with warmup."""
    
    def __init__(self, optimizer, warmup_epochs: int, total_epochs: int):
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        super().__init__(optimizer)

    def get_lr(self):
        epoch = self.last_epoch
        if epoch < self.warmup_epochs:
            # Linear warmup
            alpha = epoch / self.warmup_epochs
            return [base_lr * alpha for base_lr in self.base_lrs]
        else:
            # Cosine decay
            progress = (epoch - self.warmup_epochs) / (self.total_epochs - self.warmup_epochs)
            alpha = 0.5 * (1 + torch.cos(torch.tensor(progress * torch.pi)))
            return [base_lr * alpha for base_lr in self.base_lrs]

class Trainer:
    """Trainer class to handle model training and evaluation"""
    
    def __init__(self, 
                 model: BaseModel,
                 train_loader: DataLoader,
                 val_loader: DataLoader,
                 config: ConfigType):
        """
        Args:
            model: Model to train
            train_loader: Training data loader
            val_loader: Validation data loader
            config: Training configuration
        """
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        
        # Setup device
        self.device = torch.device(config.device)
        self.model = self.model.to(self.device)
        
        # Setup loss and optimizer
        self.criterion = nn.CrossEntropyLoss(label_smoothing=config.label_smoothing)
        self.optimizer = optim.SGD(
            self.model.parameters(),
            lr=config.learning_rate,
            momentum=config.momentum,
            weight_decay=config.weight_decay
        )
        
        # Setup learning rate scheduler
        if config.lr_schedule == 'onecycle':
            self.scheduler = OneCycleLR(
                self.optimizer,
                max_lr=config.max_lr,
                epochs=config.num_epochs,
                steps_per_epoch=len(train_loader),
                pct_start=0.1,  # 10% warmup
                div_factor=25,  # Initial lr = max_lr/25
                final_div_factor=1e4  # Final lr = max_lr/10000
            )
        else:
            self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=config.num_epochs,
                eta_min=1e-6
            )
        
        # Setup tracking variables
        self.epoch = 0
        self.best_acc = 0.0
        self.train_steps = 0
        
        # Setup output directories
        self._setup_output_dirs()
        
        # Initialize metrics tracking
        self.metrics_history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': [],
            'learning_rate': []
        }
        
        # Resume from checkpoint if specified
        if config.resume_from is not None:
            self._load_checkpoint(config.resume_from)
    
    def _setup_output_dirs(self):
        """Setup output directories for the current training run"""
        # Create timestamp for this run
        timestamp = datetime.now().strftime("%Y_%m_%d_%H_%M")
        
        # Setup directory structure
        self.run_dir = os.path.join(self.config.output_dir, "training_runs", timestamp)
        self.checkpoints_dir = os.path.join(self.run_dir, "checkpoints")
        self.logs_dir = os.path.join(self.run_dir, "logs")
        self.best_models_dir = os.path.join(self.config.output_dir, "best_models")
        
        # Create directories
        os.makedirs(self.run_dir, exist_ok=True)
        os.makedirs(self.checkpoints_dir, exist_ok=True)
        os.makedirs(self.logs_dir, exist_ok=True)
        os.makedirs(self.best_models_dir, exist_ok=True)
        
        # Setup log files
        self.log_file = os.path.join(self.logs_dir, "training.log")
        self.metrics_file = os.path.join(self.logs_dir, "metrics.csv")
        self.config_file = os.path.join(self.logs_dir, "config.json")
        
        # Save configuration
        self._save_config()
        
        # Initialize metrics CSV file with header
        with open(self.metrics_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['epoch', 'train_loss', 'train_acc', 'val_loss', 'val_acc', 'learning_rate', 
                            'time_elapsed', 'memory_used_mb', 'batch_time_ms', 'eta_hours'])
        
        # Log model architecture and parameters
        self._log_model_info()
    
    def _log_model_info(self):
        """Log detailed information about the model architecture and parameters"""
        # Count total parameters
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        # Get model configuration
        model_config = self.model.get_config() if hasattr(self.model, 'get_config') else {}
        
        # Log model information
        self._log_to_file("=" * 80)
        self._log_to_file("MODEL INFORMATION")
        self._log_to_file("=" * 80)
        self._log_to_file(f"Model Name: {self.config.model_name}")
        self._log_to_file(f"Total Parameters: {total_params:,}")
        self._log_to_file(f"Trainable Parameters: {trainable_params:,}")
        
        # Log model configuration
        if model_config:
            self._log_to_file("\nModel Configuration:")
            for key, value in model_config.items():
                self._log_to_file(f"  {key}: {value}")
        
        # Log model architecture
        self._log_to_file("\nModel Architecture:")
        self._log_to_file(str(self.model))
        
        # Log training configuration
        self._log_to_file("\n" + "=" * 80)
        self._log_to_file("TRAINING CONFIGURATION")
        self._log_to_file("=" * 80)
        for key, value in vars(self.config).items():
            if not key.startswith('_'):  # Skip private attributes
                self._log_to_file(f"{key}: {value}")
        
        self._log_to_file("\n" + "=" * 80)
        self._log_to_file("TRAINING LOG")
        self._log_to_file("=" * 80)
    
    def _save_config(self):
        """Save configuration to a JSON file"""
        # Convert config to dict
        config_dict = {}
        for key, value in vars(self.config).items():
            if not key.startswith('_'):  # Skip private attributes
                # Handle non-serializable types
                if isinstance(value, (int, float, str, bool, list, dict, tuple)) or value is None:
                    config_dict[key] = value
                else:
                    config_dict[key] = str(value)
        
        # Save to file
        with open(self.config_file, 'w') as f:
            json.dump(config_dict, f, indent=4)
    
    def _log_to_file(self, message: str):
        """Log a message to the log file"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with open(self.log_file, 'a') as f:
            f.write(f"[{timestamp}] {message}\n")
    
    def _log_metrics(self, epoch: int, train_metrics: Dict[str, float], val_metrics: Dict[str, float], 
                    lr: float, time_elapsed: float, memory_used: float, batch_time: float, eta_hours: float):
        """Log metrics to CSV file"""
        with open(self.metrics_file, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                epoch,
                train_metrics['loss'],
                train_metrics['accuracy'],
                val_metrics['loss'],
                val_metrics['accuracy'],
                lr,
                time_elapsed,
                memory_used,
                batch_time,
                eta_hours
            ])
        
        # Update metrics history
        self.metrics_history['train_loss'].append(train_metrics['loss'])
        self.metrics_history['train_acc'].append(train_metrics['accuracy'])
        self.metrics_history['val_loss'].append(val_metrics['loss'])
        self.metrics_history['val_acc'].append(val_metrics['accuracy'])
        self.metrics_history['learning_rate'].append(lr)
    
    def _generate_training_graphs(self):
        """Generate training graphs from metrics history"""
        try:
            # Create figure directory
            figures_dir = os.path.join(self.logs_dir, "figures")
            os.makedirs(figures_dir, exist_ok=True)
            
            # Set style
            plt.style.use('seaborn-v0_8-darkgrid')
            
            # Get experiment name
            experiment_name = self.config.experiment_name
            
            # 1. Training Loss and Validation Loss
            plt.figure(figsize=(10, 6))
            epochs = list(range(1, len(self.metrics_history['train_loss']) + 1))
            plt.plot(epochs, self.metrics_history['train_loss'], label='Training Loss', marker='o', linestyle='-', color='#1f77b4')
            plt.plot(epochs, self.metrics_history['val_loss'], label='Validation Loss', marker='s', linestyle='-', color='#ff7f0e')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.title(f'Training and Validation Loss\n{experiment_name}')
            plt.legend()
            plt.grid(True)
            plt.tight_layout()
            loss_plot_path = os.path.join(figures_dir, 'loss_plot.png')
            plt.savefig(loss_plot_path, dpi=300)
            plt.close()
            
            # 2. Validation Loss and Validation Accuracy
            fig, ax1 = plt.subplots(figsize=(10, 6))
            
            # Validation Loss (left y-axis)
            color = '#1f77b4'
            ax1.set_xlabel('Epoch')
            ax1.set_ylabel('Validation Loss', color=color)
            ax1.plot(epochs, self.metrics_history['val_loss'], label='Validation Loss', marker='o', linestyle='-', color=color)
            ax1.tick_params(axis='y', labelcolor=color)
            
            # Validation Accuracy (right y-axis)
            ax2 = ax1.twinx()
            color = '#ff7f0e'
            ax2.set_ylabel('Validation Accuracy (%)', color=color)
            ax2.plot(epochs, self.metrics_history['val_acc'], label='Validation Accuracy', marker='s', linestyle='-', color=color)
            ax2.tick_params(axis='y', labelcolor=color)
            
            # Add legend
            lines1, labels1 = ax1.get_legend_handles_labels()
            lines2, labels2 = ax2.get_legend_handles_labels()
            ax1.legend(lines1 + lines2, labels1 + labels2, loc='center right')
            
            plt.title(f'Validation Loss and Accuracy\n{experiment_name}')
            plt.grid(True)
            plt.tight_layout()
            val_plot_path = os.path.join(figures_dir, 'validation_plot.png')
            plt.savefig(val_plot_path, dpi=300)
            plt.close()
            
            # 3. Learning Rate
            plt.figure(figsize=(10, 6))
            plt.plot(epochs, self.metrics_history['learning_rate'], label='Learning Rate', marker='o', linestyle='-', color='#2ca02c')
            plt.xlabel('Epoch')
            plt.ylabel('Learning Rate')
            plt.title(f'Learning Rate Schedule\n{experiment_name}')
            plt.grid(True)
            plt.tight_layout()
            lr_plot_path = os.path.join(figures_dir, 'learning_rate_plot.png')
            plt.savefig(lr_plot_path, dpi=300)
            plt.close()
            
            # 4. Combined plot with all metrics
            plt.figure(figsize=(10, 15))
            
            # Create 3 subplots
            plt.subplot(3, 1, 1)
            plt.plot(epochs, self.metrics_history['train_loss'], label='Training Loss', marker='o', linestyle='-', color='#1f77b4')
            plt.plot(epochs, self.metrics_history['val_loss'], label='Validation Loss', marker='s', linestyle='-', color='#ff7f0e')
            plt.ylabel('Loss')
            plt.title(f'Training Metrics\n{experiment_name}')
            plt.legend()
            plt.grid(True)
            
            plt.subplot(3, 1, 2)
            plt.plot(epochs, self.metrics_history['val_acc'], label='Validation Accuracy', marker='o', linestyle='-', color='#2ca02c')
            plt.ylabel('Accuracy (%)')
            plt.legend()
            plt.grid(True)
            
            plt.subplot(3, 1, 3)
            plt.plot(epochs, self.metrics_history['learning_rate'], label='Learning Rate', marker='o', linestyle='-', color='#d62728')
            plt.xlabel('Epoch')
            plt.ylabel('Learning Rate')
            plt.legend()
            plt.grid(True)
            
            plt.tight_layout()
            combined_plot_path = os.path.join(figures_dir, 'combined_plot.png')
            plt.savefig(combined_plot_path, dpi=300)
            plt.close()
            
            self._log_to_file(f"Generated training graphs:")
            self._log_to_file(f"  - Loss plot: {loss_plot_path}")
            self._log_to_file(f"  - Validation plot: {val_plot_path}")
            self._log_to_file(f"  - Learning rate plot: {lr_plot_path}")
            self._log_to_file(f"  - Combined plot: {combined_plot_path}")
            
            return True
        except Exception as e:
            self._log_to_file(f"Error generating training graphs: {e}")
            return False
    
    def train(self):
        """Run the training loop"""
        start_epoch = self.epoch
        total_epochs = self.config.num_epochs
        
        # Log training start
        start_message = f"Starting training from epoch {start_epoch+1}/{total_epochs}"
        print(f"\n{start_message}")
        self._log_to_file(start_message)
        self._log_to_file(f"Model: {self.config.model_name}")
        self._log_to_file(f"Device: {self.device}")
        
        # Record start time
        training_start_time = time.time()
        
        # Create progress bar for epochs
        epoch_pbar = tqdm(
            range(start_epoch, total_epochs),
            desc="Training Progress",
            unit="epoch",
            initial=start_epoch,
            total=total_epochs
        )
        
        for self.epoch in epoch_pbar:
            # Record epoch start time
            epoch_start_time = time.time()
            
            # Train for one epoch
            train_metrics, batch_times = self._train_epoch()
            
            # Validate
            val_metrics = self._validate()
            
            # Update learning rate if using cosine annealing
            if self.config.lr_schedule == 'cosine':
                self.scheduler.step()
            
            # Get current learning rate
            current_lr = self.scheduler.get_last_lr()[0]
            
            # Calculate time metrics
            epoch_time = time.time() - epoch_start_time
            total_time = time.time() - training_start_time
            avg_batch_time = np.mean(batch_times) * 1000 if batch_times else 0  # Convert to ms
            
            # Calculate ETA
            epochs_remaining = total_epochs - (self.epoch + 1)
            eta_seconds = epochs_remaining * epoch_time
            eta_hours = eta_seconds / 3600
            
            # Get GPU memory usage if available
            memory_used = 0
            if torch.cuda.is_available():
                memory_used = torch.cuda.max_memory_allocated() / (1024 * 1024)  # Convert to MB
                torch.cuda.reset_peak_memory_stats()
            
            # Log metrics
            self._log_metrics(
                self.epoch + 1, 
                train_metrics, 
                val_metrics, 
                current_lr,
                total_time,
                memory_used,
                avg_batch_time,
                eta_hours
            )
            self._log_epoch(train_metrics, val_metrics, epoch_time, memory_used, avg_batch_time, eta_hours)
            
            # Check if this is the best model
            is_best = val_metrics['accuracy'] > self.best_acc
            if is_best:
                self.best_acc = val_metrics['accuracy']
                self._log_to_file(f"New best validation accuracy: {self.best_acc:.4f}%")
            
            # Save checkpoint
            if (self.epoch + 1) % self.config.save_interval == 0 or is_best:
                self._save_checkpoint(is_best)
        
        # Calculate total training time
        total_training_time = time.time() - training_start_time
        hours, remainder = divmod(total_training_time, 3600)
        minutes, seconds = divmod(remainder, 60)
        
        # Generate training graphs
        self._generate_training_graphs()
        
        # Log training completion
        completion_message = (
            f"Training completed in {int(hours)}h {int(minutes)}m {int(seconds)}s. "
            f"Best validation accuracy: {self.best_acc:.4f}%"
        )
        print(f"\n{completion_message}")
        self._log_to_file(completion_message)
        
        return self.best_acc
    
    def _train_epoch(self) -> Tuple[Dict[str, float], list]:
        """Train for one epoch"""
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0
        batch_times = []
        
        # Create progress bar for batches
        batch_pbar = tqdm(
            self.train_loader,
            desc="Training Batches",
            unit="batch",
            leave=False,
            position=1
        )
        
        for batch_idx, data in enumerate(batch_pbar):
            batch_start_time = time.time()
            
            # Unpack data
            inputs, targets = data
            inputs = inputs.to(self.device)
            targets = targets.to(self.device)
            
            # Apply CutMix with probability
            if self.config.use_cutmix and np.random.rand() < self.config.cutmix_prob:
                inputs, targets_a, targets_b, lam = cutmix_data(
                    inputs, targets, alpha=self.config.cutmix_alpha
                )
                
                # Forward pass
                outputs = self.model(inputs)
                
                # Compute loss with CutMix
                loss = lam * self.criterion(outputs, targets_a) + (1 - lam) * self.criterion(outputs, targets_b)
                
                # Compute accuracy for progress bar (weighted by lambda)
                _, predicted = outputs.max(1)
                correct_a = predicted.eq(targets_a).sum().item()
                correct_b = predicted.eq(targets_b).sum().item()
                correct += lam * correct_a + (1 - lam) * correct_b
            else:
                # Forward pass
                outputs = self.model(inputs)
                
                # Compute loss
                loss = self.criterion(outputs, targets)
                
                # Compute accuracy for progress bar
                _, predicted = outputs.max(1)
                correct += predicted.eq(targets).sum().item()
            
            # Update total count
            total += targets.size(0)
            
            # Backward pass and optimize
            self.optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            
            # Update learning rate for OneCycleLR
            if self.config.lr_schedule == 'onecycle':
                self.scheduler.step()
            
            # Update metrics
            total_loss += loss.item()
            
            # Calculate batch time
            batch_time = time.time() - batch_start_time
            batch_times.append(batch_time)
            
            # Update progress bar description
            current_loss = total_loss / (batch_idx + 1)
            current_acc = 100. * correct / total
            current_lr = self.scheduler.get_last_lr()[0]
            
            batch_pbar.set_description(
                f"Loss: {current_loss:.4f} Acc: {current_acc:.2f}% LR: {current_lr:.6f} "
                f"Batch time: {batch_time*1000:.1f}ms"
            )
            
            # Log batch metrics at intervals
            if (batch_idx + 1) % self.config.log_interval == 0:
                self._log_batch(batch_idx, len(self.train_loader), current_loss, current_acc, current_lr, batch_time)
            
            self.train_steps += 1
        
        return {
            'loss': total_loss / len(self.train_loader),
            'accuracy': 100. * correct / total
        }, batch_times
    
    def _validate(self) -> Dict[str, float]:
        """Validate the model"""
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0
        
        # Create progress bar for validation
        val_pbar = tqdm(
            self.val_loader,
            desc="Validating",
            unit="batch",
            leave=False,
            position=1
        )
        
        with torch.no_grad():
            for data in val_pbar:
                inputs, targets = data
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)
                
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)
                
                total_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
                
                # Update progress bar
                current_loss = total_loss / (val_pbar.n + 1)
                current_acc = 100. * correct / total
                val_pbar.set_description(f"Val Loss: {current_loss:.4f} Val Acc: {current_acc:.2f}%")
        
        return {
            'loss': total_loss / len(self.val_loader),
            'accuracy': 100. * correct / total
        }
    
    def _save_checkpoint(self, is_best: bool = False):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': self.epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_acc': self.best_acc,
            'train_steps': self.train_steps,
            'config': self.config,
            'metrics_history': self.metrics_history
        }
        
        # Save regular checkpoint in run directory
        filename = f"epoch_{self.epoch+1}.pth"
        path = os.path.join(self.checkpoints_dir, filename)
        torch.save(checkpoint, path)
        
        checkpoint_message = f"Saved checkpoint at epoch {self.epoch+1} to {path}"
        self._log_to_file(checkpoint_message)
        
        # Save best model for this run
        if is_best:
            # Save in run directory
            run_best_path = os.path.join(self.run_dir, "best.pth")
            torch.save(checkpoint, run_best_path)
            
            # Save in best_models directory
            global_best_path = os.path.join(self.best_models_dir, f"{self.config.experiment_name}_best.pth")
            torch.save(checkpoint, global_best_path)
            
            best_model_message = f"New best model! Accuracy: {self.best_acc:.4f}%"
            print(f"\n{best_model_message}")
            self._log_to_file(best_model_message)
            
            paths_message = f"Saved to:\n  {run_best_path}\n  {global_best_path}"
            print(paths_message)
            self._log_to_file(paths_message)
    
    def _load_checkpoint(self, checkpoint_path: str):
        """Load model checkpoint"""
        load_message = f"Loading checkpoint from {checkpoint_path}"
        print(load_message)
        self._log_to_file(load_message)
        
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.epoch = checkpoint['epoch'] + 1  # Start from next epoch
        self.best_acc = checkpoint['best_acc']
        self.train_steps = checkpoint['train_steps']
        
        # Load metrics history if available
        if 'metrics_history' in checkpoint:
            self.metrics_history = checkpoint['metrics_history']
        
        resume_message = f"Resuming from epoch {self.epoch} with best accuracy {self.best_acc:.4f}%"
        print(resume_message)
        self._log_to_file(resume_message)
    
    def _log_batch(self, batch_idx: int, num_batches: int, loss: float, accuracy: float, lr: float, batch_time: float):
        """Log batch metrics"""
        message = (
            f"Epoch: {self.epoch+1} Batch: [{batch_idx+1}/{num_batches}] "
            f"Loss: {loss:.4f} Acc: {accuracy:.2f}% "
            f"LR: {lr:.6f} Batch time: {batch_time*1000:.1f}ms"
        )
        self._log_to_file(message)
    
    def _log_epoch(self, train_metrics: Dict[str, float], val_metrics: Dict[str, float], 
                  epoch_time: float, memory_used: float, batch_time: float, eta_hours: float):
        """Log epoch metrics"""
        # Format time
        minutes, seconds = divmod(epoch_time, 60)
        
        message = (
            f"\n{'='*40} Epoch {self.epoch+1}/{self.config.num_epochs} {'='*40}"
            f"\nTime: {int(minutes)}m {int(seconds)}s"
            f"\nTrain Loss: {train_metrics['loss']:.4f} "
            f"Train Acc: {train_metrics['accuracy']:.4f}%"
            f"\nVal Loss: {val_metrics['loss']:.4f} "
            f"Val Acc: {val_metrics['accuracy']:.4f}%"
            f"\nBest Val Acc: {self.best_acc:.4f}%"
            f"\nLearning Rate: {self.scheduler.get_last_lr()[0]:.6f}"
        )
        
        # Add GPU memory usage if available
        if memory_used > 0:
            message += f"\nGPU Memory: {memory_used:.2f} MB"
        
        # Add batch time and ETA
        message += (
            f"\nAvg Batch Time: {batch_time:.1f} ms"
            f"\nETA: {eta_hours:.2f} hours"
            f"\n{'='*90}\n"
        )
        
        print(message)
        self._log_to_file(message)
    
    def evaluate(self, test_loader: DataLoader) -> Dict[str, float]:
        """Evaluate model on test set"""
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0
        
        # Create progress bar for test evaluation
        test_pbar = tqdm(
            test_loader,
            desc="Testing",
            unit="batch"
        )
        
        with torch.no_grad():
            for data in test_pbar:
                inputs, targets = data
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)
                
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)
                
                total_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
                
                # Update progress bar
                current_loss = total_loss / (test_pbar.n + 1)
                current_acc = 100. * correct / total
                test_pbar.set_description(f"Test Loss: {current_loss:.4f} Test Acc: {current_acc:.2f}%")
        
        test_loss = total_loss / len(test_loader)
        test_acc = 100. * correct / total
        
        # Log test results
        test_message = (
            f"\n{'='*40} Test Results {'='*40}"
            f"\nTest Loss: {test_loss:.4f}"
            f"\nTest Accuracy: {test_acc:.4f}%"
            f"\n{'='*90}\n"
        )
        print(test_message)
        self._log_to_file(test_message)
        
        return {
            'loss': test_loss,
            'accuracy': test_acc
        } 