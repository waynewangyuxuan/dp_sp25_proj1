import os
import sys
import time
import numpy as np
from typing import Dict, Optional, Tuple
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import _LRScheduler, OneCycleLR
from tqdm import tqdm

# Add src directory to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models import BaseModel
from configs.train_config import TrainingConfig
from data.augmentations import cutmix_data

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
                 config: TrainingConfig):
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
        self.best_models_dir = os.path.join(self.config.output_dir, "best_models")
        
        # Create directories
        os.makedirs(self.run_dir, exist_ok=True)
        os.makedirs(self.checkpoints_dir, exist_ok=True)
        os.makedirs(self.best_models_dir, exist_ok=True)
    
    def train(self):
        """Run the training loop"""
        start_epoch = self.epoch
        total_epochs = self.config.num_epochs
        
        # Create progress bar for epochs
        epoch_pbar = tqdm(
            range(start_epoch, total_epochs),
            desc="Training Progress",
            unit="epoch",
            initial=start_epoch,
            total=total_epochs,
            position=0,
            leave=True
        )
        
        for epoch in epoch_pbar:
            self.epoch = epoch
            
            # Training phase
            train_metrics = self._train_epoch()
            
            # Validation phase
            val_metrics = self._validate()
            
            # Update learning rate
            if self.config.lr_schedule != 'onecycle':
                self.scheduler.step()
            
            # Save checkpoint
            if (epoch + 1) % self.config.save_interval == 0:
                self._save_checkpoint()
            
            # Save best model
            if val_metrics['accuracy'] > self.best_acc:
                self.best_acc = val_metrics['accuracy']
                self._save_checkpoint(is_best=True)
            
            # Update progress bar description with metrics
            desc = (f"Epoch {epoch}/{total_epochs-1} "
                   f"Train Loss: {train_metrics['loss']:.4f} "
                   f"Train Acc: {train_metrics['accuracy']:.2f}% "
                   f"Val Loss: {val_metrics['loss']:.4f} "
                   f"Val Acc: {val_metrics['accuracy']:.2f}% "
                   f"LR: {self.scheduler.get_last_lr()[0]:.6f}")
            epoch_pbar.set_description(desc)
    
    def _train_epoch(self) -> Dict[str, float]:
        """Train for one epoch"""
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        # Create progress bar for batches
        batch_pbar = tqdm(
            self.train_loader,
            desc="Training Batches",
            unit="batch",
            leave=False,
            position=1
        )
        
        for batch_idx, data in enumerate(batch_pbar):
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
            
            # Update progress bar description
            current_loss = total_loss / (batch_idx + 1)
            current_acc = 100. * correct / total
            batch_pbar.set_description(
                f"Training Loss: {current_loss:.4f} Acc: {current_acc:.2f}%"
            )
            
            self.train_steps += 1
        
        return {
            'loss': total_loss / len(self.train_loader),
            'accuracy': 100. * correct / total
        }
    
    def _validate(self) -> Dict[str, float]:
        """Validate the model"""
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data in self.val_loader:
                inputs, targets = data
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)
                
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)
                
                total_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
        
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
            'config': self.config
        }
        
        # Save regular checkpoint in run directory
        filename = f"epoch_{self.epoch}.pth"
        path = os.path.join(self.checkpoints_dir, filename)
        torch.save(checkpoint, path)
        
        # Save best model for this run
        if is_best:
            # Save in run directory
            run_best_path = os.path.join(self.run_dir, "best.pth")
            torch.save(checkpoint, run_best_path)
            
            # Save in best_models directory
            global_best_path = os.path.join(self.best_models_dir, f"{self.config.experiment_name}_best.pth")
            torch.save(checkpoint, global_best_path)
            
            print(f"\nNew best model! Accuracy: {self.best_acc:.2f}%")
            print(f"Saved to:\n  {run_best_path}\n  {global_best_path}")
    
    def _load_checkpoint(self, checkpoint_path: str):
        """Load model checkpoint"""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.epoch = checkpoint['epoch'] + 1  # Start from next epoch
        self.best_acc = checkpoint['best_acc']
        self.train_steps = checkpoint['train_steps']
    
    def _log_batch(self, batch_idx: int, num_batches: int, loss: float, accuracy: float):
        """Log batch metrics"""
        print(f"Epoch: {self.epoch} [{batch_idx+1}/{num_batches}] "
              f"Loss: {loss:.4f} Acc: {accuracy:.2f}% "
              f"LR: {self.scheduler.get_last_lr()[0]:.6f}")
    
    def _log_epoch(self, train_metrics: Dict[str, float], val_metrics: Dict[str, float]):
        """Log epoch metrics"""
        print(f"\nEpoch: {self.epoch}")
        print(f"Train Loss: {train_metrics['loss']:.4f} "
              f"Train Acc: {train_metrics['accuracy']:.2f}%")
        print(f"Val Loss: {val_metrics['loss']:.4f} "
              f"Val Acc: {val_metrics['accuracy']:.2f}%")
        print(f"Best Val Acc: {self.best_acc:.2f}%")
        print(f"Learning Rate: {self.scheduler.get_last_lr()[0]:.6f}\n") 