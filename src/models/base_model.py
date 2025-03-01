from abc import ABC, abstractmethod
import torch
import torch.nn as nn
from typing import Dict, Any, Tuple

class BaseModel(nn.Module, ABC):
    """Base class for all models"""
    
    def __init__(self):
        super().__init__()
        
    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the model
        
        Args:
            x: Input tensor of shape [batch_size, channels, height, width]
            
        Returns:
            Output tensor of shape [batch_size, num_classes]
        """
        pass
    
    def get_loss(self, batch: Tuple[torch.Tensor, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Calculate loss for a batch of data
        
        Args:
            batch: Tuple of (inputs, targets)
            
        Returns:
            Dictionary containing loss values and any other metrics
        """
        inputs, targets = batch
        outputs = self(inputs)
        loss = nn.CrossEntropyLoss()(outputs, targets)
        
        # Calculate accuracy
        _, predicted = torch.max(outputs.data, 1)
        correct = (predicted == targets).sum().item()
        total = targets.size(0)
        accuracy = 100. * correct / total
        
        return {
            'loss': loss,
            'accuracy': torch.tensor(accuracy)
        }
    
    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """Make predictions for input data
        
        Args:
            x: Input tensor of shape [batch_size, channels, height, width]
            
        Returns:
            Predicted class indices of shape [batch_size]
        """
        self.eval()
        with torch.no_grad():
            outputs = self(x)
            _, predicted = torch.max(outputs.data, 1)
        return predicted
    
    @property
    def device(self) -> torch.device:
        """Get the device where model parameters are allocated"""
        return next(self.parameters()).device
    
    def get_config(self) -> Dict[str, Any]:
        """Get model configuration
        
        Returns:
            Dictionary containing model configuration
        """
        return {
            'model_name': self.__class__.__name__,
            'num_parameters': sum(p.numel() for p in self.parameters())
        } 