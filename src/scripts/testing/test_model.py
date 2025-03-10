import os
import sys
import torch

# Add src directory to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from models import ModelFactory

def test_model():
    """Test that the model can be created and forward pass works"""
    # Create a small ResNet model
    model = ModelFactory.create('resnet_small')
    
    # Create a random input tensor
    batch_size = 2
    channels = 3
    height = 32
    width = 32
    x = torch.randn(batch_size, channels, height, width)
    
    # Run forward pass
    output = model(x)
    
    # Check output shape
    assert output.shape == (batch_size, 10), f"Expected output shape (2, 10), got {output.shape}"
    
    print("Model test passed!")

if __name__ == "__main__":
    test_model() 