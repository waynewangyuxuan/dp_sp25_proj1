import torch
from models import ModelFactory

def test_model(model_name: str = 'resnet18'):
    """Test the model implementation
    
    Args:
        model_name: Name of the model to test
    """
    # Create a model
    model = ModelFactory.create(model_name)
    print(f"\nTesting {model_name}:")
    print("\nModel configuration:")
    for key, value in model.get_config().items():
        print(f"{key}: {value}")
    
    # Test forward pass
    batch_size = 4
    x = torch.randn(batch_size, 3, 32, 32)
    y = torch.randint(0, 10, (batch_size,))
    
    # Test forward pass
    outputs = model(x)
    print(f"\nForward pass:")
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {outputs.shape}")
    
    # Test loss calculation
    loss_dict = model.get_loss((x, y))
    print("\nLoss calculation:")
    for key, value in loss_dict.items():
        print(f"{key}: {value.item():.4f}")
    
    # Test prediction
    predictions = model.predict(x)
    print(f"\nPredictions:")
    print(f"Shape: {predictions.shape}")
    print(f"Values: {predictions}")

def main():
    """Test all available models"""
    available_models = ModelFactory.get_available_models()
    print(f"\nAvailable models: {list(available_models.keys())}")
    
    # Test each model
    for model_name in available_models:
        test_model(model_name)

if __name__ == '__main__':
    main() 