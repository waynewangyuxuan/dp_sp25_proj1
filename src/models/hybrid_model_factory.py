from typing import Dict, Any, Type, Callable
from .base_model import BaseModel
from .hybrid_resnet import hybrid_resnet

class HybridModelFactory:
    """Factory class for creating hybrid models"""
    
    _models: Dict[str, Callable[..., BaseModel]] = {
        'hybrid_resnet': hybrid_resnet,
    }
    
    @classmethod
    def create(cls, model_name: str, **kwargs) -> BaseModel:
        """Create a model instance
        
        Args:
            model_name: Name of the model to create
            **kwargs: Additional arguments to pass to the model constructor
            
        Returns:
            Instance of the specified model
            
        Raises:
            ValueError: If model_name is not recognized
        """
        if model_name not in cls._models:
            raise ValueError(
                f"Unknown model: {model_name}. "
                f"Available models: {list(cls._models.keys())}"
            )
        
        model_fn = cls._models[model_name]
        return model_fn(**kwargs)
    
    @classmethod
    def get_available_models(cls) -> Dict[str, Callable[..., BaseModel]]:
        """Get dictionary of available models"""
        return cls._models.copy()
    
    @classmethod
    def register_model(cls, name: str, model_fn: Callable[..., BaseModel]):
        """Register a new model
        
        Args:
            name: Name to register the model under
            model_fn: Function that creates a model instance
        """
        cls._models[name] = model_fn 