import torch
import torch.nn as nn
from typing import Type, Union, List, Dict, Any

from .base_model import BaseModel
from .resnet import count_parameters

class BasicBlock(nn.Module):
    """Basic ResNet block with stochastic depth"""
    expansion = 1

    def __init__(self, 
                 in_channels: int, 
                 out_channels: int, 
                 stride: int = 1,
                 downsample: nn.Module = None,
                 drop_prob: float = 0.0):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, 
                              stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3,
                              stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample
        self.stride = stride
        self.drop_prob = drop_prob

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        
        # Stochastic depth: randomly drop the entire block during training
        if self.training and self.drop_prob > 0 and torch.rand(1).item() < self.drop_prob:
            return identity

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

class StochasticResNet(BaseModel):
    """ResNet with stochastic depth"""
    
    def __init__(self, 
                 block: Type[Union[BasicBlock]],
                 layers: List[int],
                 num_classes: int = 10,
                 base_channels: int = 32,
                 dropout_rate: float = 0.3,
                 stochastic_depth_prob: float = 0.2):
        super().__init__()
        self.num_classes = num_classes
        self.base_channels = base_channels
        self.dropout_rate = dropout_rate
        self.stochastic_depth_prob = stochastic_depth_prob
        self.in_channels = base_channels  # Initialize in_channels
        
        # Initial convolution
        self.conv1 = nn.Conv2d(3, base_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(base_channels)
        self.relu = nn.ReLU(inplace=True)
        
        # Calculate total blocks for stochastic depth
        self.total_blocks = sum(layers)
        self.current_block = 0
        
        # Residual layers
        self.layer1 = self._make_layer(block, base_channels, layers[0], stride=1)
        self.layer2 = self._make_layer(block, base_channels*2, layers[1], stride=2)
        self.layer3 = self._make_layer(block, base_channels*4, layers[2], stride=2)
        self.layer4 = self._make_layer(block, base_channels*8, layers[3], stride=2)
        
        # Get the final number of channels
        final_channels = base_channels * 8 * block.expansion
        
        # Classifier
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(self.dropout_rate)
        self.fc = nn.Linear(final_channels, num_classes)
        
        # Initialize weights
        self._init_weights()
    
    def _make_layer(self, block, channels, blocks, stride=1):
        downsample = None
        
        # For the first layer of the network
        if self.current_block == 0:
            in_channels = self.base_channels
        # For the first layer of subsequent stages
        elif stride != 1:
            in_channels = self.in_channels
        # For other layers
        else:
            in_channels = self.in_channels
        
        # Create downsample if needed
        if stride != 1 or in_channels != channels * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(in_channels, channels * block.expansion, 
                         kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(channels * block.expansion),
            )
        
        layers = []
        
        # Calculate drop probability for first block in this layer
        # Linearly increase drop probability from 0 to stochastic_depth_prob
        drop_prob = 0.0
        if self.stochastic_depth_prob > 0 and self.total_blocks > 1:
            drop_prob = self.stochastic_depth_prob * self.current_block / (self.total_blocks - 1)
        
        # First block (with potential downsampling)
        layers.append(block(in_channels, channels, stride, downsample, drop_prob=drop_prob))
        self.current_block += 1
        
        # Update in_channels for subsequent blocks
        self.in_channels = channels * block.expansion
        
        # Remaining blocks in this layer
        for _ in range(1, blocks):
            # Calculate drop probability
            drop_prob = 0.0
            if self.stochastic_depth_prob > 0 and self.total_blocks > 1:
                drop_prob = self.stochastic_depth_prob * self.current_block / (self.total_blocks - 1)
            
            layers.append(block(channels * block.expansion, channels, drop_prob=drop_prob))
            self.current_block += 1
        
        return nn.Sequential(*layers)
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        x = self.fc(x)
        
        return x
    
    def get_config(self) -> Dict[str, Any]:
        """Get model configuration"""
        config = super().get_config()
        config.update({
            'num_classes': self.num_classes,
            'architecture': 'StochasticResNet',
            'base_channels': self.base_channels,
            'stochastic_depth_prob': self.stochastic_depth_prob
        })
        return config

def stochastic_resnet(num_classes: int = 10, **kwargs) -> StochasticResNet:
    """ResNet with stochastic depth for CIFAR-10 (~4.8M parameters)"""
    model = StochasticResNet(
        block=BasicBlock,
        layers=[2, 2, 2, 2],  # Balanced depth
        base_channels=32,     # Moderate width
        num_classes=num_classes,
        dropout_rate=0.3,     # Higher dropout for regularization
        stochastic_depth_prob=0.2,  # Enable stochastic depth
        **kwargs
    )
    
    num_params = count_parameters(model)
    print(f"\nModel parameter count: {num_params:,}")
    assert num_params < 5_000_000, f"Model has {num_params:,} parameters (limit: 5M)"
    
    return model 