import torch
import torch.nn as nn
from typing import Dict, Any, List, Type, Union
from .base_model import BaseModel
from .stochastic_depth import StochasticDepth

def count_parameters(model: nn.Module) -> int:
    """Count the total number of trainable parameters"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

class SELayer(nn.Module):
    """Squeeze-and-Excitation block"""
    def __init__(self, channel, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

class BasicBlock(nn.Module):
    """Basic ResNet block for ResNet-18 and ResNet-34"""
    expansion = 1

    def __init__(self, 
                 in_channels: int, 
                 out_channels: int, 
                 stride: int = 1,
                 downsample: nn.Module = None,
                 use_se: bool = False,
                 se_reduction: int = 16,
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
        
        # Add Squeeze-and-Excitation block
        self.use_se = use_se
        if use_se:
            self.se = SELayer(out_channels, reduction=se_reduction)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        
        # Apply SE if enabled
        if self.use_se:
            out = self.se(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        # Apply stochastic depth if enabled
        if hasattr(self, 'stochastic_depth') and self.drop_prob > 0:
            return self.stochastic_depth(identity, out)
        else:
            out += identity
            out = self.relu(out)
            return out

class Bottleneck(nn.Module):
    """Bottleneck block for ResNet-50 and deeper"""
    expansion = 4

    def __init__(self, 
                 in_channels: int, 
                 out_channels: int, 
                 stride: int = 1,
                 downsample: nn.Module = None,
                 use_se: bool = True,
                 drop_prob: float = 0.0):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3,
                              stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv3 = nn.Conv2d(out_channels, out_channels * self.expansion,
                              kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels * self.expansion)
        self.se = SELayer(out_channels * self.expansion) if use_se else None
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)
        
        # Apply SE if enabled
        if self.use_se:
            out = self.se(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        # Apply stochastic depth if enabled
        if hasattr(self, 'stochastic_depth') and self.drop_prob > 0:
            return self.stochastic_depth(identity, out)
        else:
            out += identity
            out = self.relu(out)
            return out

class ResNet(BaseModel):
    """ResNet model implementation"""
    
    def __init__(self, 
                 block: Type[Union[BasicBlock]],
                 layers: List[int],
                 num_classes: int = 10,
                 base_channels: int = 32,
                 dropout_rate: float = 0.1,
                 use_se: bool = False,
                 se_reduction: int = 16,
                 use_stochastic_depth: bool = False,
                 stochastic_depth_prob: float = 0.2):
        super().__init__()
        self.num_classes = num_classes
        self.base_channels = base_channels
        self.dropout_rate = dropout_rate
        self.use_se = use_se
        self.se_reduction = se_reduction
        self.use_stochastic_depth = use_stochastic_depth
        self.stochastic_depth_prob = stochastic_depth_prob
        self.in_channels = base_channels  # Track input channels separately
        
        # Initial convolution
        self.conv1 = nn.Conv2d(3, self.base_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(self.base_channels)
        self.relu = nn.ReLU(inplace=True)
        
        # Residual layers
        total_blocks = sum(layers)
        self.layer1 = self._make_layer(block, self.base_channels, layers[0], 
                                      drop_path_rate=stochastic_depth_prob if use_stochastic_depth else 0.0,
                                      block_idx=0, total_blocks=total_blocks)
        self.layer2 = self._make_layer(block, self.base_channels * 2, layers[1], stride=2,
                                      drop_path_rate=stochastic_depth_prob if use_stochastic_depth else 0.0,
                                      block_idx=layers[0], total_blocks=total_blocks)
        self.layer3 = self._make_layer(block, self.base_channels * 4, layers[2], stride=2,
                                      drop_path_rate=stochastic_depth_prob if use_stochastic_depth else 0.0,
                                      block_idx=layers[0]+layers[1], total_blocks=total_blocks)
        self.layer4 = self._make_layer(block, self.base_channels * 8, layers[3], stride=2,
                                      drop_path_rate=stochastic_depth_prob if use_stochastic_depth else 0.0,
                                      block_idx=layers[0]+layers[1]+layers[2], total_blocks=total_blocks)
        
        # Classifier
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(self.dropout_rate)
        self.fc = nn.Linear(self.base_channels * 8 * block.expansion, num_classes)
        
        # Initialize weights
        self._init_weights()
    
    def _make_layer(self, 
                   block: Type[Union[BasicBlock]],
                   out_channels: int,
                   blocks: int,
                   stride: int = 1,
                   drop_path_rate: float = 0.0,
                   block_idx: int = 0,
                   total_blocks: int = 0) -> nn.Sequential:
        """Create a layer of blocks"""
        downsample = None
        if stride != 1 or self.in_channels != out_channels * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, out_channels * block.expansion,
                         kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels * block.expansion),
            )
        
        layers = []
        
        # Calculate stochastic depth probability for each block
        # The probability increases linearly with depth
        stochastic_depth_probs = []
        if drop_path_rate > 0 and total_blocks > 0:
            for i in range(blocks):
                # Calculate drop probability for this block
                # Formula: drop_prob = drop_path_rate * (block_idx + i) / (total_blocks - 1)
                curr_block_idx = block_idx + i
                drop_prob = drop_path_rate * curr_block_idx / (total_blocks - 1)
                stochastic_depth_probs.append(drop_prob)
        else:
            stochastic_depth_probs = [0.0] * blocks
        
        # First block may need downsampling
        layers.append(block(self.in_channels, out_channels, stride, downsample, 
                           use_se=self.use_se, se_reduction=self.se_reduction,
                           drop_prob=stochastic_depth_probs[0]))
        
        # Update in_channels for subsequent blocks
        self.in_channels = out_channels * block.expansion
        
        # Add remaining blocks
        for i in range(1, blocks):
            layers.append(block(self.in_channels, out_channels, 
                               use_se=self.use_se, se_reduction=self.se_reduction,
                               drop_prob=stochastic_depth_probs[i]))
        
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
            'architecture': 'ResNet',
            'base_channels': self.base_channels,
            'use_se': self.use_se,
            'se_reduction': self.se_reduction,
            'use_stochastic_depth': self.use_stochastic_depth,
            'stochastic_depth_prob': self.stochastic_depth_prob,
            'total_params': count_parameters(self)
        })
        return config

def resnet_small(num_classes: int = 10, **kwargs) -> ResNet:
    """Enhanced ResNet for CIFAR-10 with SE blocks (~4.8M parameters)"""
    model = ResNet(
        block=BasicBlock,
        layers=[2, 2, 2, 2],  # Balanced depth
        base_channels=32,     # Moderate width
        num_classes=num_classes,
        dropout_rate=0.3,     # Higher dropout for regularization
        use_se=True,          # Enable SE blocks
        se_reduction=16,      # Standard reduction ratio
        use_stochastic_depth=False,  # Disable stochastic depth by default
        stochastic_depth_prob=0.2,   # Default stochastic depth probability
        **kwargs
    )
    
    num_params = count_parameters(model)
    print(f"\nModel parameter count: {num_params:,}")
    assert num_params < 5_000_000, f"Model has {num_params:,} parameters (limit: 5M)"
    
    return model

def resnet18(num_classes: int = 10, **kwargs) -> ResNet:
    """ResNet-18 model"""
    return ResNet(BasicBlock, [2, 2, 2, 2], num_classes=num_classes, **kwargs)

def resnet34(num_classes: int = 10, **kwargs) -> ResNet:
    """ResNet-34 model"""
    return ResNet(BasicBlock, [3, 4, 6, 3], num_classes=num_classes, **kwargs)

def resnet50(num_classes: int = 10, **kwargs) -> ResNet:
    """ResNet-50 model"""
    return ResNet(Bottleneck, [3, 4, 6, 3], num_classes=num_classes, **kwargs)

def resnet101(num_classes: int = 10, **kwargs) -> ResNet:
    """ResNet-101 model"""
    return ResNet(Bottleneck, [3, 4, 23, 3], num_classes=num_classes, **kwargs)

def resnet_medium(num_classes: int = 10, **kwargs) -> ResNet:
    """Medium-sized ResNet model with ~1.1M parameters"""
    return ResNet(
        block=BasicBlock,
        layers=[3, 3, 3, 3],  # 3 blocks per stage
        base_channels=32,     # Increased from 24
        num_classes=num_classes,
        use_se=True,          # Use SE blocks
        dropout_rate=0.2,     # Add dropout
        **kwargs
    )

class ResNetWithLongSkips(BaseModel):
    """ResNet model with long skip connections"""

    def __init__(self, 
                 block: Type[Union[BasicBlock, Bottleneck]],
                 layers: List[int],
                 num_classes: int = 10,
                 base_channels: int = 24,  # Increased from 16
                 zero_init_residual: bool = False):
        """
        Args:
            block: Type of residual block to use (BasicBlock or Bottleneck)
            layers: Number of blocks in each layer
            num_classes: Number of output classes
            base_channels: Number of base channels (first layer)
            zero_init_residual: Whether to initialize residual block BN weights to 0
        """
        super().__init__()
        self.in_channels = base_channels
        self.num_classes = num_classes
        self.base_channels = base_channels

        # Initial convolution layer
        self.conv1 = nn.Conv2d(3, self.in_channels, kernel_size=3, 
                              stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(self.in_channels)
        self.relu = nn.ReLU(inplace=True)

        # Residual layers
        self.layer1 = self._make_layer(block, base_channels, layers[0])
        self.layer2 = self._make_layer(block, base_channels * 2, layers[1], stride=2)
        self.layer3 = self._make_layer(block, base_channels * 4, layers[2], stride=2)
        self.layer4 = self._make_layer(block, base_channels * 8, layers[3], stride=2)

        # Long skip connections
        self.skip1 = self._make_skip_connection(base_channels, base_channels * 2)
        self.skip2 = self._make_skip_connection(base_channels * 2, base_channels * 4)
        self.skip3 = self._make_skip_connection(base_channels * 4, base_channels * 8)

        # Final classifier
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(base_channels * 8 * block.expansion, num_classes)

        # Initialize weights
        self._init_weights(zero_init_residual)

    def _make_skip_connection(self, in_channels: int, out_channels: int) -> nn.Module:
        """Create a skip connection between layers"""
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=2, bias=False),
            nn.BatchNorm2d(out_channels)
        )

    def _make_layer(self, 
                   block: Type[Union[BasicBlock, Bottleneck]],
                   out_channels: int,
                   blocks: int,
                   stride: int = 1) -> nn.Sequential:
        downsample = None
        if stride != 1 or self.in_channels != out_channels * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, out_channels * block.expansion,
                         kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels * block.expansion),
            )

        layers = []
        layers.append(block(self.in_channels, out_channels, stride, downsample))
        self.in_channels = out_channels * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.in_channels, out_channels))

        return nn.Sequential(*layers)

    def _init_weights(self, zero_init_residual: bool = False):
        """Initialize model weights"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Initial conv
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        # Layer 1
        x1 = self.layer1(x)
        
        # Layer 2 with skip
        x2 = self.layer2(x1)
        x2 = x2 + self.skip1(x1)
        
        # Layer 3 with skip
        x3 = self.layer3(x2)
        x3 = x3 + self.skip2(x2)
        
        # Layer 4 with skip
        x4 = self.layer4(x3)
        x4 = x4 + self.skip3(x3)

        # Final classification
        x4 = self.avgpool(x4)
        x4 = torch.flatten(x4, 1)
        x4 = self.fc(x4)

        return x4

    def get_config(self) -> Dict[str, Any]:
        """Get model configuration"""
        config = super().get_config()
        config.update({
            'num_classes': self.num_classes,
            'architecture': 'ResNetWithLongSkips',
            'base_channels': self.base_channels
        })
        return config

def resnet_medium(num_classes: int = 10, **kwargs) -> ResNetWithLongSkips:
    """Medium-sized ResNet model with ~1.1M parameters and long skip connections"""
    return ResNetWithLongSkips(
        block=BasicBlock,
        layers=[3, 3, 3, 3],  # 3 blocks per stage
        base_channels=24,     # Increased from 16
        num_classes=num_classes,
        **kwargs
    ) 