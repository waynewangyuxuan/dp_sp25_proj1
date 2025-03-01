import torch
import torch.nn as nn
from typing import Dict, Any, List, Type, Union
from .base_model import BaseModel

class BasicBlock(nn.Module):
    """Basic ResNet block for ResNet-18 and ResNet-34"""
    expansion = 1

    def __init__(self, 
                 in_channels: int, 
                 out_channels: int, 
                 stride: int = 1,
                 downsample: nn.Module = None):
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x

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

class Bottleneck(nn.Module):
    """Bottleneck block for ResNet-50 and deeper"""
    expansion = 4

    def __init__(self, 
                 in_channels: int, 
                 out_channels: int, 
                 stride: int = 1,
                 downsample: nn.Module = None):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3,
                              stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv3 = nn.Conv2d(out_channels, out_channels * self.expansion,
                              kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels * self.expansion)
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

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

class ResNet(BaseModel):
    """ResNet model implementation"""

    def __init__(self, 
                 block: Type[Union[BasicBlock, Bottleneck]],
                 layers: List[int],
                 num_classes: int = 10,
                 base_channels: int = 16,  # Reduced from 64
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

        # Final classifier
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(base_channels * 8 * block.expansion, num_classes)

        # Initialize weights
        self._init_weights(zero_init_residual)

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
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x

    def get_config(self) -> Dict[str, Any]:
        """Get model configuration"""
        config = super().get_config()
        config.update({
            'num_classes': self.num_classes,
            'architecture': 'ResNet',
            'base_channels': self.base_channels
        })
        return config

def resnet18(num_classes: int = 10, **kwargs) -> ResNet:
    """ResNet-18 model"""
    return ResNet(BasicBlock, [2, 2, 2, 2], num_classes=num_classes, **kwargs)

def resnet34(num_classes: int = 10, **kwargs) -> ResNet:
    """ResNet-34 model"""
    return ResNet(BasicBlock, [3, 4, 6, 3], num_classes=num_classes, **kwargs)

def resnet50(num_classes: int = 10, **kwargs) -> ResNet:
    """ResNet-50 model"""
    return ResNet(Bottleneck, [3, 4, 6, 3], num_classes=num_classes, **kwargs)

def resnet_small(num_classes: int = 10, **kwargs) -> ResNet:
    """Small ResNet model with ~5M parameters"""
    return ResNet(
        block=BasicBlock,
        layers=[2, 2, 2, 2],  # Same as ResNet-18 but with fewer channels
        base_channels=16,  # Start with 16 channels instead of 64
        num_classes=num_classes,
        **kwargs
    ) 