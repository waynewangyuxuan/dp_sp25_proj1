import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any

from .base_model import BaseModel
from .resnet import count_parameters, SELayer

class BasicBlock(nn.Module):
    """Basic ResNet block with SE and stochastic depth options"""
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1, downsample=None, 
                 use_se=True, drop_path_rate=0.0):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, 
                               stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.use_se = use_se
        self.drop_path_rate = drop_path_rate
        
        if use_se:
            self.se = SELayer(out_channels, reduction=8)

    def forward(self, x):
        identity = x

        # Apply stochastic depth (drop path)
        if self.training and self.drop_path_rate > 0.0 and torch.rand(1).item() < self.drop_path_rate:
            return identity

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        
        if self.use_se:
            out = self.se(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

class HybridResNet(BaseModel):
    """Simple ResNet implementation with optional SE and stochastic depth"""
    
    def __init__(self, num_classes=10, use_se=True, stochastic_depth=0.2):
        super(HybridResNet, self).__init__()
        
        self.num_classes = num_classes
        self.use_se = use_se
        self.stochastic_depth = stochastic_depth
        
        # Model parameters
        self.inplanes = 64
        
        # Initial layers
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        
        # Define stochastic depth rates
        dpr = [x.item() for x in torch.linspace(0, stochastic_depth, 8)]
        
        # ResNet layers
        self.layer1 = self._make_layer(64, 2, stride=1, drop_path_rates=dpr[0:2])
        self.layer2 = self._make_layer(128, 2, stride=2, drop_path_rates=dpr[2:4])
        self.layer3 = self._make_layer(256, 2, stride=2, drop_path_rates=dpr[4:6])
        self.layer4 = self._make_layer(256, 2, stride=1, drop_path_rates=dpr[6:8])
        
        # Pooling and classification
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(256, num_classes)
        self.dropout = nn.Dropout(0.3)  # Higher dropout for better regularization
        
        # Initialize weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
                
        # Zero-initialize the last BN in each residual branch
        for m in self.modules():
            if isinstance(m, BasicBlock):
                nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, planes, blocks, stride=1, drop_path_rates=None):
        downsample = None
        if stride != 1 or self.inplanes != planes:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes),
            )

        layers = []
        # First block may have a different stride and downsample
        layers.append(BasicBlock(self.inplanes, planes, stride, downsample, 
                                use_se=self.use_se, drop_path_rate=drop_path_rates[0]))
        self.inplanes = planes
        
        # Subsequent blocks
        for i in range(1, blocks):
            layers.append(BasicBlock(self.inplanes, planes, 1, None,
                                   use_se=self.use_se, drop_path_rate=drop_path_rates[i]))

        return nn.Sequential(*layers)

    def forward(self, x):
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
            'architecture': 'HybridResNet',
            'use_se': self.use_se,
            'stochastic_depth': self.stochastic_depth
        })
        return config

def hybrid_resnet(num_classes=10, **kwargs):
    """Create a hybrid ResNet model with SE blocks and stochastic depth"""
    model = HybridResNet(
        num_classes=num_classes,
        use_se=True,
        stochastic_depth=0.2,
        **kwargs
    )
    
    num_params = count_parameters(model)
    print(f"\nModel parameter count: {num_params:,}")
    assert num_params < 5_000_000, f"Model has {num_params:,} parameters (limit: 5M)"
    
    return model 