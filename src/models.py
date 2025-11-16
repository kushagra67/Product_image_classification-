"""
Deep learning models for product image classification
"""
import torch
import torch.nn as nn
from torchvision import models
from torchvision.models import ResNet50_Weights, EfficientNet_B0_Weights


class ProductClassifier(nn.Module):
    """
    Custom CNN for product image classification
    """
    
    def __init__(self, num_classes=10, pretrained=True):
        """
        Args:
            num_classes (int): Number of product classes
            pretrained (bool): Use pretrained weights
        """
        super(ProductClassifier, self).__init__()
        self.num_classes = num_classes
        
        # Load pretrained ResNet50
        weights = ResNet50_Weights.DEFAULT if pretrained else None
        self.backbone = models.resnet50(weights=weights)
        
        # Modify final layer
        num_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(num_features, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )
    
    def forward(self, x):
        """Forward pass"""
        return self.backbone(x)
    
    def get_features(self, x):
        """Extract features before classification"""
        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)
        
        x = self.backbone.layer1(x)
        x = self.backbone.layer2(x)
        x = self.backbone.layer3(x)
        x = self.backbone.layer4(x)
        
        x = self.backbone.avgpool(x)
        x = torch.flatten(x, 1)
        return x


class EfficientProductClassifier(nn.Module):
    """
    Lightweight classifier using EfficientNet
    """
    
    def __init__(self, num_classes=10, pretrained=True):
        """
        Args:
            num_classes (int): Number of product classes
            pretrained (bool): Use pretrained weights
        """
        super(EfficientProductClassifier, self).__init__()
        self.num_classes = num_classes
        
        # Load pretrained EfficientNet
        weights = EfficientNet_B0_Weights.DEFAULT if pretrained else None
        self.backbone = models.efficientnet_b0(weights=weights)
        
        # Modify final layer
        num_features = self.backbone.classifier[1].in_features
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(num_features, 256),
            nn.ReLU(),
            nn.Linear(256, num_classes)
        )
    
    def forward(self, x):
        """Forward pass"""
        return self.backbone(x)


def create_model(model_name='resnet50', num_classes=10, pretrained=True, device='cpu'):
    """
    Factory function to create different models
    
    Args:
        model_name (str): Name of model ('resnet50', 'efficientnet_b0', etc.)
        num_classes (int): Number of classes
        pretrained (bool): Use pretrained weights
        device (str or torch.device): Device to create model on
    
    Returns:
        nn.Module: Model instance
    """
    if model_name == 'resnet50':
        model = ProductClassifier(num_classes=num_classes, pretrained=pretrained)
    elif model_name == 'efficientnet_b0':
        model = EfficientProductClassifier(num_classes=num_classes, pretrained=pretrained)
    else:
        raise ValueError(f"Unknown model: {model_name}")
    
    # Ensure device is CPU if CUDA is not available
    if isinstance(device, str):
        if device == 'cuda' and not torch.cuda.is_available():
            device = 'cpu'
        device = torch.device(device)
    
    model = model.to(device)
    return model
