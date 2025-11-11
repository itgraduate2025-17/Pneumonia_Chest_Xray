import torch
import torch.nn as nn
from torchvision import models
from torchvision.models import MobileNet_V2_Weights


class PneumoniaModel(nn.Module):
    """
    PneumoniaModel
    ---------------
    MobileNetV2-based CNN architecture for binary classification (Normal vs Pneumonia).
    """

    def __init__(self, num_classes: int = 2, pretrained: bool = True):
        """
        Initializes the MobileNetV2 model and replaces the classifier layer.

        Args:
            num_classes (int): Number of output classes. Default = 2
            pretrained (bool): If True, loads pretrained ImageNet weights.
        """
        super(PneumoniaModel, self).__init__()

        # Load MobileNetV2 base
        weights = MobileNet_V2_Weights.DEFAULT if pretrained else None
        self.model = models.mobilenet_v2(weights=weights)

        # Replace classifier for our custom dataset
        self.model.classifier[1] = nn.Linear(self.model.last_channel, num_classes)

    def forward(self, x):
        """
        Defines forward pass.

        Args:
            x (torch.Tensor): Input tensor of shape (B, 3, 224, 224)
        Returns:
            torch.Tensor: Raw output logits of shape (B, num_classes)
        """
        return self.model(x)

    def freeze_feature_extractor(self):
        """
        Freezes feature extractor layers for transfer learning.
        Useful when retraining classifier only.
        """
        for param in self.model.features.parameters():
            param.requires_grad = False

    def unfreeze_all_layers(self):
        """
        Unfreezes all layers for full model fine-tuning.
        """
        for param in self.model.parameters():
            param.requires_grad = True
