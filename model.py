import torch
import torch.nn as nn
from torchvision import models

from dataset import N_TABULAR


class HybridMRIClassifier(nn.Module):
    """ResNet-18 CNN + tabular DICOM metadata fusion model."""

    def __init__(self, n_classes=4, tabular_dim=64, freeze_early=True):
        super().__init__()

        # CNN branch
        self.cnn = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        if freeze_early:
            for name, param in self.cnn.named_parameters():
                if not name.startswith(("layer3", "layer4", "fc")):
                    param.requires_grad = False
        cnn_out_dim = self.cnn.fc.in_features
        self.cnn.fc = nn.Identity()

        # Tabular branch
        self.tabular_branch = nn.Sequential(
            nn.Linear(N_TABULAR, tabular_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
        )

        # Classifier head
        self.classifier = nn.Sequential(
            nn.Linear(cnn_out_dim + tabular_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, n_classes),
        )

    def forward(self, image, tabular, tabular_mask):
        img_features = self.cnn(image)
        tab_features = self.tabular_branch(tabular * tabular_mask)
        combined = torch.cat([img_features, tab_features], dim=1)
        return self.classifier(combined)

    def unfreeze_all(self):
        """Unfreeze all CNN layers for full fine-tuning."""
        for param in self.cnn.parameters():
            param.requires_grad = True
