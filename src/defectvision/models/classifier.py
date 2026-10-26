"""Modele supervise : ResNet-18 en transfert d'apprentissage (OK / defaut)."""

from __future__ import annotations

import torch
import torch.nn as nn
from torchvision import models


class DefectClassifier(nn.Module):
    """ResNet-18 avec une tete de classification a deux sorties (OK, defaut)."""

    def __init__(self, num_classes: int = 2, pretrained: bool = True):
        super().__init__()
        weights = models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        self.backbone = models.resnet18(weights=weights)
        num_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(num_features, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes),
        )

    def freeze_backbone(self) -> None:
        """Gele toutes les couches sauf la tete."""
        for param in self.backbone.parameters():
            param.requires_grad = False
        for param in self.backbone.fc.parameters():
            param.requires_grad = True

    def unfreeze_top_layers(self, n_layers: int = 2) -> int:
        """Degele les n derniers blocs convolutifs ; renvoie le nombre de parametres entrainables."""
        self.freeze_backbone()
        blocks = [self.backbone.layer1, self.backbone.layer2, self.backbone.layer3, self.backbone.layer4]
        for block in blocks[len(blocks) - n_layers :] if n_layers > 0 else []:
            for param in block.parameters():
                param.requires_grad = True
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)

    @torch.no_grad()
    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """Probabilite de defaut pour chaque image du lot."""
        return torch.softmax(self(x), dim=1)[:, 1]
