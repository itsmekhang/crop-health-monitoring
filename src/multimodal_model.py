"""
Multimodal disease classifier: fuses image (ResNet-18 backbone) with
weather tabular features (temperature, humidity, days_since_rain) inside
the neural network — not as a post-hoc rule.

Architecture
------------
Image  → ResNet-18 (frozen backbone) → 512-d embedding ─┐
                                                          ├→ fusion MLP → num_classes
Weather → [temp, humidity, rain]     →  64-d MLP ────────┘
"""

import torch
import torch.nn as nn
from torchvision import models

# Weather feature stats for normalization (reasonable field ranges)
WEATHER_MEAN = torch.tensor([22.0, 65.0, 5.0])   # temp°C, humidity%, days_since_rain
WEATHER_STD  = torch.tensor([10.0, 20.0, 4.0])


def normalize_weather(weather: torch.Tensor) -> torch.Tensor:
    mean = WEATHER_MEAN.to(weather.device)
    std  = WEATHER_STD.to(weather.device)
    return (weather - mean) / std


class MultimodalCropNet(nn.Module):
    def __init__(self, num_classes: int, freeze_backbone: bool = True):
        super().__init__()

        # ── Image branch ─────────────────────────────────────────────────────
        backbone = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        # Remove the final FC layer — we want the 512-d feature vector
        self.image_encoder = nn.Sequential(*list(backbone.children())[:-1])
        if freeze_backbone:
            for p in self.image_encoder.parameters():
                p.requires_grad = False

        # ── Weather branch ────────────────────────────────────────────────────
        self.weather_encoder = nn.Sequential(
            nn.Linear(3, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Linear(32, 64),
            nn.ReLU(),
        )

        # ── Fusion classifier ─────────────────────────────────────────────────
        self.classifier = nn.Sequential(
            nn.Linear(512 + 64, 256),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(256, num_classes),
        )

    def forward(self, image: torch.Tensor, weather: torch.Tensor) -> torch.Tensor:
        """
        image:   (B, 3, 224, 224)
        weather: (B, 3)  — [temperature, humidity, days_since_rain], raw values
        """
        img_feat = self.image_encoder(image)         # (B, 512, 1, 1)
        img_feat = img_feat.flatten(1)               # (B, 512)

        w = normalize_weather(weather)
        weather_feat = self.weather_encoder(w)       # (B, 64)

        fused = torch.cat([img_feat, weather_feat], dim=1)  # (B, 576)
        return self.classifier(fused)
