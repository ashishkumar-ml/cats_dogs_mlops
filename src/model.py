import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    """Conv2d → BatchNorm → ReLU → MaxPool block."""
    def __init__(self, in_ch: int, out_ch: int, pool: bool = True):
        super().__init__()
        layers = [
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        ]
        if pool:
            layers.append(nn.MaxPool2d(2, 2))
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)


class SimpleCNN(nn.Module):
    """
    Lightweight CNN baseline for binary image classification (Cats vs Dogs).
    Input:  (B, 3, 224, 224)
    Output: (B, num_classes)
    """
    def __init__(self, num_classes: int = 2, dropout: float = 0.5):
        super().__init__()
        self.features = nn.Sequential(
            ConvBlock(3,   32),   # → 32 × 112 × 112
            ConvBlock(32,  64),   # → 64 ×  56 ×  56
            ConvBlock(64, 128),   # → 128 × 28 ×  28
            ConvBlock(128, 256),  # → 256 × 14 ×  14
            ConvBlock(256, 256),  # → 256 ×  7 ×   7
        )
        self.pool     = nn.AdaptiveAvgPool2d((4, 4))  # → 256 × 4 × 4
        self.dropout  = nn.Dropout(dropout)
        self.fc1      = nn.Linear(256 * 4 * 4, 512)
        self.fc2      = nn.Linear(512, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(F.relu(self.fc1(x)))
        return self.fc2(x)


def load_model(weights_path: str, num_classes: int = 2) -> SimpleCNN:
    """Load a saved SimpleCNN from a weights file."""
    model = SimpleCNN(num_classes=num_classes)
    state = torch.load(weights_path, map_location="cpu")
    model.load_state_dict(state)
    model.eval()
    return model
