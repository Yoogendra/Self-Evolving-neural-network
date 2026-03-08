from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
from evolution.dna_schema import ArchitectureDNA

_ACT = {
    "relu": lambda: nn.ReLU(inplace=True),
    "leaky_relu": lambda: nn.LeakyReLU(negative_slope=0.1, inplace=True),
}

_POOL = {
    "max": lambda: nn.MaxPool2d(2, 2),
    "avg": lambda: nn.AvgPool2d(2, 2),
    "none": lambda: nn.Identity(),
}

class DNACNN(nn.Module):
    def __init__(self, dna: ArchitectureDNA):
        super().__init__()
        in_ch = dna.input_shape[0]
        layers = []

        for b in dna.conv_blocks:
            pad = b.kernel_size // 2
            layers.append(nn.Conv2d(in_ch, b.out_channels, kernel_size=b.kernel_size, padding=pad, bias=not b.batchnorm))
            if b.batchnorm:
                layers.append(nn.BatchNorm2d(b.out_channels))
            layers.append(_ACT[b.activation]())
            if b.dropout is not None and b.dropout > 0:
                layers.append(nn.Dropout2d(p=b.dropout))
            layers.append(_POOL[b.pooling]())
            in_ch = b.out_channels

        self.features = nn.Sequential(*layers)
        self.use_gap = dna.use_gap

        self.head_dropout = None
        if dna.head.dropout is not None and dna.head.dropout > 0:
            self.head_dropout = nn.Dropout(p=dna.head.dropout)

        # Determine feature dimension deterministically (no randomness involved)
        with torch.no_grad():
            x = torch.zeros(1, *dna.input_shape)
            z = self._forward_features(x)
            feat_dim = z.shape[1]

        self.hidden = None
        if dna.head.hidden_dim is None:
            self.classifier = nn.Linear(feat_dim, dna.num_classes)
        else:
            self.hidden = nn.Linear(feat_dim, dna.head.hidden_dim)
            self.classifier = nn.Linear(dna.head.hidden_dim, dna.num_classes)

    def _forward_features(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        if self.use_gap:
            x = F.adaptive_avg_pool2d(x, 1)
            x = torch.flatten(x, 1)
        else:
            x = torch.flatten(x, 1)
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self._forward_features(x)
        if self.head_dropout is not None:
            x = self.head_dropout(x)
        if self.hidden is not None:
            x = F.relu(self.hidden(x))
        return self.classifier(x)

def build_model_from_dna(dna: ArchitectureDNA) -> nn.Module:
    return DNACNN(dna)
