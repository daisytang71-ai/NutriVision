"""A small custom architecture; no pretrained weights or torchvision dependency."""
import torch
from torch import nn


class CustomCNN(nn.Module):
    def __init__(self, dropout=0.3):
        super().__init__()
        blocks = []
        for cin, cout in ((3, 32), (32, 64), (64, 128)):
            blocks.extend([
                nn.Conv2d(cin, cout, 3, padding=1, bias=False),
                nn.BatchNorm2d(cout), nn.ReLU(inplace=True), nn.MaxPool2d(2),
            ])
        self.features = nn.Sequential(*blocks, nn.AdaptiveAvgPool2d(1))
        self.head = nn.Sequential(nn.Flatten(), nn.Linear(128, 64),
                                  nn.ReLU(inplace=True), nn.Dropout(dropout), nn.Linear(64, 4))
        self.apply(self._init)

    @staticmethod
    def _init(module):
        if isinstance(module, nn.Conv2d):
            nn.init.kaiming_normal_(module.weight, nonlinearity="relu")
        elif isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            nn.init.zeros_(module.bias)

    def forward(self, x):
        return self.head(self.features(x))  # linear output in standardized label space
