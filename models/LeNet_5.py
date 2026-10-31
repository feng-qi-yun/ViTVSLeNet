import torch
import torch.nn as nn

class LeNet(nn.Module):
    def __init__(self, num_classes=10, dropout=0.3):
        super(LeNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 6, kernel_size=5), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(6, 16, kernel_size=5), nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(16 * 4 * 4, 120), nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(120, 84), nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(84, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x
