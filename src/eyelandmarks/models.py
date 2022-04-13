from torch import nn
from torch import optim

import torchmetrics


class GazeNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        # Input is 128x128
        self.conv = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=5, padding="same"),
            nn.ReLU(),
            nn.BatchNorm2d(8),
            nn.MaxPool2d(2),
            nn.Conv2d(8, 16, kernel_size=5, padding="same"),
            nn.ReLU(),
            nn.BatchNorm2d(16),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding="same"),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding="valid"),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding="valid"),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.MaxPool2d(2),
            nn.MaxPool2d(2),
        )

        self.linear = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, 64),
            nn.LeakyReLU(),
            nn.Linear(64, 3),
            nn.Tanh(),
        )

    def forward(self, x):
        return self.linear(self.conv(x))


class PupilLandmarkNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()

        # Input is 128x128
        self.conv = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=5, padding="same"),
            nn.ReLU(),
            nn.BatchNorm2d(8),
            nn.MaxPool2d(2),
            nn.Conv2d(8, 16, kernel_size=5, padding="same"),
            nn.ReLU(),
            nn.BatchNorm2d(16),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding="same"),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding="valid"),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding="valid"),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.MaxPool2d(2),
            nn.MaxPool2d(2),
        )

        self.linear = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, 64),
            nn.LeakyReLU(),
            nn.Linear(64, 16),
            nn.Tanh(),
        )


class ManyLandmarks(nn.Module):
    def __init__(self) -> None:
        super().__init__()

        # Input is 128x128
        self.conv = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=5, padding="same"),
            nn.ReLU(),
            nn.BatchNorm2d(8),
            nn.MaxPool2d(2),
            nn.Conv2d(8, 16, kernel_size=5, padding="same"),
            nn.ReLU(),
            nn.BatchNorm2d(16),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding="same"),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding="valid"),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding="valid"),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.MaxPool2d(2),
            nn.MaxPool2d(2),
        )

        self.linear = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, 512),
            nn.LeakyReLU(),
            nn.Linear(512, (8 + 8 + 34) * 2),
            nn.Tanh(),
        )

    def forward(self, x):
        return self.linear(self.conv(x))
