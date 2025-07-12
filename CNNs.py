import torch
import torch.nn as nn
import torch.nn.functional as F

# Define the activation function
class Mish(nn.Module):
    def __init__(self):
        super(Mish, self).__init__()

    def forward(self, inputs):
        return inputs * torch.tanh(F.softplus(inputs))

# Defining the structure of a CNN
class CnnModel(nn.Module):
    def __init__(self, num_outputs):
        super(CnnModel, self).__init__()
        init = torch.nn.init.kaiming_normal_

        self.cnn = nn.Sequential(
            nn.Conv2d(4, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            Mish(),

            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            Mish(),

            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            Mish(),

            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            Mish(),

            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            Mish(),

            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            Mish(),

            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            Mish(),

            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            Mish(),

            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            Mish(),

            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            Mish(),

            nn.AdaptiveMaxPool2d((1, 1)),
        )

        self.fc = nn.Sequential(
            nn.Linear(256, 300),
            nn.BatchNorm1d(300),
            Mish(),

            nn.Linear(300, 100),
            nn.BatchNorm1d(100),
            Mish(),

            nn.Linear(100, 50),
            nn.BatchNorm1d(50),
            Mish(),

            nn.Linear(50, num_outputs),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.cnn(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

# Create model instance
model = CnnModel(num_outputs = 1)

# Print model structure
print(model)

