
import torch
import torch.nn as nn

class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)

def mnist_net2(classes=10):
    model = nn.Sequential(
        nn.Conv2d(1, 64, 5, stride=1, padding=2),
        nn.ReLU(),
        nn.Conv2d(64, 64, 5, stride=1, padding=2),
        nn.ReLU(),
        nn.Dropout(0.25),
        Flatten(),
        nn.Linear(64 * 28 * 28, 128),
        nn.Dropout(0.5),
        nn.Linear(128, classes)
    )
    return model
