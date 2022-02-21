import numpy as np
from torch.autograd import Variable
import torch.nn as nn
import torch

cuda = True if torch.cuda.is_available() else False
Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

class Classifier(nn.Module):
    def __init__(self, n_classes):
        super().__init__()
        self.conv_block = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm2d(32),
            nn.ELU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, padding=1, stride=2),
            nn.BatchNorm2d(64),
            nn.ELU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, padding=1, stride=2),
            nn.BatchNorm2d(128),
            nn.ELU(inplace=True),
        )

        feature_map_shape = (128, 8, 8)

        self.linear_block = nn.Sequential(
            nn.Linear(np.prod(feature_map_shape), n_classes),
            nn.Softmax(dim=1)
        )

    def forward(self, imgs):
        x = self.conv_block(imgs)
        return self.linear_block(torch.flatten(x, 1))


