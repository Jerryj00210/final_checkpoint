import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torch.nn as nn
from torch.nn import Dropout
import torch.nn.functional as F
import torch.optim as optim
from torchvision import models
from tqdm.notebook import tqdm

def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False
            
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        model_ft = models.resnet18(pretrained=True)
        set_parameter_requires_grad(model_ft, True)
        self.features = nn.Sequential(
            *list(model_ft.children())[:6]
        )
        self.conv1 = nn.Conv2d(in_channels = 128, out_channels = 128, kernel_size = 3)
        self.conv2 = nn.Conv2d(in_channels = 128, out_channels = 128, kernel_size = 3)
        self.conv3 = nn.Conv2d(in_channels = 128, out_channels = 64, kernel_size = 3)
        self.conv4 = nn.Conv2d(in_channels = 64, out_channels = 64, kernel_size = 3)
        self.bh1 = nn.BatchNorm2d(num_features = 128)
        self.bh2 = nn.BatchNorm2d(num_features = 64)
        self.maxpool = nn.MaxPool2d(2, 2)
        self.dropout = Dropout(p = 0.5)
        self.bh3 = nn.BatchNorm1d(num_features = 500)
        self.lin1 = nn.Linear(in_features = 2304, out_features =500)
        self.lin2 = nn.Linear(in_features = 500, out_features = 1)

    def forward(self, x):
        x = self.features(x)
        x = self.bh1(F.relu(self.conv1(x)))
        x = self.maxpool(x)
        x = self.bh1(F.relu(self.conv2(x)))
        x = self.bh2(F.relu(self.conv3(x)))
        x = self.bh2(F.relu(self.conv4(x)))
        x = self.maxpool(x)
        x = torch.flatten(x, start_dim = 1)
        x = self.dropout(x)
        x = self.bh3(F.relu(self.lin1(x)))
        x = self.lin2(x)  
        return x