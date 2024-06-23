import torch.nn as nn
from torchvision.models import ResNet50_Weights
import torchvision
import torch.nn.functional as F


class HWNet(nn.Module):
    def __init__(self,num_lables):
        super().__init__()
        self.conv1x1 = nn.Conv2d(in_channels=1,out_channels=3,kernel_size=1,stride=1,padding=0,bias=False)
        self.num_labels = num_lables
        resnet101 = torchvision.models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
        # resnet101 = torchvision.models.resnet101()
        self.res101 = nn.Sequential(*list(resnet101.children())[:-1])
        self.dense = nn.Linear(in_features=2048,out_features=self.num_labels)
        
    def forward(self,x):
        x = self.conv1x1(x)
        x = self.res101(x).reshape(len(x),-1)
        x = self.dense(x)
        return x
    
