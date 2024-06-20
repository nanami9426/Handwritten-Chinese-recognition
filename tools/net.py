import torch.nn as nn
import torch
from torchvision.models import ResNet101_Weights
import torchvision
import torch.nn.functional as F


class HWNet(nn.Module):
    def __init__(self,num_lables):
        super().__init__()
        self.conv1x1 = nn.Conv2d(in_channels=1,out_channels=3,kernel_size=1,stride=1,padding=0,bias=False)
        self.num_labels = num_lables
        resnet101 = torchvision.models.resnet101(weights=ResNet101_Weights.IMAGENET1K_V2)
        self.res101 = nn.Sequential(*list(resnet101.children())[:-1])
        self.dense = nn.Linear(in_features=2048,out_features=self.num_labels)
        
    def forward(self,x):
        x = self.conv1x1(x)
        x = self.res101(x).reshape(len(x),-1)
        x = self.dense(x)
        return x
    


if __name__ == '__main__':
    from dataset import get_data_loader
    loader,num_labels = get_data_loader('../data_for_test/train',2,True)
    # loader,num_labels = get_data_loader('../data_for_test/test',2,True)
    net = HWNet(num_labels)
    # with torch.no_grad():
    #     for x,y in loader:
    #         print(net(x))
    #         print(y)
    #         break
    print(len(loader.dataset))
    print(len(loader))
    print("endtest")