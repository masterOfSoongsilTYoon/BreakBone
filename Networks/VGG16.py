from torchvision.models import vgg, VGG16_BN_Weights
from torch import nn
import torch

class VGG16(nn.Module):
    
    def __init__(self):
        super(VGG16, self).__init__()
        self.net = vgg.vgg16_bn()
        self.encoder =  nn.Sequential(*(list(self.net.modules())[3:-9]))
        self.initial_encoder = nn.Sequential(nn.Conv2d(1, 64, 3, 1, 1, 1))
        self.fc = nn.Sequential(
                                nn.AdaptiveAvgPool2d((7,7)),
                                nn.Flatten(),
                                nn.Linear(25088, 4096),
                                nn.ReLU(inplace=True),
                                nn.Dropout(p=0.5, inplace=False),
                                nn.Linear(4096,4096, bias=True),
                                nn.ReLU(inplace=True),
                                nn.Dropout(p=0.5, inplace=False),
                                nn.Linear(4096,1000, bias=True),
                                nn.Linear(1000, 1))
        self.final_encoder = nn.Sequential(self.initial_encoder, self.encoder, self.fc)
        print(self.final_encoder)

    def forward(self, x):
        x=self.final_encoder(x)
        return x