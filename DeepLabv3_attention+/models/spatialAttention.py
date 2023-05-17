import torch
from torch import nn
import torch.nn.functional as F

class spatialAttention(nn.Module):
    def __init__(self, _width, _height):
        super(spatialAttention, self).__init__()
        self.dense_layer = nn.Sequential(nn.Linear((_width//4)*(_height//4), 256, bias=False),
                                         nn.ReLU(),
                                         nn.Linear(256, (_width//4)*(_height//4), bias=False),
                                         nn.ReLU(),
                                         nn.Linear((_width//4)*(_height//4), (_width//4)*(_height//4), bias=False),
                                         nn.Sigmoid())

    def forward(self, x):
        x2 = torch.mean(x, 1) # RGB -> 1 channel
        avg_pooling = nn.AvgPool2d(kernel_size=4) # size down
        x2 = avg_pooling(x2)
        x2 = torch.flatten(x2, start_dim=1) # 1D data
        x2 = self.dense_layer(x2) # 0 to 1
        nElem = torch.numel(x2) # the number of elements
        x2 = torch.reshape(x2, (nElem//(x.shape[2]//4)//(x.shape[3]//4), x.shape[2]//4, x.shape[3]//4))
        x2 = x2.unsqueeze(dim=1)
        x2 = F.interpolate(x2, size=x.shape[2:], mode='bilinear', align_corners=False)
        x2 = torch.stack((x2, x2, x2), dim=1) # 1 ch to 3 ch
        x2 = x2.squeeze(2)
        x3 = torch.mul(x, x2) # image X weight

        return x3, x2

class ASPPConv(nn.Sequential):
    def __init__(self, in_channels, out_channels, dilation):
        modules = [
            nn.Conv2d(in_channels, out_channels, 3, padding=dilation, dilation=dilation, bias=False),
            nn.GroupNorm(32, out_channels),
            nn.ReLU()
        ]
        super(ASPPConv, self).__init__(*modules)

def preAttention(_width, _height):
    out = spatialAttention(_width, _height)
    return out
