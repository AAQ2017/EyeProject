import torch
from torch import nn
from torch.nn import functional as F
import copy

class _SegmentationModel(nn.Module):
    def __init__(self, backbone, classifier=None):
        super(_SegmentationModel, self).__init__()
        self.backbone = backbone
        self.classifier = classifier
        # self.c1 = nn.Linear(1000, 256)
        # self.c2 = nn.Linear(256, 32)
        # self.c3 = nn.Linear(32, 1)

    def forward(self, x):
        input_shape = x.shape[-2:]
        x = self.backbone(x)
        # m = nn.ReLU()
        # m2 = nn.Sigmoid()
        # x = m(x)
        # x = self.c1(x)
        # x = m(x)
        # x = self.c2(x)
        # x = m(x)
        # x = self.c3(x)
        # x = m2(x)
        x = self.classifier(x)
        x = F.interpolate(x, size=input_shape, mode='bilinear', align_corners=False)
        return x
