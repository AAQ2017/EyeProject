import torch
from torch import nn
from torch.nn import functional as F
import copy

class _SegmentationModel(nn.Module):
    def __init__(self, backbone, classifier):
        super(_SegmentationModel, self).__init__()
        self.backbone = backbone
        self.classifier = classifier

    def forward(self, x):
        input_shape = x.shape[-2:]
        x = self.backbone(x)
        x = self.classifier(x)
        x = F.interpolate(x, size=input_shape, mode='bilinear', align_corners=False)
        return x
