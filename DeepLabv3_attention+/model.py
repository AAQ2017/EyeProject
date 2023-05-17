""" DeepLabv3 Model download and change the head for your prediction"""
from models.segmentation.deeplabv3 import DeepLabHead
from torch import nn
from models.resnet import ResNet101
from models.segmentation._utils import _SegmentationModel
# from models.resnet_regress import resnet18

def createDeepLabv3(outputchannels=2):

    # resnet = resnet18(pretrained=True)
    resnet = ResNet101(pretrained=True, output_stride=8)
    ASPP = DeepLabHead(2048, outputchannels)
    model = _SegmentationModel(backbone=resnet, classifier=ASPP)

    for param in model.parameters():
        param.requires_grad = True

    # Set the model in training mode
    model.train()
    return model
