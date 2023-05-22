""" DeepLabv3 Model download and change the head for your prediction"""
from torch import nn
from models.segmentation.conresnet import ConResNet

def createConResnet(outputchannels=2):

    model = ConResNet((8, 256, 256), num_classes=outputchannels, weight_std=True)

    for param in model.parameters():
        param.requires_grad = True

    # Set the model in training mode
    model.train()
    return model
