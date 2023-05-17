import torch
from torch import nn
from ._utils import _SegmentationModel
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo
from models.batchnorm import SynchronizedBatchNorm2d
import math

__all__ = ["DeepLabV3"]

class DeepLabV3(_SegmentationModel):
    """
    Arguments:
        backbone (nn.Module): the network used to compute the features for the model.
            The backbone should return an OrderedDict[Tensor], with the key being
            "out" for the last feature map used, and "aux" if an auxiliary classifier
            is used.
        classifier (nn.Module): module that takes the "out" element returned from
            the backbone and returns a dense prediction.
        aux_classifier (nn.Module, optional): auxiliary classifier used during training
    """
    pass

class DeepLabHead(nn.Sequential):
    def __init__(self, in_channels, num_classes):
        # input_shape = (512, 512, 3)
        super(DeepLabHead, self).__init__(
            ASPP(in_channels, [12, 24, 36]),
            # Decoder
            AttentionDeepLabDecoder(num_classes)
        )

class AttentionDeepLabDecoder(nn.Module):
    def __init__(self, num_classes):
        super(AttentionDeepLabDecoder, self).__init__()
        out_channels = 256
        self.decoder1 = nn.Sequential(nn.Conv2d(3*out_channels, out_channels, 3, padding=1, bias=False),
                                      nn.BatchNorm2d(out_channels),
                                      nn.ReLU())

        self.decoder2 = nn.Sequential(nn.Conv2d(2*out_channels, out_channels, 3, padding=1, bias=False),
                                      nn.BatchNorm2d(out_channels),
                                      nn.ReLU(),
                                      nn.Dropout(),
                                      nn.Conv2d(out_channels, num_classes, 1, bias=False))
        
    def forward(self, x):
        m = nn.UpsamplingBilinear2d(scale_factor=2)
        y = self.decoder1(torch.cat((x[0],  x[2]), dim=1))
        y = self.decoder2(torch.cat((m(y),  x[1]), dim=1))

        return y

class ASPPConv(nn.Sequential):
    def __init__(self, in_channels, out_channels, dilation):
        modules = [
            nn.Conv2d(in_channels, out_channels, 3, padding=dilation, dilation=dilation, bias=False),
            nn.GroupNorm(32, out_channels),
            nn.ReLU()
        ]
        super(ASPPConv, self).__init__(*modules)

class ASPP(nn.Module):
    def __init__(self, in_channels, atrous_rates):
        super(ASPP, self).__init__()
        out_channels = 256
        # attention deeplabv3+
        module_list = []
        module_list.append(nn.Sequential(nn.Conv2d(in_channels, out_channels, 1, bias=False),
                                         nn.GroupNorm(32, out_channels),
                                         nn.ReLU()))

        rate1, rate2, rate3 = tuple(atrous_rates)
        module_list.append(ASPPConv(in_channels, out_channels, rate1))
        module_list.append(ASPPConv(in_channels, out_channels, rate2))
        module_list.append(ASPPConv(in_channels, out_channels, rate3))
        module_list.append(nn.Sequential(nn.AvgPool2d((16, 16)),
                                         nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
                                         nn.GroupNorm(32, out_channels),
                                         nn.ReLU()))

        self.first_operation_list = nn.ModuleList(module_list)

        self.second_operation = nn.Sequential(nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
                                              nn.ReLU(),
                                              nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
                                              nn.ReLU(),
                                              nn.Dropout(0.5))

        self.third_operation = nn.Sequential(nn.Conv2d(2 * out_channels, out_channels, kernel_size=3, padding=1, bias=False),
                                             nn.ReLU(),
                                             nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
                                             nn.ReLU())

        self.dense_layer = nn.Sequential(nn.Linear(out_channels, out_channels, bias=False),
                                         nn.ReLU(),
                                         nn.Linear(out_channels, 32, bias=False),
                                         nn.ReLU(),
                                         nn.Linear(32, out_channels, bias=False),
                                         nn.Sigmoid())

        self.fourth_operator = nn.Sequential(nn.Conv3d(out_channels, out_channels, kernel_size=(1, 1, 5), bias=False),
                                             nn.ReLU())

        self.project = nn.Sequential(nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
                                     nn.ReLU(),
                                     nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
                                     nn.ReLU(),
                                     nn.GroupNorm(32, out_channels),
                                     nn.ReLU(),
                                     nn.Dropout(0.1))

    def forward(self, x):
        res = []
        for i in range(len(self.first_operation_list)):
            x1 = self.first_operation_list[i](x[0])
            if i == 4:
                x1 = F.interpolate(x1, size=x[0].shape[2:], mode='bilinear', align_corners=False)

            x2 = self.second_operation(x1)
            x1 = torch.cat((x1, x2), dim=1)
            x1 = self.third_operation(x1)
            # Global Average Pooling
            global_average_pooling = nn.AvgPool2d((x1.shape[2], x1.shape[3]))
            x3_dense = global_average_pooling(x1)

            x3_dense = torch.squeeze(x3_dense, 2) # Shrink dimension for the dense layer
            x3_dense = torch.squeeze(x3_dense, 2)
            x3_dense = self.dense_layer(x3_dense)
            x3_dense = x3_dense.unsqueeze(2) # Expand dimension after dense layer
            x3_dense = x3_dense.unsqueeze(3)

            # multiply
            x1 = torch.mul(x1, x3_dense)
            x1 = x1.unsqueeze(4) # Expand dimension for 3D convolution
            res.append(x1)

        res = torch.cat(res, dim=4)
        x4 = self.fourth_operator(res)
        x4 = x4.squeeze(4)
        return self.project(x4), x[1], x[2]
