"""SegmentationNN"""
import torch
import torch.nn as nn
from torchvision import models


class SegmentationNN(nn.Module):

    def __init__(self, num_classes=23):
        super(SegmentationNN, self).__init__()

        #######################################################################
        #                             YOUR CODE                               #
        ######################################################################
        self.model_fcn = models.segmentation.fcn_resnet101(pretrained=True)#.eval()
        for param in self.model_fcn.parameters():
            param.requires_grad = False
        # update number of classes from 21 to 23
        self.model_fcn.classifier[4] = nn.Conv2d(512, num_classes, kernel_size=(1, 1), stride=(1, 1))
        # self.model_fcn.classifier[4].param.requires_grad = True
        # self.upconv = nn.ConvTranspose2d(num_classes, num_classes, kernel_size=(3, 3), stride=(1, 1))
        #######################################################################
        #                           END OF YOUR CODE                          #
        #######################################################################

    def forward(self, x):
        """
        Forward pass of the convolutional neural network. Should not be called
        manually but by calling a model instance directly.

        Inputs:
        - x: PyTorch input Variable
        """
        #######################################################################
        #                             YOUR CODE                               #
        #######################################################################
        [N, C, H, W] = x.size()  # N = batch size, C = number of channels, H = height, W = width
        # upscale network in order to achieve desired output dimensions (N, num_classes, H, W)
        # upsample = nn.Upsample(scale_factor= H * W, mode='bilinear', align_corners=True)
        x_inputs = x
        x = self.model_fcn(x)['out']
        # x = self.upconv(x)
        x = nn.functional.upsample(x, x_inputs.size()[2:], mode='bilinear', align_corners=True).contiguous()

        #######################################################################
        #                           END OF YOUR CODE                          #
        #######################################################################

        return x

    @property
    def is_cuda(self):
        """
        Check if model parameters are allocated on the GPU.
        """
        return next(self.parameters()).is_cuda

    def save(self, path):
        """
        Save model with its parameters to the given path. Conventionally the
        path should end with "*.model".

        Inputs:
        - path: path string
        """
        print('Saving model... %s' % path)
        torch.save(self, path)
