

import torch
from torch import nn as nn




def conv_nd(dims, *args, **kwargs):
    """
    Create a 1D, 2D, or 3D convolution module.
    """
    if dims == 1:
        return nn.Conv1d(*args, **kwargs)
    elif dims == 2:
        return nn.Conv2d(*args, **kwargs)
    elif dims == 3:
        return nn.Conv3d(*args, **kwargs)
    raise ValueError(f"unsupported dimensions: {dims}")


def conv_transpose_nd(dims, *args, **kwargs):
    if dims == 1:
        # maybe not really necessary
        return nn.ConvTranspose1d(*args, **kwargs)
    elif dims == 2:
        return nn.ConvTranspose2d(*args, **kwargs)
    elif dims == 3:
        return nn.ConvTranspose3d(*args, **kwargs)
    raise ValueError(f"unsupported dimensions: {dims}")


def batchnorm_nd(dims, num_features, **kwargs):
    if dims == 1:
        return nn.BatchNorm1d(num_features, **kwargs)
    if dims == 2:
        return nn.BatchNorm2d(num_features, **kwargs)
    if dims == 3:
        return nn.BatchNorm3d(num_features, **kwargs)
    raise ValueError(f"unsupported dimensions: {dims}")

def maxpool_nd(dims, *args, **kwargs):
    """
    Create a 1D, 2D, or 3D MaxPool module.
    """
    if dims == 1:
        return nn.MaxPool1d(*args, **kwargs)
    elif dims == 2:
        return nn.MaxPool2d(*args, **kwargs)
    elif dims == 3:
        return nn.MaxPool3d(*args, **kwargs)
    raise ValueError(f"unsupported dimensions: {dims}")


def avgpool_nd(dims, *args, **kwargs):
    """
    Create a 1D, 2D, or 3D AvgPool module.
    """
    if dims == 1:
        return nn.AvgPool1d(*args, **kwargs)
    elif dims == 2:
        return nn.AvgPool2d(*args, **kwargs)
    elif dims == 3:
        return nn.AvgPool3d(*args, **kwargs)
    raise ValueError(f"unsupported dimensions: {dims}")