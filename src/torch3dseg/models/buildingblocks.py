from functools import partial

import torch
from torch import nn as nn
from torch.nn import functional as F

from .nn import *



def create_conv(dims,in_channels, out_channels, kernel_size, order, num_groups, padding):
    """
    Create a list of modules with together constitute a single conv layer with non-linearity
    and optional batchnorm/groupnorm.

    Args:
        in_channels (int): number of input channels
        out_channels (int): number of output channels
        kernel_size(int or tuple): size of the convolving kernel
        order (string): order of things, e.g.
            'cr' -> conv + ReLU
            'gcr' -> groupnorm + conv + ReLU
            'cl' -> conv + LeakyReLU
            'ce' -> conv + ELU
            'bcr' -> batchnorm + conv + ReLU
        num_groups (int): number of groups for the GroupNorm
        padding (int or tuple): add zero-padding added to all three sides of the input

    Return:
        list of tuple (name, module)
    """
    assert 'c' in order, "Conv layer MUST be present"
    assert order[0] not in 'rle', 'Non-linearity cannot be the first operation in the layer'



    modules = []
    c_index = order.index('c')
    for i, char in enumerate(order):
        if char == 'r':
            modules.append(('ReLU', nn.ReLU(inplace=True)))
        elif char == 'l':
            modules.append(('LeakyReLU', nn.LeakyReLU(inplace=True)))
        elif char == 'e':
            modules.append(('ELU', nn.ELU(inplace=True)))
        elif char == 'c':
            # add learnable bias only in the absence of batchnorm/groupnorm
            bias = not ('g' in order or 'b' in order)
            modules.append((
                "conv",
                conv_nd(
                    dims,
                    in_channels,
                    out_channels,
                    kernel_size,
                    padding=padding,
                    bias=bias,
                )
            ))
        elif char == 'g':
            # before conv -> normalize in_channels, after -> out_channels
            num_channels = in_channels if i < c_index else out_channels

            ng = num_groups
            if num_channels < ng:
                ng = 1
            assert num_channels % ng == 0, (
                f"Expected num_channels divisible by num_groups. "
                f"{num_channels=} {ng=}"
            )
            modules.append(('groupnorm', nn.GroupNorm(num_groups=ng, num_channels=num_channels)))

        elif char == 'b':
            num_features = in_channels if i < c_index else out_channels
            modules.append(('batchnorm', batchnorm_nd(dims, num_features)))

        else:
            raise ValueError(
                f"Unsupported layer type '{char}'. "
                "Use one of ['b','g','r','l','e','c']"
            )

    return modules


class SingleConv(nn.Sequential):
    """
    Basic convolutional module consisting of a Conv3d, non-linearity and optional batchnorm/groupnorm. The order
    of operations can be specified via the `order` parameter

    Args:
        in_channels (int): number of input channels
        out_channels (int): number of output channels
        kernel_size (int or tuple): size of the convolving kernel
        order (string): determines the order of layers, e.g.
            'cr' -> conv + ReLU
            'crg' -> conv + ReLU + groupnorm
            'cl' -> conv + LeakyReLU
            'ce' -> conv + ELU
        num_groups (int): number of groups for the GroupNorm
        padding (int or tuple):
    """

    def __init__(self, dims, in_channels, out_channels, kernel_size=3, order='gcr', num_groups=8, padding=1):
        super(SingleConv, self).__init__()

        for name, module in create_conv(dims,in_channels, out_channels, kernel_size, order, num_groups, padding=padding):
            self.add_module(name, module)


class DoubleConv(nn.Sequential):
    """
    A module consisting of two consecutive convolution layers (e.g. BatchNorm3d+ReLU+Conv3d).
    We use (Conv3d+ReLU+GroupNorm3d) by default.
    This can be changed however by providing the 'order' argument, e.g. in order
    to change to Conv3d+BatchNorm3d+ELU use order='cbe'.
    Use padded convolutions to make sure that the output (H_out, W_out) is the same
    as (H_in, W_in), so that you don't have to crop in the decoder path.

    Args:
        in_channels (int): number of input channels
        out_channels (int): number of output channels
        encoder (bool): if True we're in the encoder path, otherwise we're in the decoder
        kernel_size (int or tuple): size of the convolving kernel
        order (string): determines the order of layers, e.g.
            'cr' -> conv + ReLU
            'crg' -> conv + ReLU + groupnorm
            'cl' -> conv + LeakyReLU
            'ce' -> conv + ELU
        num_groups (int): number of groups for the GroupNorm
        padding (int or tuple): add zero-padding added to all three sides of the input
    """

    def __init__(self, dims, in_channels, out_channels, encoder, kernel_size=3, order='gcr', num_groups=8, padding=1):
        super(DoubleConv, self).__init__()
        if encoder:
            # we're in the encoder path
            conv1_in_channels = in_channels
            conv1_out_channels = out_channels // 2
            if conv1_out_channels < in_channels:
                conv1_out_channels = in_channels
            conv2_in_channels, conv2_out_channels = conv1_out_channels, out_channels
        else:
            # we're in the decoder path, decrease the number of channels in the 1st convolution
            conv1_in_channels, conv1_out_channels = in_channels, out_channels
            conv2_in_channels, conv2_out_channels = out_channels, out_channels

        # conv1
        self.add_module('SingleConv1',
                        SingleConv(dims, conv1_in_channels, conv1_out_channels, kernel_size, order, num_groups,
                                   padding=padding))
        # conv2
        self.add_module('SingleConv2',
                        SingleConv(dims, conv2_in_channels, conv2_out_channels, kernel_size, order, num_groups,
                                   padding=padding))


class ExtResNetBlock(nn.Module):
    """
    Basic UNet block consisting of a SingleConv followed by the residual block.
    The SingleConv takes care of increasing/decreasing the number of channels and also ensures that the number
    of output channels is compatible with the residual block that follows.
    This block can be used instead of standard DoubleConv in the Encoder module.
    Motivated by: https://arxiv.org/pdf/1706.00120.pdf

    Notice we use ELU instead of ReLU (order='cge') and put non-linearity after the groupnorm.
    """

    def __init__(self,dims, in_channels, out_channels, kernel_size=3, order='cge', num_groups=8, **kwargs):
        super(ExtResNetBlock, self).__init__()

        # first convolution
        self.conv1 = SingleConv(dims,in_channels, out_channels, kernel_size=kernel_size, order=order, num_groups=num_groups)
        # residual block
        self.conv2 = SingleConv(dims,out_channels, out_channels, kernel_size=kernel_size, order=order, num_groups=num_groups)
        # remove non-linearity from the 3rd convolution since it's going to be applied after adding the residual
        n_order = order
        
        for c in 'rel':
            n_order = n_order.replace(c, '')
        self.conv3 = SingleConv(dims, out_channels, out_channels, kernel_size=kernel_size, order=n_order,
                                num_groups=num_groups)

        # create non-linearity separately
        if 'l' in order:
            self.non_linearity = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        elif 'e' in order:
            self.non_linearity = nn.ELU(inplace=True)
        else:
            self.non_linearity = nn.ReLU(inplace=True)

    def forward(self, x):
        # apply first convolution and save the output as a residual
        out = self.conv1(x)
        residual = out

        # residual block
        out = self.conv2(out)
        out = self.conv3(out)

        out += residual
        out = self.non_linearity(out)

        return out


class Encoder(nn.Module):
    """
    A single module from the encoder path consisting of the optional max
    pooling layer (one may specify the MaxPool kernel_size to be different
    than the standard (2,2,2), e.g. if the volumetric data is anisotropic
    (make sure to use complementary scale_factor in the decoder path) followed by
    a DoubleConv module.
    Args:
        in_channels (int): number of input channels
        out_channels (int): number of output channels
        conv_kernel_size (int or tuple): size of the convolving kernel
        apply_pooling (bool): if True use MaxPool3d before DoubleConv
        pool_kernel_size (int or tuple): the size of the window
        pool_type (str): pooling layer: 'max' or 'avg'
        basic_module(nn.Module): either ResNetBlock or DoubleConv
        conv_layer_order (string): determines the order of layers
            in `DoubleConv` module. See `DoubleConv` for more info.
        num_groups (int): number of groups for the GroupNorm
        padding (int or tuple): add zero-padding added to all three sides of the input
    """

    def __init__(self, dims,in_channels, out_channels, conv_kernel_size=3, apply_pooling=True,
                 pool_kernel_size=2, down_pooling='max', basic_module=DoubleConv, conv_layer_order='gcr',
                 num_groups=8, padding=1):
        super(Encoder, self).__init__()

        assert down_pooling in ["max", "avg", "conv"]

        self.downsampling = Downsampling(
            kind=down_pooling if apply_pooling else "none",
            dims=dims,
            in_channels=in_channels,
            kernel_size=pool_kernel_size,
            stride=pool_kernel_size,   # typical UNet: stride == kernel_size
            padding=0,
        )


        self.basic_module = basic_module(dims, in_channels, out_channels,
                                         encoder=True,
                                         kernel_size=conv_kernel_size,
                                         order=conv_layer_order,
                                         num_groups=num_groups,
                                         padding=padding)

    def forward(self, x):
        if self.downsampling is not None:
            x = self.downsampling(x)
        x = self.basic_module(x)
        return x


class Decoder(nn.Module):
    def __init__(
        self,
        dims,
        in_channels,                 # channels BEFORE upsampling (from previous decoder / bottleneck)
        out_channels,                # channels AFTER upsampling (and skip channels)
        conv_kernel_size=3,
        scale_factor=2,
        basic_module=DoubleConv,
        conv_layer_order='gcr',
        num_groups=8,
        mode='nearest',
        padding=1,
        upsample=True,
        up_pooling='interp',
        join='auto',                 # 'auto' | 'concat' | 'sum'
    ):
        super().__init__()
        assert up_pooling in ['interp', 'trans_conv']
        assert join in ['auto', 'concat', 'sum']

        # decide join mode
        if join == 'auto':
            concat = (basic_module == DoubleConv)  # original convention
        else:
            concat = (join == 'concat')
        self._concat = concat

        # upsampling op
        self.upsampling = Upsampling(
            kind=up_pooling if upsample else "none",  # 'interp'/'trans_conv'
            dims=dims,
            in_channels=in_channels,
            out_channels=out_channels,
            mode=mode,
            scale_factor=scale_factor,
            kernel_size=2,
            padding=0,
            output_padding=0,
        )

        # If sum-join, ensure x channels match skip channels (out_channels).
        # If using interpolation, x channels won't change -> project if needed.
        self.x_proj = None
        if not self._concat:
            # for sum, we need x to be out_channels before adding
            if up_pooling == 'interp':
                # interpolate keeps channels = in_channels, so project to out_channels if different
                if in_channels != out_channels:
                    self.x_proj = conv_nd(dims, in_channels, out_channels, kernel_size=1, padding=0, bias=False)

        # channels into basic_module AFTER joining
        basic_in_channels = (2 * out_channels) if self._concat else out_channels

        self.basic_module = basic_module(
            dims,
            basic_in_channels,
            out_channels,
            encoder=False,
            kernel_size=conv_kernel_size,
            order=conv_layer_order,
            num_groups=num_groups,
            padding=padding,
        )

    def forward(self, encoder_features, x):
        x = self.upsampling(encoder_features=encoder_features, x=x)

        if self.x_proj is not None:
            x = self.x_proj(x)

        if self._concat:
            x = torch.cat((encoder_features, x), dim=1)
        else:
            x = encoder_features + x

        x = self.basic_module(x)
        return x

class Downsampling(nn.Module):
    """
    Unified downsampling:
      kind='max'   -> MaxPool{d}
      kind='avg'   -> AvgPool{d}
      kind='conv'  -> depthwise strided Conv{d} (channel-preserving)
      kind='none'  -> identity
    """
    def __init__(
        self,
        kind: str,
        dims: int,
        in_channels: int | None = None,   # needed for 'conv'
        kernel_size=2,
        stride=None,                      # default: kernel_size
        padding=0,
        ceil_mode: bool = False,          # for pooling
    ):
        super().__init__()
        assert kind in ("max", "avg", "conv", "none"), "kind must be 'max', 'avg', 'conv', or 'none'"

        self.kind = kind
        if stride is None:
            stride = kernel_size

        if kind == "max":
            self.op = maxpool_nd(dims, kernel_size=kernel_size, stride=stride, padding=padding, ceil_mode=ceil_mode)
        elif kind == "avg":
            self.op = avgpool_nd(dims, kernel_size=kernel_size, stride=stride, padding=padding, ceil_mode=ceil_mode)
        elif kind == "conv":
            if in_channels is None:
                raise ValueError("in_channels must be set for kind='conv'")
            # depthwise strided conv: downsamples each channel independently
            self.op = conv_nd(
                dims,
                in_channels,
                in_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                groups=in_channels,
                bias=False,
            )
        else:  # 'none'
            self.op = nn.Identity()

    def forward(self, x):
        return self.op(x)
    

class Upsampling(nn.Module):
    """
    Unified upsampling:
      kind='interp'      -> F.interpolate to encoder_features spatial size
      kind='trans_conv'  -> ConvTranspose{d} with fixed stride
      kind='none'        -> identity
    """
    def __init__(
        self,
        kind: str,
        dims: int,
        in_channels: int | None = None,
        out_channels: int | None = None,
        mode: str = "nearest",
        scale_factor=2,
        # transposed conv params:
        kernel_size=2,
        padding=0,
        output_padding=0,
    ):
        super().__init__()
        assert kind in ("interp", "trans_conv", "none")

        self.kind = kind
        self.mode = mode

        if kind == "trans_conv":
            if in_channels is None or out_channels is None:
                raise ValueError("in_channels/out_channels must be set for trans_conv")
            self.op = conv_transpose_nd(
                dims,
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=scale_factor,
                padding=padding,
                output_padding=output_padding,
                bias=False,
            )
        elif kind == "none":
            self.op = nn.Identity()
        else:
            # interp doesn't need a module; keep Identity for consistency
            self.op = nn.Identity()

    def forward(self, encoder_features, x):
        if self.kind == "interp":
            size = encoder_features.shape[2:]
            return F.interpolate(x, size=size, mode=self.mode)
        else:
            return self.op(x)



def create_encoders(dims, in_channels, f_maps, basic_module, conv_kernel_size, 
                    conv_padding, layer_order, num_groups,
                    pool_kernel_size,down_pooling,**kwargs):
    # create encoder path consisting of Encoder modules. Depth of the encoder is equal to `len(f_maps)`
    encoders = []
    for i, out_feature_num in enumerate(f_maps):
        if i == 0:
            encoder = Encoder(dims,in_channels, out_feature_num,
                              apply_pooling=False,  # skip pooling in the firs encoder
                              basic_module=basic_module,
                              conv_layer_order=layer_order,
                              conv_kernel_size=conv_kernel_size,
                              num_groups=num_groups,
                              padding=conv_padding)
        else:
            # TODO: adapt for anisotropy in the data, i.e. use proper 
            # pooling kernel to make the data isotropic after 1-2 pooling operations
            encoder = Encoder(dims,f_maps[i - 1], out_feature_num,
                              basic_module=basic_module,
                              conv_layer_order=layer_order,
                              conv_kernel_size=conv_kernel_size,
                              num_groups=num_groups,
                              pool_kernel_size=pool_kernel_size,
                              padding=conv_padding,
                              down_pooling=down_pooling
                              )

        encoders.append(encoder)

    return nn.ModuleList(encoders)


def create_decoders(
    dims, f_maps, basic_module, conv_kernel_size, conv_padding,
    layer_order, num_groups, upsample,
    up_pooling='interp',
    scale_factor=2,
    join='auto',
    **kwargs
):
    decoders = []
    reversed_f_maps = list(reversed(f_maps))

    for i in range(len(reversed_f_maps) - 1):
        in_ch = reversed_f_maps[i]         # channels before upsampling
        out_ch = reversed_f_maps[i + 1]    # channels after upsampling (and skip channels)

        _upsample = True
        if i == 0:
            _upsample = upsample

        decoder = Decoder(
            dims=dims,
            in_channels=in_ch,
            out_channels=out_ch,
            basic_module=basic_module,
            conv_layer_order=layer_order,
            conv_kernel_size=conv_kernel_size,
            num_groups=num_groups,
            padding=conv_padding,
            upsample=_upsample,
            up_pooling=up_pooling,
            scale_factor=scale_factor,
            join=join,
        )
        decoders.append(decoder)

    return nn.ModuleList(decoders)

   

