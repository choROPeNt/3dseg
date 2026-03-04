from functools import partial

import torch
from torch import nn as nn
from torch.nn import functional as F

from .nn import *




class SingleConv(nn.Sequential):
    """
    Basic convolutional module consisting of:
      - ConvNd
      - optional normalization
      - activation

    The order of operations is controlled by `layer_order`:

      c = convolution
      n = normalization (type selected via `norm=...`)
      a = activation     (type selected via `activation=...`)

    Examples:
      'cna' -> Conv → Norm → Act
      'nca' -> Norm → Conv → Act (pre-norm style)
      'ca'  -> Conv → Act
      'c'   -> Conv only

    Args:
        dims (int): 1, 2, or 3 (Conv1d/2d/3d etc.)
        in_channels (int): input channels
        out_channels (int): output channels
        kernel_size (int or tuple): convolution kernel
        layer_order (str): sequence of ['c','n','a']
        num_groups (int): number of groups for GroupNorm (if norm='group')
        padding (int or tuple): convolution padding
        activation (str): activation name (e.g. "ReLU", "SiLU", "LeakyReLU")
        norm (str): normalization type: "group" | "batch" | "none"
        **kwargs: forwarded to the activation constructor (e.g. inplace=True, negative_slope=0.1)
    """

    def __init__(
        self,
        dims,
        in_channels,
        out_channels,
        kernel_size=3,
        layer_order="cna",          
        num_groups=8,
        padding=1,
        activation="ReLU",
        norm="group",
        **kwargs,
    ):
        super().__init__()
        for name, module in create_conv(
            dims=dims,
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            layer_order=layer_order,
            num_groups=num_groups,
            padding=padding,
            activation=activation,
            norm=norm,
            **kwargs,
        ):
            self.add_module(name, module)

class DoubleConv(nn.Sequential):
    """
    Two consecutive SingleConv blocks.

    Uses padded convolutions so spatial size is preserved.

    Conventions:
      layer_order chars:
        c = conv
        n = norm (selected via `norm`)
        a = activation (selected via `activation`)

      Examples:
        order="cna" -> Conv → Norm → Act
        order="nca" -> Norm → Conv → Act (pre-norm)

    Args:
        dims (int): 1/2/3 for Conv1d/2d/3d blocks
        in_channels (int): input channels
        out_channels (int): output channels
        encoder (bool): True for encoder path (optionally reduces channels in first conv)
        kernel_size (int or tuple): conv kernel size
        layer_order (str): layer order using ['c','n','a']
        num_groups (int): groups for GroupNorm if norm='group'
        padding (int or tuple): conv padding
        activation (str): activation name, e.g. "ReLU", "SiLU", "LeakyReLU"
        norm (str): "group" | "batch" | "none"
        **kwargs: forwarded to activation ctor (e.g. inplace=True, negative_slope=0.1)
    """

    def __init__(
        self,
        dims,
        in_channels,
        out_channels,
        encoder: bool,
        kernel_size=3,
        layer_order="cna",
        num_groups=8,
        padding=1,
        activation="ReLU",
        norm="group",
        **kwargs,
    ):
        super().__init__()

        if encoder:
            # encoder path: optionally reduce channels in the first conv
            conv1_in = in_channels
            conv1_out = out_channels // 2
            if conv1_out < in_channels:
                conv1_out = in_channels
            conv2_in, conv2_out = conv1_out, out_channels
        else:
            # decoder path: keep channels stable
            conv1_in, conv1_out = in_channels, out_channels
            conv2_in, conv2_out = out_channels, out_channels

        self.add_module(
            "SingleConv1",
            SingleConv(
                dims,
                conv1_in,
                conv1_out,
                kernel_size=kernel_size,
                layer_order=layer_order,
                num_groups=num_groups,
                padding=padding,
                activation=activation,
                norm=norm,
                **kwargs,
            ),
        )

        self.add_module(
            "SingleConv2",
            SingleConv(
                dims,
                conv2_in,
                conv2_out,
                kernel_size=kernel_size,
                layer_order=layer_order,
                num_groups=num_groups,
                padding=padding,
                activation=activation,
                norm=norm,
                **kwargs,
            ),
        )
class ExtResNetBlock(nn.Module):
    """
    Residual UNet block consisting of:

        SingleConv → SingleConv → SingleConv + residual → activation

    The first SingleConv adapts the number of channels so the residual
    connection is valid.

    Conventions (same as SingleConv):

        layer_order chars:
            c = convolution
            n = normalization
            a = activation

        activation type is controlled by `activation`
        normalization type by `norm`

    Args:
        dims (int): 1/2/3 for ConvNd
        in_channels (int)
        out_channels (int)
        kernel_size (int or tuple)
        layer_order (str): combination of ['c','n','a']
        num_groups (int): groups for GroupNorm
        activation (str): activation name
        norm (str): "group" | "batch" | "none"
        **kwargs: forwarded to activation (e.g. inplace=True)
    """

    def __init__(
        self,
        dims,
        in_channels,
        out_channels,
        kernel_size=3,
        layer_order="cna",
        num_groups=8,
        activation="ReLU",
        norm="group",
        **kwargs,
    ):
        super().__init__()

        # First conv adjusts channels
        self.conv1 = SingleConv(
            dims,
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            layer_order=layer_order,
            num_groups=num_groups,
            activation=activation,
            norm=norm,
            **kwargs,
        )

        # Residual block convs
        self.conv2 = SingleConv(
            dims,
            out_channels,
            out_channels,
            kernel_size=kernel_size,
            layer_order=layer_order,
            num_groups=num_groups,
            activation=activation,
            norm=norm,
            **kwargs,
        )

        # Remove activation from third conv (applied after residual)
        n_order = layer_order.replace("a", "")

        self.conv3 = SingleConv(
            dims,
            out_channels,
            out_channels,
            kernel_size=kernel_size,
            layer_order=n_order,
            num_groups=num_groups,
            activation=activation,
            norm=norm,
            **kwargs,
        )

        # final activation
        self.non_linearity = get_activation(activation, **kwargs)

    def forward(self, x):
        out = self.conv1(x)
        residual = out

        out = self.conv2(out)
        out = self.conv3(out)

        out += residual
        out = self.non_linearity(out)

        return out

class Encoder(nn.Module):
    """
    One encoder stage: optional downsampling followed by a basic conv block.

    Downsampling is controlled by `down_pooling`:
      - "max"  : MaxPool{d}
      - "avg"  : AvgPool{d}
      - "conv" : depthwise strided Conv{d}
      - disabled via apply_pooling=False -> Identity

    The conv block order is controlled by `layer_order` using:
      c = convolution
      n = normalization (type via `norm`)
      a = activation     (type via `activation`)

    Examples:
      layer_order="cna" -> Conv → Norm → Act
      layer_order="nca" -> Norm → Conv → Act (pre-norm)
    """

    def __init__(
        self,
        dims: int,
        in_channels: int,
        out_channels: int,
        conv_kernel_size=3,
        apply_pooling: bool = True,
        pool_kernel_size=2,
        down_pooling: str = "max",
        basic_module=DoubleConv,
        layer_order: str = "cna",
        num_groups: int = 8,
        padding=1,
        activation: str = "ReLU",
        norm: str = "group",
        **kwargs,
    ):
        super().__init__()

        assert down_pooling in ["max", "avg", "conv"], "down_pooling must be one of ['max','avg','conv']"

        # Always define a module; if no pooling desired, use identity
        self.downsampling = Downsampling(
            kind=down_pooling if apply_pooling else "none",
            dims=dims,
            in_channels=in_channels,      # required for kind='conv'
            kernel_size=pool_kernel_size,
            stride=pool_kernel_size,
            padding=0,
        )

        self.basic_module = basic_module(
            dims,
            in_channels,
            out_channels,
            encoder=True,
            kernel_size=conv_kernel_size,
            layer_order=layer_order,
            num_groups=num_groups,
            padding=padding,
            activation=activation,
            norm=norm,
            **kwargs,
        )

    def forward(self, x):
        x = self.downsampling(x)
        x = self.basic_module(x)
        return x

class Decoder(nn.Module):
    def __init__(
        self,
        dims: int,
        in_channels: int,             # channels BEFORE upsampling (prev decoder / bottleneck)
        out_channels: int,            # channels AFTER upsampling (and skip channels)
        conv_kernel_size=3,
        scale_factor=2,
        basic_module=DoubleConv,
        layer_order: str = "cna",
        num_groups: int = 8,
        mode: str = "nearest",
        padding=1,
        upsample: bool = True,
        up_pooling: str = "interp",   # 'interp' | 'trans_conv'
        join: str = "auto",           # 'auto' | 'concat' | 'sum'
        activation: str = "ReLU",
        norm: str = "group",
        **kwargs,
    ):
        super().__init__()
        assert up_pooling in ["interp", "trans_conv"]
        assert join in ["auto", "concat", "sum"]

        # decide join mode
        if join == "auto":
            concat = (basic_module == DoubleConv)   # original convention
        else:
            concat = (join == "concat")
        self._concat = concat

        # choose effective upsampling kind
        up_kind = up_pooling if upsample else "none"

        self.upsampling = Upsampling(
            kind=up_kind,
            dims=dims,
            in_channels=in_channels,
            out_channels=out_channels,
            mode=mode,
            scale_factor=scale_factor,
            kernel_size=2,
            padding=0,
            output_padding=0,
        )

        # For sum-join, x must have same channels as encoder_features (= out_channels).
        # If we used interpolation (or no upsampling), channels don't change -> project if needed.
        self.x_proj = None
        if not self._concat:
            if up_kind in ("interp", "none"):
                if in_channels != out_channels:
                    self.x_proj = conv_nd(
                        dims, in_channels, out_channels,
                        kernel_size=1, padding=0, bias=False
                    )
            # if up_kind == "trans_conv": conv-transpose already outputs out_channels

        # channels into basic_module AFTER joining
        basic_in_channels = (2 * out_channels) if self._concat else out_channels

        self.basic_module = basic_module(
            dims,
            basic_in_channels,
            out_channels,
            encoder=False,
            kernel_size=conv_kernel_size,
            layer_order=layer_order,
            num_groups=num_groups,
            padding=padding,
            activation=activation,
            norm=norm,
            **kwargs,
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


def create_encoders(
    dims: int,
    in_channels: int,
    f_maps,
    basic_module,
    conv_kernel_size=3,
    conv_padding=1,
    layer_order: str = "cna",
    num_groups: int = 8,
    pool_kernel_size=2,
    down_pooling: str = "max",
    activation: str = "ReLU",
    norm: str = "group",
    **kwargs,
) -> nn.ModuleList:
    """
    Create encoder path consisting of Encoder modules.
    Depth is len(f_maps).

    f_maps example: [32, 64, 128]
    encoders:
      (in=in_channels -> 32, no pool),
      (in=32 -> 64, pool),
      (in=64 -> 128, pool)
    """
    assert down_pooling in ["max", "avg", "conv"], "down_pooling must be one of ['max','avg','conv']"

    encoders = []
    for i, out_ch in enumerate(f_maps):
        apply_pooling = (i != 0)
        in_ch = in_channels if i == 0 else f_maps[i - 1]

        encoders.append(
            Encoder(
                dims=dims,
                in_channels=in_ch,
                out_channels=out_ch,
                conv_kernel_size=conv_kernel_size,
                apply_pooling=apply_pooling,
                pool_kernel_size=pool_kernel_size,
                down_pooling=down_pooling,
                basic_module=basic_module,
                layer_order=layer_order,
                num_groups=num_groups,
                padding=conv_padding,
                activation=activation,
                norm=norm,
                **kwargs,
            )
        )

    return nn.ModuleList(encoders)


def create_decoders(
    dims: int,
    f_maps,
    basic_module,
    conv_kernel_size=3,
    conv_padding=1,
    layer_order: str = "cna",
    num_groups: int = 8,
    upsample: bool = True,
    up_pooling: str = "interp",
    scale_factor=2,
    join: str = "auto",
    mode: str = "nearest",
    activation: str = "ReLU",
    norm: str = "group",
    **kwargs,
) -> nn.ModuleList:
    """
    Create decoder path consisting of Decoder modules.
    Length is len(f_maps) - 1.

    f_maps example: [32, 64, 128, 256]
    reversed:      [256, 128, 64, 32]
    decoders:
      (in=256 -> out=128), (in=128 -> out=64), (in=64 -> out=32)
    """
    assert up_pooling in ["interp", "trans_conv"]
    assert join in ["auto", "concat", "sum"]

    decoders = []
    rf = list(reversed(f_maps))

    for i in range(len(rf) - 1):
        in_ch = rf[i]        # from previous decoder/bottleneck
        out_ch = rf[i + 1]   # target channels (and skip channels)

        # allow skipping upsampling only for the first decoder if requested
        _upsample = upsample if i == 0 else True

        decoders.append(
            Decoder(
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
                mode=mode,
                activation=activation,
                norm=norm,
                **kwargs,
            )
        )

    return nn.ModuleList(decoders)

   

