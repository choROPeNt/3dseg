import warnings

import torch.nn as nn

from .buildingblocks import DoubleConv, ExtResNetBlock, create_encoders, \
    create_decoders, conv_nd
from torch3dseg.utils.utils import number_of_features_per_level, get_class


class AbstractUNet(nn.Module):
    """
    Base class for standard and residual UNet.

    Args:
        in_channels (int): number of input channels
        out_channels (int): number of output segmentation masks;
            Note that that the of out_channels might correspond to either
            different semantic classes or to different binary segmentation mask.
            It's up to the user of the class to interpret the out_channels and
            use the proper loss criterion during training (i.e. CrossEntropyLoss (multi-class)
            or BCEWithLogitsLoss (two-class) respectively)
        f_maps (int, tuple): number of feature maps at each level of the encoder; if it's an integer the number
            of feature maps is given by the geometric progression: f_maps ^ k, k=1,2,3,4
        final_activation (str): activation applied after the final 1x1 convolution,
            either "sigmoid" or "softmax". Use "sigmoid" if nn.BCELoss (two-class) is used
            to train the model, "softmax" if nn.CrossEntropyLoss (multi-class) is used.
        basic_module: basic model for the encoder/decoder (DoubleConv, ExtResNetBlock, ....)
        layer_order (string): determines the order of layers
            in `SingleConv` module. e.g. 'crg' stands for Conv+ReLU+GroupNorm.
            See `SingleConv` for more info
        num_groups (int): number of groups for the GroupNorm
        num_levels (int): number of levels in the encoder/decoder path. Used ONLY when
            f_maps is an int (geometric expansion); ignored when f_maps is an explicit
            list, where the depth is len(f_maps).
        attention: self-attention flags, highest-res level first. None (off), a single
            bool (all levels), or a list[bool] of length len(f_maps), e.g.
            [False, False, True] to attend only at the bottleneck. The SAME spec drives
            the decoders, mirrored by resolution (a decoder attends iff its same-res
            encoder level does), so attention is symmetric and you only specify one list.
            O(N^2) in tokens -> keep to low-res levels.
        attention_num_heads (int): number of heads for the attention blocks, shared by
            encoder and decoder (each enabled stage's channels must be divisible by it)
        is_segmentation (bool): if True (semantic segmentation problem) Sigmoid/Softmax normalization is applied
            after the final convolution; if False (regression problem) the normalization layer is skipped at the end
        conv_kernel_size (int or tuple): size of the convolving kernel in the basic_module
        pool_kernel_size (int or tuple): the size of the window
        conv_padding (int or tuple): add zero-padding added to all three sides of the input
        pool_type (str): 
    """

    def __init__(
        self,
        dims,
        in_channels,
        out_channels,
        final_activation,
        basic_module,
        f_maps=64,
        layer_order="cna",
        num_groups=8,
        num_levels=None,
        is_segmentation=True,
        conv_kernel_size=3,
        pool_kernel_size=2,
        conv_padding=1,
        down_pooling="max",
        up_pooling="interp",
        activation="ReLU",
        norm="group",
        attention=None,
        attention_num_heads=4,
        **kwargs,
    ):
        super(AbstractUNet, self).__init__()

        # `num_levels` is only the depth knob for the int (geometric) form of
        # `f_maps`. When `f_maps` is an explicit list, len(f_maps) is the single
        # source of truth and `num_levels` is redundant -> only warn if the user
        # explicitly passed a conflicting value.
        if isinstance(f_maps, int):
            f_maps = number_of_features_per_level(f_maps, num_levels=num_levels or 4)
        elif num_levels is not None and num_levels != len(f_maps):
            warnings.warn(
                f"`num_levels`={num_levels} is ignored when `f_maps` is an explicit "
                f"list; network depth is len(f_maps)={len(f_maps)}.",
                stacklevel=2,
            )

        assert isinstance(f_maps, (list, tuple))
        assert len(f_maps) > 1, "Required at least 2 levels in the U-Net"

        # create encoder path
        self.encoders = create_encoders(
            dims=dims,
            in_channels=in_channels,
            f_maps=f_maps,
            basic_module=basic_module,
            conv_kernel_size=conv_kernel_size,
            conv_padding=conv_padding,
            layer_order=layer_order,
            num_groups=num_groups,
            pool_kernel_size=pool_kernel_size,
            down_pooling=down_pooling,
            activation=activation,
            norm=norm,
            attention=attention,
            attention_num_heads=attention_num_heads,
            **kwargs,
        )

        # create decoder path
        self.decoders = create_decoders(
            dims=dims,
            f_maps=f_maps,
            basic_module=basic_module,
            conv_kernel_size=conv_kernel_size,
            conv_padding=conv_padding,
            layer_order=layer_order,
            num_groups=num_groups,
            upsample=True,              # <-- keep as bool; if you want to control skipping first upsample, pass it here
            up_pooling=up_pooling,
            scale_factor=pool_kernel_size,  # typical: mirror pooling kernel/stride
            join="auto",                # or expose as param if you want
            mode="nearest",             # or expose as param if you want
            activation=activation,
            norm=norm,
            attention=attention,   # same spec as encoders; mirrored by resolution
            attention_num_heads=attention_num_heads,
            **kwargs,
        )

        # in the last layer a 1×1 convolution reduces the number of output
        # channels to the number of labels
        self.final_conv = conv_nd(dims,f_maps[0], out_channels, 1)

        if is_segmentation:
            # semantic segmentation problem
            act = (final_activation or "").lower()
            if act == "sigmoid":
                self.final_activation = nn.Sigmoid()
            elif act == "softmax":
                # [b,classes,x,y,z] dim=1 are the classes -> logits to probs
                self.final_activation = nn.Softmax(dim=1)
            else:
                raise ValueError(
                    f"Unsupported final_activation '{final_activation}'. "
                    f"Use 'sigmoid' or 'softmax'."
                )
        else:
            # regression problem
            self.final_activation = None

    def forward(self, x, return_logits=False):
        """
        Forward pass through the network.

        Args:
            x (torch.Tensor): Input tensor of shape (B, C, D, H, W) for 3D or (B, C, H, W) for 2D,
                              where N is the batch size, C is the number of channels,
                              D is the depth, H is the height, and W is the width.
            return_logits (bool): If True, returns both the output and the logits.
                                  If False, returns only the output. Default is False.

        Returns:
            torch.Tensor: The output tensor after passing through the network.
                          If return_logits is True, returns a tuple of (output, logits).
        """
        output, logits = self._forward_logits(x)
        if return_logits:
            return output, logits
        return output

    def _forward_logits(self, x):
        # encoder part
        encoders_features = []
        for encoder in self.encoders:
            x = encoder(x)
            # reverse the encoder outputs to be aligned with the decoder
            encoders_features.insert(0, x)

        # remove the last encoder's output from the list
        # !!remember: it's the 1st in the list
        encoders_features = encoders_features[1:]

        # decoder part
        for decoder, encoder_features in zip(self.decoders, encoders_features):
            # pass the output from the corresponding encoder and the output
            # of the previous decoder
            x = decoder(encoder_features, x)

        x = self.final_conv(x)

        if self.final_activation is not None:
            # compute final activation
            out = self.final_activation(x)
            # return both probabilities and logits
            return out, x

        return x, None

class UNet(AbstractUNet):
    """
    3DUnet model from
    `"3D U-Net: Learning Dense Volumetric Segmentation from Sparse Annotation"
        <https://arxiv.org/pdf/1606.06650.pdf>`.

    Uses `DoubleConv` as a basic_module and nearest neighbor upsampling in the decoder
    """

    def __init__(self, dims, in_channels, out_channels, final_activation="softmax", f_maps=64, layer_order='gcr',
                 num_groups=8, num_levels=None, is_segmentation=True, conv_padding=1, **kwargs):
        super(UNet, self).__init__(dims=dims,
                                    in_channels=in_channels,
                                    out_channels=out_channels,
                                    final_activation=final_activation,
                                    basic_module=DoubleConv,
                                    f_maps=f_maps,
                                    layer_order=layer_order,
                                    num_groups=num_groups,
                                    num_levels=num_levels,
                                    is_segmentation=is_segmentation,
                                    conv_padding=conv_padding,
                                    **kwargs)


class ResidualUNet(AbstractUNet):
    """
    Residual 3DUnet model implementation based on https://arxiv.org/pdf/1706.00120.pdf.
    Uses ExtResNetBlock as a basic building block, summation joining instead
    of concatenation joining and transposed convolutions for upsampling (watch out for block artifacts).
    Since the model effectively becomes a residual net, in theory it allows for deeper UNet.
    """

    def __init__(self, dims, in_channels, out_channels, final_activation="softmax", f_maps=64, layer_order='gcr',
                 num_groups=8, num_levels=None, is_segmentation=True, conv_padding=1, **kwargs):
        super(ResidualUNet, self).__init__(dims=dims,
                                            in_channels=in_channels,
                                            out_channels=out_channels,
                                            final_activation=final_activation,
                                            basic_module=ExtResNetBlock,
                                            f_maps=f_maps,
                                            layer_order=layer_order,
                                            num_groups=num_groups,
                                            num_levels=num_levels,
                                            is_segmentation=is_segmentation,
                                            conv_padding=conv_padding,
                                            **kwargs)





def get_model(model_config: dict) -> nn.Module:
    model_class = get_class(model_config["name"], modules=["torch3dseg.models.model"])
    return model_class(**{k: v for k, v in model_config.items() if k != "name"})
