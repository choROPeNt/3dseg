


from torch import nn as nn
import inspect




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
    """
    Create a 1D, 2D, or 3D transpose convolution module.
    """
    if dims == 1:
        # maybe not really necessary
        return nn.ConvTranspose1d(*args, **kwargs)
    elif dims == 2:
        return nn.ConvTranspose2d(*args, **kwargs)
    elif dims == 3:
        return nn.ConvTranspose3d(*args, **kwargs)
    raise ValueError(f"unsupported dimensions: {dims}")


def batchnorm_nd(dims, num_features, **kwargs):
    """
    Create a 1D, 2D, or 3D batchnorm module.
    """
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



def get_activation(name: str, **kwargs) -> nn.Module:
    """
    Return an activation module from torch.nn by name.

    Extra kwargs that are not supported by the activation constructor
    are automatically filtered out.
    """

    if not isinstance(name, str):
        raise TypeError(f"Activation name must be a string, got {type(name)}")

    name = name.lower()

    registry = {
        "relu": nn.ReLU,
        "leakyrelu": nn.LeakyReLU,
        "elu": nn.ELU,
        "silu": nn.SiLU,
        "swish": nn.SiLU,
        "gelu": nn.GELU,
        "prelu": nn.PReLU,
        "tanh": nn.Tanh,
        "sigmoid": nn.Sigmoid,
        "softplus": nn.Softplus,
        "identity": nn.Identity,
    }

    if name not in registry:
        raise ValueError(
            f"Unsupported activation '{name}'. "
            f"Available: {list(registry.keys())}"
        )

    activation_cls = registry[name]

    # filter kwargs based on constructor signature
    sig = inspect.signature(activation_cls.__init__)
    valid_params = sig.parameters

    filtered_kwargs = {
        k: v for k, v in kwargs.items()
        if k in valid_params
    }

    return activation_cls(**filtered_kwargs)


def get_norm(
    norm: str,
    dims: int,
    num_channels: int,
    num_groups: int = 8,
    **kwargs,
) -> nn.Module:
    """
    norm: 'group' | 'batch' | 'none'
    kwargs are forwarded to the norm module when applicable.
    """
    norm = (norm or "none").lower()

    if norm == "none":
        return nn.Identity()

    if norm == "group":
        ng = num_groups if num_channels >= num_groups else 1
        if num_channels % ng != 0:
            raise ValueError(f"GroupNorm requires num_channels % num_groups == 0, got {num_channels=} {ng=}")
        # GroupNorm kwargs typically unused; keep for flexibility
        return nn.GroupNorm(num_groups=ng, num_channels=num_channels, **kwargs)

    if norm == "batch":
        # BatchNorm kwargs: eps, momentum, affine, track_running_stats...
        if dims == 1:
            return nn.BatchNorm1d(num_channels, **kwargs)
        if dims == 2:
            return nn.BatchNorm2d(num_channels, **kwargs)
        if dims == 3:
            return nn.BatchNorm3d(num_channels, **kwargs)
        raise ValueError(f"unsupported dimensions: {dims}")

    raise ValueError("norm must be one of ['group', 'batch', 'none']")




def create_conv(
    dims,
    in_channels,
    out_channels,
    kernel_size,
    layer_order,
    num_groups,
    padding,
    activation="ReLU",
    norm="group",          
    **kwargs,
) -> list[tuple[str, nn.Module]]:
    """
    layer_order chars:
      c = convolution
      n = normalization (type chosen via `norm`)
      a = activation (type chosen via `activation`)

    examples:
      'cna' -> Conv → Norm → Act
      'nca' -> Norm → Conv → Act (pre-norm style)
      'ca'  -> Conv → Act
      'c'   -> Conv only
    """
    assert "c" in layer_order, "Conv layer MUST be present"
    assert layer_order[0] != "a", "Activation cannot be first"

    modules = []
    c_index = layer_order.index("c")

    # if norm present anywhere, disable conv bias (standard practice)
    bias = not ("n" in layer_order and (norm or "none").lower() != "none")

    for i, ch in enumerate(layer_order):
        if ch == "c":
            modules.append((
                f"conv{i}",
                conv_nd(
                    dims,
                    in_channels,
                    out_channels,
                    kernel_size,
                    padding=padding,
                    bias=bias,
                )
            ))

        elif ch == "n":
            # before conv -> normalize in_channels, after -> out_channels
            num_ch = in_channels if i < c_index else out_channels
            modules.append((
                f"norm{i}",
                get_norm(norm=norm, dims=dims, num_channels=num_ch, num_groups=num_groups)
            ))

        elif ch == "a":
            modules.append((f"act{i}", get_activation(activation, **kwargs)))

        else:
            raise ValueError(f"Unsupported layer type '{ch}'. Use one of ['c','n','a'].")

    return modules