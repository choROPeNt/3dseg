import torch
import torch.nn.functional as F


## commented function version of 
# def compute_s2_torch(ms: torch.Tensor, limit_to = 64, device=None):
#     """
#     Compute the Two-Point Correlation Function (S2) for a microstructure with optional shape limiting.

#     Parameters:
#         ms (torch.Tensor): Binary textile microstructure phase map (1 = phase, 0 = None).
#         desired_shape (tuple, optional): Target shape to reshape the microstructure before computing S2.
#         limit_to (int, optional): Extracts only a central region of S2 within the given limit.
#         device (str): Computation device ("mps", "cuda", or "cpu").

#     Returns:
#         torch.Tensor: The S2 correlation function (full or limited region).
#     """
#     if device is None:
#         device = ms.device
        
#     # Move tensor to device and ensure float32
#     ms = ms.to(dtype=torch.float32, device=device)

#     # Compute Fourier transform of the phase indicator function
#     ms_fourier = torch.fft.rfftn(ms)

#     # Compute squared magnitude (autocorrelation in frequency domain)
#     s2_fourier = ms_fourier * torch.conj(ms_fourier)

#     # Compute inverse Fourier transform to get real-space correlation function
#     s2 = torch.fft.irfftn(s2_fourier, s=ms.shape).real

#     # Normalize by total number of elements
#     s2 /= ms.numel()

#     # Shift zero frequency component to center
#     s2 = torch.fft.fftshift(s2)
#     # Apply `limit_to` to extract a centered region
#     if limit_to is not None:
#         center = [dim // 2 for dim in s2.shape]  # Find center
#         slices = tuple(
#             slice(c - limit_to + 1, c + limit_to) for c in center
#         )
#         s2 = s2[slices]
#     return s2

## old version removed
# class compute_s2(torch.nn.Module):
#     def __init__(self, normalize: bool = True,limit_to: int = None):
#         """
#         Parameters:
#             normalize (bool): Whether to normalize the autocorrelation by its maximum.
#             limit_to (int or None): If set, returns a cropped central region of size 2*limit_to - 1 in each dim.
#         """
#         super().__init__()
#         self.normalize = normalize
#         self.limit_to = limit_to

#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         """
#         Computes the autocorrelation via FFT for 2D or 3D binary input.

#         Parameters:
#             x (Tensor): Shape (B, C, H, W) or (B, C, D, H, W)

#         Returns:
#             Tensor: Autocorrelation of shape (B, C, ...) possibly cropped.
#         """
#         spatial_dims = x.shape[2:]
#         pad_sizes = []
#         for dim in reversed(spatial_dims):
#             pad_sizes += [0, dim]  # pad both sides

#         # x_mean = x.mean(dim=tuple(range(2, x.dim())), keepdim=True)
#         # x_centered = x - x_mean
#         x_padded = torch.nn.functional.pad(x, pad=pad_sizes, mode='constant', value=0)

#         fft_x = torch.fft.fftn(x_padded, dim=tuple(range(2, x.dim())))
#         autocorr_fft = fft_x * torch.conj(fft_x)
#         autocorr = torch.fft.ifftn(autocorr_fft, dim=tuple(range(2, x.dim()))).real
#         autocorr = torch.fft.fftshift(autocorr, dim=tuple(range(2, x.dim())))

#         if self.normalize:
#             maxval = autocorr.amax(dim=tuple(range(2, x.dim())), keepdim=True) + 1e-8
#             autocorr = autocorr / maxval

#         if self.limit_to is not None:
#                     centers = [s // 2 for s in autocorr.shape[2:]]
#                     slices = tuple(
#                         slice(c - self.limit_to, c + self.limit_to) for c in centers
#                     )
#                     autocorr = autocorr[(...,) + slices]  # keep B, C dims

#         return autocorr
    


class ComputeS2(torch.nn.Module):
    def __init__(self, normalize: bool = True, limit_to: int = None, norm_mode: str = None, eps: float = 1e-8):
        """
        Parameters
        ----------
        normalize : bool
            Backward-compat flag. If True and norm_mode is None -> use 'max' normalization.
            If norm_mode is set, this flag is ignored.
        limit_to : int or None
            If set, crops a centered region of size 2*limit_to in each spatial dim after shift.
        norm_mode : {'none','max','pairs','covariance', None}
            - 'none'       : return raw autocorrelation (no scaling).
            - 'max'        : divide by global max per (B,C).
            - 'pairs'      : divide by number of valid pairs (edge-effect correction) -> proper S2.
            - 'covariance' : first 'pairs', then (S2 - φ^2)/(φ(1-φ)).
            None           : uses 'max' if normalize=True else 'none'.
        eps : float
            Numerical stabilizer.
        """
        super().__init__()
        if norm_mode is None:
            norm_mode = 'max' if normalize else 'none'
        assert norm_mode in {'none', 'max', 'pairs', 'covariance'}
        self.norm_mode = norm_mode
        self.limit_to = limit_to
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Computes autocorrelation via FFT for 2D or 3D inputs.

        Parameters
        ----------
        x : Tensor
            Shape (B, C, H, W) or (B, C, D, H, W). Values are 0/1 indicators or probabilities.

        Returns
        -------
        Tensor
            Autocorrelation/S2 of shape (B, C, ...), possibly cropped.
        """
        # Spatial dims and padding (pad both sides by original size for full support)
        spatial_axes = tuple(range(2, x.dim()))
        pad_sizes = []
        for dim in reversed(x.shape[2:]):
            pad_sizes += [0, dim]  # (left, right) per dim, reversed order for F.pad

        # Pad signal
        x_padded = F.pad(x, pad=pad_sizes, mode='constant', value=0)

        # FFT autocorrelation
        fft_x = torch.fft.fftn(x_padded, dim=spatial_axes)
        autocorr = torch.fft.ifftn(fft_x * torch.conj(fft_x), dim=spatial_axes).real
        autocorr = torch.fft.fftshift(autocorr, dim=spatial_axes)

        if self.norm_mode in ('pairs', 'covariance'):
            # Edge-effect correction: valid pair counts
            ones = torch.ones_like(x)
            ones_padded = F.pad(ones, pad=pad_sizes, mode='constant', value=0)
            fft_m = torch.fft.fftn(ones_padded, dim=spatial_axes)
            counts = torch.fft.ifftn(fft_m * torch.conj(fft_m), dim=spatial_axes).real
            counts = torch.fft.fftshift(counts, dim=spatial_axes).clamp_min(1.0)
            S2 = autocorr / counts
        else:
            S2 = autocorr

        if self.norm_mode == 'max':
            # Per (B,C) global max across spatial dims
            maxval = S2.amax(dim=spatial_axes, keepdim=True) + self.eps
            S2 = S2 / maxval
        elif self.norm_mode == 'covariance':
            # φ from unpadded x (fraction of ones per (B,C))
            phi = x.mean(dim=spatial_axes, keepdim=True).clamp(0.0, 1.0)
            S2 = (S2 - phi**2) / (phi * (1.0 - phi) + self.eps)

        # Optional central crop after shift
        if self.limit_to is not None:
            centers = [s // 2 for s in S2.shape[2:]]
            slices = tuple(slice(c - self.limit_to, c + self.limit_to) for c in centers)
            S2 = S2[(...,) + slices]

        return S2