import torch




def compute_s2_torch(ms: torch.Tensor, limit_to = 64, device=None):
    """
    Compute the Two-Point Correlation Function (S2) for a microstructure with optional shape limiting.

    Parameters:
        ms (torch.Tensor): Binary textile microstructure phase map (1 = phase, 0 = None).
        desired_shape (tuple, optional): Target shape to reshape the microstructure before computing S2.
        limit_to (int, optional): Extracts only a central region of S2 within the given limit.
        device (str): Computation device ("mps", "cuda", or "cpu").

    Returns:
        torch.Tensor: The S2 correlation function (full or limited region).
    """
    if device is None:
        device = ms.device
        
    # Move tensor to device and ensure float32
    ms = ms.to(dtype=torch.float32, device=device)

    # Compute Fourier transform of the phase indicator function
    ms_fourier = torch.fft.rfftn(ms)

    # Compute squared magnitude (autocorrelation in frequency domain)
    s2_fourier = ms_fourier * torch.conj(ms_fourier)

    # Compute inverse Fourier transform to get real-space correlation function
    s2 = torch.fft.irfftn(s2_fourier, s=ms.shape).real

    # Normalize by total number of elements
    s2 /= ms.numel()

    # Shift zero frequency component to center
    s2 = torch.fft.fftshift(s2)
    # Apply `limit_to` to extract a centered region
    if limit_to is not None:
        center = [dim // 2 for dim in s2.shape]  # Find center
        slices = tuple(
            slice(c - limit_to + 1, c + limit_to) for c in center
        )
        s2 = s2[slices]
    return s2

class compute_s2(torch.nn.Module):
    def __init__(self, normalize: bool = True,limit_to: int = None):
        """
        Parameters:
            normalize (bool): Whether to normalize the autocorrelation by its maximum.
            limit_to (int or None): If set, returns a cropped central region of size 2*limit_to - 1 in each dim.
        """
        super().__init__()
        self.normalize = normalize
        self.limit_to = limit_to

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Computes the autocorrelation via FFT for 2D or 3D binary input.

        Parameters:
            x (Tensor): Shape (B, C, H, W) or (B, C, D, H, W)

        Returns:
            Tensor: Autocorrelation of shape (B, C, ...) possibly cropped.
        """
        spatial_dims = x.shape[2:]
        pad_sizes = []
        for dim in reversed(spatial_dims):
            pad_sizes += [0, dim]  # pad both sides

        x_mean = x.mean(dim=tuple(range(2, x.dim())), keepdim=True)
        x_centered = x - x_mean
        x_padded = torch.nn.functional.pad(x_centered, pad=pad_sizes, mode='constant', value=0)

        fft_x = torch.fft.fftn(x_padded, dim=tuple(range(2, x.dim())))
        autocorr_fft = fft_x * torch.conj(fft_x)
        autocorr = torch.fft.ifftn(autocorr_fft, dim=tuple(range(2, x.dim()))).real
        autocorr = torch.fft.fftshift(autocorr, dim=tuple(range(2, x.dim())))

        if self.normalize:
            maxval = autocorr.amax(dim=tuple(range(2, x.dim())), keepdim=True) + 1e-8
            autocorr = autocorr / maxval

        if self.limit_to is not None:
                    centers = [s // 2 for s in autocorr.shape[2:]]
                    slices = tuple(
                        slice(c - self.limit_to, c + self.limit_to) for c in centers
                    )
                    autocorr = autocorr[(...,) + slices]  # keep B, C dims

        return autocorr