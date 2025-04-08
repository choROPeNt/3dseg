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