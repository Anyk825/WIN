# models/preprocess.py

import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# Pre-Emphasis Filter
# ============================================================

class PreEmphasis(nn.Module):
    """
    Pre-emphasis filtering using 1D convolution.

    y[t] = x[t] - a * x[t-1]
    """

    def __init__(self, pre_emphasis: float = 0.97):

        super().__init__()

        filt = torch.tensor(
            [[-pre_emphasis, 1.0]],
            dtype=torch.float32
        ).unsqueeze(0)

        # Register as buffer (moves with .to(device))
        self.register_buffer("filter", filt)


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, T)

        Returns:
            (B, T)
        """

        x = x.unsqueeze(1)          # (B, 1, T)
        x = F.pad(x, (1, 0), mode="reflect")
        x = F.conv1d(x, self.filter)

        return x.squeeze(1)
