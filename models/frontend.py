# models/frontend.py

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# Sinc Convolution Layer
# ============================================================

class SincConv(nn.Module):
    """
    Adapted from AASIST. One input channel only.
    """

    @staticmethod
    def to_mel(hz):
        return 2595 * np.log10(1 + hz / 700)

    @staticmethod
    def to_hz(mel):
        return 700 * (10**(mel / 2595) - 1)


    def __init__(
        self,
        out_channels: int,
        kernel_size: int,
        sample_rate: int = 16000,
        in_channels: int = 1,
        stride: int = 1,
        padding: int = 0,
        dilation: int = 1,
    ):

        super().__init__()

        if in_channels != 1:
            raise ValueError("SincConv supports only one input channel.")

        self.out_channels = out_channels
        self.sample_rate = sample_rate

        # Ensure odd kernel size
        self.kernel_size = kernel_size + (kernel_size % 2 == 0)

        self.stride = stride
        self.padding = padding
        self.dilation = dilation

        self._init_filters()


    def _init_filters(self):

        NFFT = 512

        f = int(self.sample_rate / 2) * np.linspace(
            0, 1, int(NFFT / 2) + 1
        )

        fmel = self.to_mel(f)

        mel_points = np.linspace(
            fmel.min(),
            fmel.max(),
            self.out_channels + 1,
        )

        hz_points = self.to_hz(mel_points)

        self.hsupp = torch.arange(
            -(self.kernel_size - 1) / 2,
            (self.kernel_size - 1) / 2 + 1,
        )

        band_pass = torch.zeros(
            self.out_channels,
            self.kernel_size,
        )

        for i in range(self.out_channels):

            fmin = hz_points[i]
            fmax = hz_points[i + 1]

            h_high = (
                2 * fmax / self.sample_rate
            ) * np.sinc(
                2 * fmax * self.hsupp / self.sample_rate
            )

            h_low = (
                2 * fmin / self.sample_rate
            ) * np.sinc(
                2 * fmin * self.hsupp / self.sample_rate
            )

            hideal = h_high - h_low

            band_pass[i, :] = (
                torch.tensor(np.hamming(self.kernel_size))
                * torch.tensor(hideal)
            )

        self.register_buffer("band_pass", band_pass)


    def forward(self, x: torch.Tensor):

        # x: (B, 1, T)

        filt = self.band_pass.view(
            self.out_channels,
            1,
            self.kernel_size,
        )

        return F.conv1d(
            x,
            filt,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            groups=1,
        )


# ============================================================
# Conv Block
# ============================================================

class Conv2DBlock_S(nn.Module):

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        is_first_block: bool = False,
    ):

        super().__init__()

        # Optional normalizer
        self.normalizer = None

        if not is_first_block:

            self.normalizer = nn.Sequential(
                nn.BatchNorm2d(in_channels),
                nn.SELU(inplace=True),
            )


        self.layers = nn.Sequential(

            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=(2, 5),
                padding=(1, 2),
            ),

            nn.BatchNorm2d(out_channels),
            nn.SELU(inplace=True),

            nn.Conv2d(
                out_channels,
                out_channels,
                kernel_size=(2, 3),
                padding=(0, 1),
            ),
        )


        self.downsampler = None

        if in_channels != out_channels:

            self.downsampler = nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=(1, 3),
                padding=(0, 1),
            )


        self.pooling = nn.MaxPool2d(kernel_size=(1, 6))


    def forward(self, x):

        identity = x

        if self.downsampler is not None:
            identity = self.downsampler(identity)

        if self.normalizer is not None:
            x = self.normalizer(x)

        x = self.layers(x) + identity

        x = self.pooling(x)

        return x


# ============================================================
# Frontend Network
# ============================================================

class Frontend_S(nn.Module):

    """
    Sinc + CNN Frontend
    """

    def __init__(
        self,
        sinc_kernel_size: int = 128,
        sample_rate: int = 16000,
    ):

        super().__init__()

        # Sinc Layer
        self.sinc_layer = SincConv(
            in_channels=1,
            out_channels=70,
            kernel_size=sinc_kernel_size,
            sample_rate=sample_rate,
        )

        # BatchNorm
        self.bn = nn.BatchNorm2d(num_features=1)

        self.selu = nn.SELU(inplace=True)

        # Conv Blocks
        self.conv_blocks = nn.Sequential(

            Conv2DBlock_S(1,  32, is_first_block=True),
            Conv2DBlock_S(32, 32),
            Conv2DBlock_S(32, 64),
            Conv2DBlock_S(64, 64),

        )


    def forward(self, x):

        # x : (B, T)

        x = x.unsqueeze(1)              # (B,1,T)

        x = self.sinc_layer(x)          # (B,70,T')

        x = x.unsqueeze(1)              # (B,1,70,T')

        x = F.max_pool2d(
            torch.abs(x),
            kernel_size=(3, 3),
        )

        x = self.bn(x)

        LFM = self.selu(x)

        HFM = self.conv_blocks(LFM)

        return HFM
