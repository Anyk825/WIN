import torch
import torch.nn as nn
from models.wavelets import BumpWavelet

class WIN(nn.Module):
    """
    Wavelet Integrated Network (WIN-Bump)
    """

    def __init__(self, input_dim, hidden_dim, num_layers):
        super().__init__()

        self.wavelet = BumpWavelet(input_dim, hidden_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=4,
            batch_first=True
        )

        self.encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers
        )

        self.classifier = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        # x: (B, T, C)
        x = self.wavelet(x)
        x = self.encoder(x)
        x = x.mean(dim=1)
        return self.classifier(x)
