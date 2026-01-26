# models/WIN_classifier.py

import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# Base Wavelet FAN
# ============================================================

class BaseWaveletFAN(nn.Module):

    def __init__(self, d_head, d_phi=None):

        super().__init__()

        if d_phi is None:
            d_phi = d_head

        assert d_phi % 2 == 0

        self.linear_p = nn.Linear(d_head, d_phi // 2)
        self.linear_g = nn.Linear(d_head, d_phi // 2)

        self.act = nn.GELU()

        self.log_scale = nn.Parameter(torch.zeros(1))
        self.gate = nn.Parameter(torch.randn(1))


    def content_branch(self, x):

        return self.act(self.linear_g(x))


    def forward(self, x):

        raise NotImplementedError
        

# ============================================================
# Bump Wavelet
# ============================================================

class BumpWavelet(BaseWaveletFAN):

    def forward(self, x):

        z = self.linear_p(x)
        g = self.content_branch(x)

        s = torch.exp(self.log_scale) + 1e-4
        u = z / s

        wavelet = torch.zeros_like(u)

        mask = u.abs() < 1.0

        wavelet[mask] = torch.exp(
            -1.0 / (1.0 - u[mask] ** 2)
        )

        wavelet = wavelet / (
            wavelet.std(dim=-1, keepdim=True) + 1e-5
        )

        gate = torch.sigmoid(self.gate)

        return torch.cat([
            gate * wavelet,
            (1 - gate) * g
        ], dim=-1)


# ============================================================
# DoG Wavelet
# ============================================================

class DoGWavelet(BaseWaveletFAN):

    def forward(self, x):

        z = self.linear_p(x)
        g = self.content_branch(x)

        s = torch.exp(self.log_scale) + 1e-4
        u = z / s

        wavelet = u * torch.exp(-0.5 * u**2)

        wavelet = wavelet / (
            wavelet.std(dim=-1, keepdim=True) + 1e-5
        )

        gate = torch.sigmoid(self.gate)

        return torch.cat([
            gate * wavelet,
            (1 - gate) * g
        ], dim=-1)


# ============================================================
# Morlet Wavelet
# ============================================================

class MorletWavelet(BaseWaveletFAN):

    def __init__(self, d_head, d_phi=None):

        super().__init__(d_head, d_phi)

        self.freq = nn.Parameter(torch.randn(1))


    def forward(self, x):

        z = self.linear_p(x)
        g = self.content_branch(x)

        s = torch.exp(self.log_scale) + 1e-6
        omega = self.freq

        wavelet = torch.exp(
            -0.5 * (z / s) ** 2
        ) * torch.cos(omega * z)

        gate = torch.sigmoid(self.gate)

        return torch.cat([
            gate * wavelet,
            (1 - gate) * g
        ], dim=-1)


# ============================================================
# Morse Wavelet
# ============================================================

class MorseWavelet(BaseWaveletFAN):

    def __init__(self, d_head, d_phi=None, beta=1.0, gamma=3.0):

        super().__init__(d_head, d_phi)

        self.beta = beta
        self.gamma = gamma


    def forward(self, x):

        z = self.linear_p(x)
        g = self.content_branch(x)

        s = torch.exp(self.log_scale) + 1e-4
        u = z / s

        abs_u = torch.abs(u)

        wavelet = (
            abs_u ** self.beta
        ) * torch.exp(
            -(abs_u ** self.gamma)
        )

        wavelet = wavelet / (
            wavelet.std(dim=-1, keepdim=True) + 1e-5
        )

        gate = torch.sigmoid(self.gate)

        return torch.cat([
            gate * wavelet,
            (1 - gate) * g
        ], dim=-1)

# ============================================================
# Mexican Hat (Ricker) Wavelet
# ============================================================

class MexicanHatWavelet(BaseWaveletFAN):
    """
    Mexican-Hat (Ricker) Wavelet FAN (SAFE)
    """

    def forward(self, x):

        z = self.linear_p(x)
        g = self.content_branch(x)

        s = torch.exp(self.log_scale) + 1e-4
        u = z / s


        # ---- Mexican Hat (Ricker) Wavelet ----
        # ψ(u) = (1 - u^2) * exp(-u^2 / 2)

        wavelet = (1.0 - u**2) * torch.exp(-0.5 * u**2)


        # Normalize
        wavelet = wavelet / (
            wavelet.std(dim=-1, keepdim=True) + 1e-5
        )


        gate = torch.sigmoid(self.gate)

        return torch.cat([
            gate * wavelet,
            (1 - gate) * g
        ], dim=-1)

# ============================================================
# Wavelet Factory
# ============================================================

def get_wavelet_map(name: str, d_head: int):

    name = name.lower()


    if name == "bump":
        return BumpWavelet(d_head)

    elif name == "dog":
        return DoGWavelet(d_head)

    elif name == "morlet":
        return MorletWavelet(d_head)

    elif name == "morse":
        return MorseWavelet(d_head)
        
    elif name ==  "mex_h":
        return MexicanHatWavelet(d_head)
        
    else:
        raise ValueError(
            f"Unknown wavelet type: {name}"
        )


# ============================================================
# Multi-Head Wavelet
# ============================================================

class MultiHeadWavelet(nn.Module):

    def __init__(
        self,
        d_model,
        n_head=8,
        d_head=None,
        wavelet_type="bump",   # <-- NEW for python users, jupyter notebook users have seprate notebooks.
    ):

        super().__init__()

        self.n_head = n_head
        self.wavelet_type = wavelet_type


        if d_head is None:
            assert d_model % n_head == 0
            d_head = d_model // n_head


        self.d_head = d_head


        self.heads = nn.ModuleList([
            get_wavelet_map(wavelet_type, d_head)
            for _ in range(n_head)
        ])


        self.out_proj = nn.Linear(
            n_head * d_head,
            d_model,
        )


    def forward(self, x):

        B, S, D = x.shape

        x = (
            x.view(B, S, self.n_head, self.d_head)
             .permute(0, 2, 1, 3)
        )


        outs = []

        for h in range(self.n_head):

            outs.append(
                self.heads[h](x[:, h])
            )


        out = torch.cat(outs, dim=-1)

        return self.out_proj(out)


# ============================================================
# Transformer Encoder
# ============================================================

class TransformerEncoderLayerWavelet(nn.Module):

    def __init__(
        self,
        d_model=64,
        n_head=8,
        ffn_hidden=2048,
        drop_prob=0.1,
        wavelet_type="bump",   # <-- NEW
    ):

        super().__init__()


        self.attn = MultiHeadWavelet(
            d_model,
            n_head,
            wavelet_type=wavelet_type,
        )


        self.dropout1 = nn.Dropout(drop_prob)

        self.norm1 = nn.LayerNorm(d_model)


        self.ffn = nn.Sequential(
            nn.Linear(d_model, ffn_hidden),
            nn.ReLU(),
            nn.Dropout(drop_prob),
            nn.Linear(ffn_hidden, d_model),
        )


        self.dropout2 = nn.Dropout(drop_prob)

        self.norm2 = nn.LayerNorm(d_model)


    def forward(self, x):

        res = x

        x = self.attn(x)

        x = self.dropout1(x)

        x = self.norm1(x + res)


        res = x

        x = self.ffn(x)

        x = self.dropout2(x)

        x = self.norm2(x + res)


        return x
