# model_info.py

import torch
import numpy as np

from torchinfo import summary
from fvcore.nn import FlopCountAnalysis

# ---------------- Local Imports ---------------- #

from config import exp_cfg
from utils.device import get_device

from models.WIN import WIN
from models.preprocess import PreEmphasis


# --------------------------------------------------------------
# 1. Parameter Count + Model Size
# --------------------------------------------------------------

def print_model_params(model):

    trainable = sum(
        p.numel() for p in model.parameters()
        if p.requires_grad
    )

    total = sum(
        p.numel() for p in model.parameters()
    )

    size_mb = sum(
        p.numel() * p.element_size()
        for p in model.parameters()
    ) / (1024 ** 2)


    print("\n" + "=" * 60)
    print("MODEL PARAMETER SUMMARY")
    print("=" * 60)

    print(f"{'Trainable params':<25}: {trainable:,}")
    print(f"{'Total params'    :<25}: {total:,}")
    print(f"{'Model size (MiB)':<25}: {size_mb:.2f}")

    print("=" * 60 + "\n")


# --------------------------------------------------------------
# Main
# --------------------------------------------------------------

def main():

    # ---------------- Device ---------------- #
    device = get_device()


    # ---------------- Build Model ---------------- #
    model = WIN(
        sample_rate=exp_cfg.SAMPLE_RATE,
        pre_emphasis=exp_cfg.PRE_EMPHASIS,
        transformer_hidden=exp_cfg.TRANSFORMER_HIDDEN,
        n_encoder=2,
        C=64,
    ).to(device)


    # ---------------- Pre-Emphasis ---------------- #
    pre = PreEmphasis(
        exp_cfg.PRE_EMPHASIS
    ).to(device)


    model.eval()


    # ==========================================================
    # PARAMETER COUNT
    # ==========================================================

    print_model_params(model)


    # ==========================================================
    # FLOPs / MACs
    # ==========================================================

    # Fallback duration (seconds)
    max_len_sec = getattr(
        exp_cfg,
        "MAX_LEN_SEC",
        4.0,
    )


    max_samples = int(
        exp_cfg.SAMPLE_RATE * max_len_sec
    )


    # Dummy waveform
    dummy_wav = torch.randn(
        1,
        max_samples,
        device=device,
    )


    # Apply pre-emphasis (same as training)
    if pre is not None:
        dummy_wav = pre(dummy_wav)


    # ---- fvcore (accurate FLOPs) ----
    flops = FlopCountAnalysis(
        model,
        dummy_wav,
    )

    macs = flops.total()

    flops_2 = macs * 2


    print("\n" + "=" * 60)
    print("FLOPs / MACs (per forward pass)")
    print("=" * 60)

    print(f"{'Input shape'   :<25}: {list(dummy_wav.shape)}")
    print(f"{'MACs'          :<25}: {macs/1e9:.3f} G")
    print(f"{'FLOPs'         :<25}: {flops_2/1e9:.3f} G")

    print("=" * 60 + "\n")


    # ==========================================================
    # TORCHINFO SUMMARY
    # ==========================================================

    print("Detailed layer-wise breakdown (torchinfo):\n")


    summary(
        model,
        input_data=dummy_wav,
        col_names=[
            "input_size",
            "output_size",
            "num_params",
            "mult_adds",
        ],
        depth=4,
        verbose=0,
    )


    # ==========================================================
    # GFLOPs / Second
    # ==========================================================

    seconds = max_samples / exp_cfg.SAMPLE_RATE

    gflops_per_sec = flops_2 / 1e9 / seconds


    print(
        f"\nGFLOPs per second of audio : "
        f"{gflops_per_sec:.3f}"
    )


# --------------------------------------------------------------
# Entry
# --------------------------------------------------------------

if __name__ == "__main__":

    main()
