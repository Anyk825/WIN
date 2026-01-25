# tests/test_forward.py

import torch

from config import exp_cfg, HAS_TORCHINFO
from utils.device import get_device

from models.WIN import WIN


def main():

    # ---------------- Device ----------------
    device = get_device(verbose=True)


    # ---------------- Build Model ----------------
    model = WIN(
        sample_rate=exp_cfg.SAMPLE_RATE,
        pre_emphasis=exp_cfg.PRE_EMPHASIS,
        transformer_hidden=exp_cfg.TRANSFORMER_HIDDEN,
    ).to(device)

    model.eval()


    # ---------------- Dummy Input ----------------
    B = 2

    T = int(
        exp_cfg.SAMPLE_RATE
        * exp_cfg.TRAIN_DURATION
    )

    dummy_audio = torch.randn(B, T).to(device)


    # ---------------- Forward Pass ----------------
    with torch.no_grad():

        out = model(dummy_audio)


    print(
        "Model output shape:",
        out.shape,
        "| values ~",
        (out.min().item(), out.max().item()),
    )


    # ---------------- Torchinfo Summary ----------------
    if HAS_TORCHINFO:

        try:

            from torchinfo import summary

            summary(
                model,
                input_size=(B, T),
            )

        except Exception as e:

            print(
                "torchinfo summary error (safe to ignore):",
                e,
            )


if __name__ == "__main__":
    main()
