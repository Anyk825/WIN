# utils/device.py

import torch


def get_device(verbose: bool = True):
    """
    Check CUDA availability and return device.
    """

    if verbose:
        print("CUDA available:", torch.cuda.is_available())
        print("GPU count:", torch.cuda.device_count())

        for i in range(torch.cuda.device_count()):
            print(f"[{i}] {torch.cuda.get_device_name(i)}")

        print("Torch version:", torch.__version__)
        print("Hi!!!")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    return device
