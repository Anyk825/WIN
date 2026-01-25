# data/dataloader.py

import os
import random

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import torchaudio
import soundfile as sf

from config import sys_cfg, exp_cfg


# ============================================================
# ASVSpoof Folder Dataset
# ============================================================

class ASVspoofFolderDataset(torch.utils.data.Dataset):

    def __init__(self, root_dir, sample_rate=16000, duration_sec=4):

        self.root_dir = root_dir
        self.sample_rate = sample_rate
        self.duration_sec = duration_sec

        self.audio_paths = []
        self.labels = []


        for label_name, label_value in [("bonafide", 1), ("spoof", 0)]:

            class_dir = os.path.join(root_dir, label_name)

            if os.path.exists(class_dir):

                for file in os.listdir(class_dir):

                    if file.endswith((".wav", ".flac")):

                        self.audio_paths.append(
                            os.path.join(class_dir, file)
                        )

                        self.labels.append(label_value)


        print(f"📁 Loaded {len(self.audio_paths)} files from {root_dir}")


    def __len__(self):

        return len(self.audio_paths)


    def __getitem__(self, idx):

        path = self.audio_paths[idx]

        label = torch.tensor(
            self.labels[idx],
            dtype=torch.float32
        )


        # Load audio
        wav, sr = torchaudio.load(path)   # (C, T)


        # Convert to mono
        if wav.size(0) > 1:

            wav = wav.mean(dim=0, keepdim=True)


        # Resample
        if sr != self.sample_rate:

            wav = torchaudio.functional.resample(
                wav,
                sr,
                self.sample_rate,
            )


        # Crop / Pad
        num_samples = int(
            self.sample_rate * self.duration_sec
        )


        if wav.size(1) > num_samples:

            start = random.randint(
                0,
                wav.size(1) - num_samples
            )

            wav = wav[:, start:start + num_samples]


        elif wav.size(1) < num_samples:

            wav = F.pad(
                wav,
                (0, num_samples - wav.size(1)),
            )


        return wav.squeeze(0), label


# ============================================================
# DataLoader Factory
# ============================================================

def create_dataloaders():

    train_ds = ASVspoofFolderDataset(
        sys_cfg.TRAIN_PATH,
        exp_cfg.SAMPLE_RATE,
        exp_cfg.TRAIN_DURATION,
    )

    val_ds = ASVspoofFolderDataset(
        sys_cfg.DEV_PATH,
        exp_cfg.SAMPLE_RATE,
        exp_cfg.TEST_DURATION,
    )

    test_ds = ASVspoofFolderDataset(
        sys_cfg.TEST_PATH,
        exp_cfg.SAMPLE_RATE,
        exp_cfg.TEST_DURATION,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=exp_cfg.BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=exp_cfg.BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
    )

    test_loader = DataLoader(
        test_ds,
        batch_size=exp_cfg.BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
    )

    return train_loader, val_loader, test_loader
