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

class ASVspoofFolderDataset(Dataset):
    """
    Loads audio files from:
        root/
          ├── bonafide/
          └── spoof/

    Returns:
        waveform: (T,)
        label: float (1 = bonafide, 0 = spoof)
    """

    def __init__(
        self,
        root_dir: str,
        sample_rate: int = 16000,
        duration_sec: int = 4,
    ):

        self.root_dir = root_dir
        self.sample_rate = sample_rate
        self.duration_sec = duration_sec

        self.audio_paths = []
        self.labels = []

        self._load_files()


    def _load_files(self):

        for label_name, label_value in [("bonafide", 1), ("spoof", 0)]:

            class_dir = os.path.join(self.root_dir, label_name)

            if not os.path.exists(class_dir):
                continue

            for file in os.listdir(class_dir):

                if file.endswith((".flac", ".wav")):

                    path = os.path.join(class_dir, file)

                    self.audio_paths.append(path)
                    self.labels.append(label_value)

        print(f"📁 Loaded {len(self.audio_paths)} files from {self.root_dir}")


    def __len__(self):
        return len(self.audio_paths)


    def _load_audio(self, path):

        # FLAC → soundfile
        if path.lower().endswith(".flac"):

            wav_np, sr = sf.read(path)

            if wav_np.ndim > 1:
                wav_np = wav_np.mean(axis=1)

            wav = torch.tensor(
                wav_np, dtype=torch.float32
            ).unsqueeze(0)

        # WAV → torchaudio
        else:

            wav, sr = torchaudio.load(path)

        return wav, sr


    def _fix_length(self, wav):

        num_samples = int(self.sample_rate * self.duration_sec)

        # Crop
        if wav.size(1) > num_samples:

            start = random.randint(
                0, wav.size(1) - num_samples
            )

            wav = wav[:, start:start + num_samples]

        # Pad
        elif wav.size(1) < num_samples:

            wav = F.pad(
                wav,
                (0, num_samples - wav.size(1))
            )

        return wav


    def __getitem__(self, idx):

        path = self.audio_paths[idx]

        label = torch.tensor(
            self.labels[idx],
            dtype=torch.float32
        )

        wav, sr = self._load_audio(path)

        # Resample
        if sr != self.sample_rate:

            wav = torchaudio.functional.resample(
                wav, sr, self.sample_rate
            )

        # Crop / Pad
        wav = self._fix_length(wav)

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
