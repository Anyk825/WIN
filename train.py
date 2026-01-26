# train.py

import numpy as np
import torch
import torch.nn as nn

from tqdm import tqdm


# ---------------- Local Imports ---------------- #

from config import exp_cfg, SAVE_PATH
from utils.device import get_device
from utils.metrics import calculate_eer, compute_tDCF

from data.dataloader import create_dataloaders

from models.WIN import WIN
from models.preprocess import PreEmphasis


# ============================================================
# Main
# ============================================================

def main():

    # ---------------- Device ---------------- #
    device = get_device()


    # ---------------- Data ---------------- #
    train_loader, val_loader, _ = create_dataloaders()


    # ---------------- Model ---------------- #
    model = WIN(
        sample_rate=exp_cfg.SAMPLE_RATE,
        pre_emphasis=exp_cfg.PRE_EMPHASIS,
        transformer_hidden=exp_cfg.TRANSFORMER_HIDDEN,
        n_encoder=2,
        C=64,
        wavelet_type=exp_cfg.WAVELET_TYPE,
    ).to(device)



    # ---------------- Pre-Emphasis ---------------- #
    pre = PreEmphasis(
        exp_cfg.PRE_EMPHASIS
    ).to(device)


    # ---------------- Optimizer & Loss ---------------- #
    opt = torch.optim.Adam(
        model.parameters(),
        lr=exp_cfg.LR,
    )

    criterion = nn.BCELoss()


    # ---------------- t-DCF Parameters ---------------- #
    Pfa_asv = 0.05
    Pmiss_asv = 0.01
    Pfa_spoof_asv = 0.05

    cost_model = {
        "Cmiss": 1,
        "Cfa": 10,
        "Cfa_spoof": 10,
        "Ptar": 0.9801,
        "Pnon": 0.0099,
        "Pspoof": 0.01,
    }


    # ---------------- Training ---------------- #
    best_val_eer = 1.0


    for epoch in range(1, exp_cfg.EPOCHS + 1):

        # === TRAIN ===
        model.train()

        total_loss = 0.0
        total_samples = 0


        pbar = tqdm(
            train_loader,
            desc=f"Epoch {epoch}/{exp_cfg.EPOCHS} [Train]",
            leave=True,
        )


        for wav, label in pbar:

            wav = wav.to(device)
            label = label.to(device)

            wav = pre(wav)


            opt.zero_grad()

            pred = model(wav).squeeze(-1)

            loss = criterion(pred, label)

            loss.backward()

            opt.step()


            bs = wav.size(0)

            total_loss += loss.item() * bs

            total_samples += bs


            pbar.set_postfix(
                loss=f"{total_loss / total_samples:.4f}"
            )


        avg_train_loss = total_loss / total_samples


        # === VALIDATE ===
        model.eval()

        val_loss = 0.0
        val_samples = 0

        all_scores = []
        all_labels = []


        with torch.no_grad():

            pbar = tqdm(
                val_loader,
                desc=f"Epoch {epoch}/{exp_cfg.EPOCHS} [Val]",
                leave=True,
            )


            for wav, label in pbar:

                wav = wav.to(device)
                label = label.to(device)

                wav = pre(wav)


                pred = model(wav).squeeze(-1)

                loss = criterion(pred, label)


                bs = wav.size(0)

                val_loss += loss.item() * bs

                val_samples += bs


                all_scores.extend(
                    pred.cpu().numpy()
                )

                all_labels.extend(
                    label.cpu().numpy()
                )


        avg_val_loss = val_loss / val_samples


        eer = calculate_eer(
            all_labels,
            all_scores,
        )


        # --- Compute t-DCF ---
        bona_cm = np.array(all_scores)[
            np.array(all_labels) == 1
        ]

        spoof_cm = np.array(all_scores)[
            np.array(all_labels) == 0
        ]


        tDCF_curve, thr = compute_tDCF(
            bona_cm,
            spoof_cm,
            Pfa_asv,
            Pmiss_asv,
            Pfa_spoof_asv,
            cost_model,
        )


        min_tDCF = np.min(tDCF_curve)


        print(f"\n🧾 Epoch {epoch} Summary:")

        print(f"   Train Loss: {avg_train_loss:.4f}")

        print(f"   Val Loss:   {avg_val_loss:.4f}")

        print(f"   Val EER:    {eer * 100:.2f}%")

        print(f"   min-tDCF:   {min_tDCF:.4f}")


        # === SAVE BEST MODEL ===
        if eer < best_val_eer:

            best_val_eer = eer

            torch.save(model, SAVE_PATH)

            print(
                f"💾 Saved new best model "
                f"(EER={eer*100:.2f}%) to {SAVE_PATH}"
            )


        print("-" * 60)


    print("✅ Training Finished")


# ============================================================
# Entry
# ============================================================

if __name__ == "__main__":

    main()
