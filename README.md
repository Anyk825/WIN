Understood 👍 — you want **one clean, continuous README text** that you can **copy–paste directly** into your `README.md` file, without broken formatting, “copy code” blocks, or separators.

Below is the **complete, unified README** in proper Markdown format.

You can copy everything from `#` to the end and paste it directly.

---

## ✅ **FINAL README.md (Copy Everything Below)**

```markdown
# Wavelet Interface Network (WIN) for Audio Deepfake Detection

This repository contains the official implementation of the **Wavelet Interface Network (WIN)** for audio deepfake detection using wavelet-based feature mapping and transformer-style modeling.

The proposed model integrates signal preprocessing, learnable Sinc-based frontend, positional aggregation, and wavelet-based attention for robust anti-spoofing.

---

## 📌 Features

- End-to-end learning from raw waveform  
- Pre-emphasis filtering  
- Sinc-based convolutional frontend  
- CNN feature extraction  
- Positional encoding  
- Wavelet-based multi-head attention  
- Transformer-style encoder  
- Attention-based sequence pooling  
- EER and t-DCF evaluation  
- FLOPs and parameter analysis  

---

## 📁 Project Structure

```

bump-waveformer/
│
├── train.py              # Training script
├── test.py               # Testing / evaluation
├── model_info.py         # Parameter & FLOPs analysis
│
├── config.py             # Configuration
├── requirements.txt      # Dependencies
│
├── utils/
│   ├── device.py
│   └── metrics.py
│
├── data/
│   └── dataloader.py
│
├── models/
│   ├── preprocess.py
│   ├── frontend.py
│   ├── encoder.py
│   ├── WIN_classifier.py
│   └── WIN.py
│
├── tests/
│   └── test_forward.py
│
└── README.md

````

---

## ⚙️ Requirements

Install dependencies:

```bash
pip install -r requirements.txt
````

Optional tools:

```bash
pip install torchinfo fvcore
```

---

## 📊 Dataset Structure

The dataset must be organized as:

```
dataset_root/
├── train/
│   ├── bonafide/
│   └── spoof/
│
├── dev/
│   ├── bonafide/
│   └── spoof/
│
└── test/
    ├── bonafide/
    └── spoof/
```

Each folder contains `.wav` or `.flac` audio files.

Update dataset paths in `config.py`:

```python
class SysConfig:
    TRAIN_PATH = "path/to/train"
    DEV_PATH   = "path/to/dev"
    TEST_PATH  = "path/to/test"
```

---

## 🚀 Training

To train the model:

```bash
python train.py
```

The best model is saved automatically based on validation EER.

---

## 🧪 Testing

To evaluate on the test set:

```bash
python test.py
```

Outputs:

* Final EER
* Minimum t-DCF

---

## 🔍 Sanity Check

To verify forward pass and architecture:

```bash
python tests/test_forward.py
```

This performs a dummy inference and prints the output shape.

---

## 📐 Model Complexity

To compute parameters and FLOPs:

```bash
python model_info.py
```

This reports:

* Trainable parameters
* Total parameters
* Model size (MiB)
* MACs / FLOPs
* GFLOPs per second
* Layer-wise summary

---

## 🧠 Model Architecture

The proposed WIN architecture consists of:

1. Pre-emphasis filtering
2. Sinc convolution layer
3. CNN frontend
4. Positional aggregation
5. Wavelet-based multi-head attention
6. Transformer encoder layers
7. Sequence pooling
8. Binary classifier

Pipeline:

```
Waveform
   ↓
Pre-Emphasis
   ↓
Sinc + CNN Frontend
   ↓
Positional Encoding
   ↓
Wavelet Transformer
   ↓
Sequence Pooling
   ↓
Classifier
```

---

## 📈 Evaluation Metrics

The following metrics are used:

* Equal Error Rate (EER)
* Tandem Detection Cost Function (t-DCF)

Implemented in:

```
utils/metrics.py
```

---

## 🔧 Configuration

Hyperparameters and paths are defined in:

```
config.py
```

Example:

```python
class ExpConfig:
    BATCH_SIZE = 32
    LR = 8e-4
    EPOCHS = 30
```

Modify this file to tune experiments.

---

## 💾 Checkpoints

The best model is saved at:

```python
SAVE_PATH = "Waveformer.pth"
```

Defined in `config.py`.

---

## 📄 Citation

If you use this work in your research, please cite:

```
@article{win2026,
  title={Wavelet Interface Network for Audio Deepfake Detection},
  author={Author Names},
  journal={Journal/Conference},
  year={2026}
}
```

(Replace with your actual citation.)

---

## 📜 License

This project is intended for academic and research use.

Please contact the authors for commercial usage.

---

## 🙏 Acknowledgements

* ASVspoof Challenge
* PyTorch
* torchaudio
* fvcore
* torchinfo

---

## 📬 Contact

For questions and collaboration:

Author: [Your Name]
Email: [[your.email@domain.com](mailto:your.email@domain.com)]

---

## ✅ What This README Provides

This repository is:

* Reviewer-friendly
* Reproducible
* Professional
* Journal-ready
* Easy to understand

It supports:

* Training
* Testing
* Evaluation
* Model analysis
* Reproducibility
* Research reporting

---

```

---

If you want, next I can help you **customize this with your actual paper title, author names, and venue** so it is submission-ready.
```
