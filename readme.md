# ChromoSwin

## Optimizing Chromosomal Karyotyping and Abnormality Detection using Swin Transformer with Hybrid Moth-Flame Optimization

---

## Project Overview

ChromoSwin is a deep learning research framework for automated chromosomal karyotyping and genetic abnormality detection. It compares three model architectures on the AutoKary2022 clinical dataset:

- **Model 1** — Basic Vision Transformer (ViT-Base/16)
- **Model 2** — Swin Transformer Tiny (Swin-T) without optimization
- **Model 3** — Swin-T with Hybrid Moth-Flame Optimization (HMFO) — proposed model

The system classifies individual chromosome images into 24 classes (chromosomes 1–22, X, Y) and detects chromosomal abnormalities such as Down Syndrome (Trisomy 21), Turner Syndrome (45,X), and Klinefelter Syndrome (47,XXY).

---

## Dataset

- **Name:** AutoKary2022
- **Source:** Roboflow Universe — `universe.roboflow.com/karyotypezhongxin/autokary2022`
- **Size:** 848 metaphase spread images, 37,641 chromosome annotations, 24 classes
- **Download:** Automatic via `download_data.py` (requires free Roboflow API key)
- **Note:** Dataset is NOT included in this repository. Run `download_data.py` to download and prepare it.

---

## Requirements

- Python 3.12 (not 3.13 — PyTorch CUDA does not support 3.13 yet)
- NVIDIA GPU with CUDA 11.8 or higher (recommended: 6GB+ VRAM)
- Internet connection for dataset download and model weights

---

## Repository Structure

```
ChromoSwin/
├── main_vit.py              — Train Model 1 (ViT)
├── main_swin.py             — Train Model 2 (Swin-T)
├── main_hmfo.py             — Train Model 3 (Swin-T + HMFO)
├── test_vit.py              — Test and evaluate Model 1
├── test_swin.py             — Test and evaluate Model 2
├── test_hmfo.py             — Test and evaluate Model 3
├── download_data.py            — Downloads and prepares dataset automatically
├── crop_chromosomes.py      — Crops individual chromosomes from COCO annotations
├── augment_xy.py            — Augments chromosome X and Y (50 → 1000 images)
├── balanced_dt.py           — Creates balanced dataset with exact split counts
├── detect_abnormality.py    — Rule-based abnormality detection engine
├── src/
│   ├── dataset.py           — DataLoader and class weight computation
│   ├── vit_model.py         — ViT model definition
│   ├── swin_model.py        — Swin-T model definition
│   ├── train.py             — Shared training loop for all 3 models
│   └── hmfo.py              — Hybrid Moth-Flame Optimization algorithm
└── results/
    ├── vit/                 — Saved ViT model weights and confusion matrix
    ├── swin/                — Saved Swin-T model weights and confusion matrix
    └── swin_hmfo/           — Saved HMFO model weights, best params, confusion matrix
```

---

---

# SECTION 1 — Running on VS Code (Local Machine)

> **Use this section if you have a GPU laptop or desktop with an NVIDIA GPU.**

---

## Step 1 — Prerequisites

Before starting, make sure you have:

1. **Python 3.12** installed — download from `python.org/downloads` and choose Python 3.12.x
   - During installation check **"Add Python to PATH"**
2. **Git** installed — download from `git-scm.com`
3. **VS Code** installed — download from `code.visualstudio.com`
4. **NVIDIA GPU drivers** installed — run `nvidia-smi` in terminal to verify

---

## Step 2 — Check your CUDA version

Open PowerShell and run:

```powershell
nvidia-smi
```

Look for `CUDA Version` in the top right corner of the output. Note the version number — you will need it in Step 5.

---

## Step 3 — Clone the repository

```powershell
git clone https://github.com/AthisivaSudalai/ChromoSwin.git
cd ChromoSwin
```

---

## Step 4 — Create Python 3.12 virtual environment

```powershell
# create environment using Python 3.12 specifically
py -3.12 -m venv venv312

# activate it
venv312\Scripts\activate
```

You should see `(venv312)` at the start of your terminal line. You must activate this every time you open a new terminal.

---

## Step 5 — Install PyTorch with GPU support

Choose the command that matches your CUDA version from Step 2:

**If CUDA 12.1 or higher:**
```powershell
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

**If CUDA 11.8:**
```powershell
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

---

## Step 6 — Install remaining libraries

```powershell
pip install transformers scikit-learn matplotlib numpy pandas Pillow roboflow
```

---

## Step 7 — Verify GPU is working

```powershell
python -c "import torch; print('GPU:', torch.cuda.is_available()); print('Name:', torch.cuda.get_device_name(0))"
```

Expected output:
```
GPU: True
Name: NVIDIA GeForce RTX XXXX
```

If GPU shows `False`, your PyTorch was installed without CUDA. Repeat Step 5 with the correct CUDA version.

---

## Step 8 — Get your Roboflow API key

1. Go to `roboflow.com` and create a free account
2. Click your profile icon → Settings → Copy your API key

---

## Step 9 — Download and prepare the dataset

```powershell
python download_data.py
```

When prompted, paste your Roboflow API key. This script will:
- Download AutoKary2022 dataset (~110MB)
- Crop 37,641 individual chromosome images from full metaphase spreads
- Augment chromosome X and Y from 50 to 1,000 images each
- Create the balanced dataset with 200 train / 40 val / 20 test per class

This takes approximately 15–20 minutes. Do not close the terminal.

---

## Step 10 — Train all 3 models

Run each model one at a time. Wait for each to finish before running the next.

**Model 1 — Basic ViT (~15 minutes on RTX 3050):**
```powershell
py main_vit.py
```

**Model 2 — Swin-T without HMFO (~15 minutes on RTX 3050):**
```powershell
py main_swin.py
```

**Model 3 — Swin-T with HMFO (~2–3 hours on RTX 3050):**
```powershell
py main_hmfo.py
```

HMFO takes longer because it runs 20 moths × 30 iterations × 5 epochs each to find optimal hyperparameters before the final 50-epoch training run.

---

## Step 11 — Test and evaluate all 3 models

```powershell
py test_vit.py
py test_swin.py
py test_hmfo.py
```

Each test script outputs:
- Accuracy, F1-Score, Precision, Recall
- Per-class breakdown for all 24 chromosome types
- Normalized confusion matrix saved to `results/` folder
- Best hyperparameters found by HMFO (for Model 3)

---

## Step 12 — Run abnormality detection

```powershell
py detect_abnormality.py
```

This runs the rule-based Stage 2 engine that takes the 24-class predictions and identifies specific genetic disorders.

---

## Troubleshooting — VS Code

| Error | Fix |
|---|---|
| `ModuleNotFoundError: No module named 'torchvision'` | Run `venv312\Scripts\activate` first |
| `CUDA not available` | Reinstall PyTorch with correct CUDA version (Step 5) |
| `FileNotFoundError: data_balanced_200/train` | Run `python download_data.py` first (Step 9) |
| `RuntimeError: Expected all tensors on same device` | GPU/CPU mismatch — restart terminal and rerun |
| `CUDA out of memory` | Reduce batch size: change `batch_size=32` to `batch_size=16` in main files |
| Terminal shows `>>>` instead of PowerShell | Type `exit()` to leave Python interpreter |

---

---

# SECTION 2 — Running on Google Colab (No GPU Required on Your Machine)

> **Use this section if you do not have a GPU on your machine. Google Colab provides a free NVIDIA T4 GPU in your browser.**

---

## Important notes before starting

- Google Colab sessions disconnect after ~12 hours of inactivity
- Free Colab gives you a T4 GPU (16GB VRAM) — much faster than a laptop GPU
- Save results to Google Drive after each training run so you never lose them
- HMFO training (Model 3) takes 4–5 hours — run it last and leave the browser open

---

## Step 1 — Open Google Colab

Go to `colab.research.google.com` and sign in with your Google account.

Click **New Notebook**.

---

## Step 2 — Enable GPU

Go to **Runtime → Change runtime type → Hardware accelerator → T4 GPU → Save**

---

## Step 3 — Verify GPU (run this first cell)

```python
import torch
print("GPU available:", torch.cuda.is_available())
print("GPU name:", torch.cuda.get_device_name(0))
print("GPU memory:", round(torch.cuda.get_device_properties(0).total_memory / 1024**3, 1), "GB")
```

Expected output:
```
GPU available: True
GPU name: Tesla T4
GPU memory: 15.9 GB
```

---

## Step 4 — Install libraries

```python
!pip install torch torchvision transformers scikit-learn matplotlib roboflow Pillow -q
```

---

## Step 5 — Clone the repository

```python
!git clone https://github.com/AthisivaSudalai/ChromoSwin.git
%cd ChromoSwin
```

---

## Step 6 — Download and prepare dataset

Replace `YOUR_ROBOFLOW_API_KEY` with your actual key from `roboflow.com`:

```python
from roboflow import Roboflow

rf      = Roboflow(api_key="YOUR_ROBOFLOW_API_KEY")
project = rf.workspace("karyotypezhongxin").project("autokary2022")
version = project.version(1)
dataset = version.download("coco")

print("Dataset downloaded!")
```

---

## Step 7 — Prepare dataset

Run each line as a separate cell so you can see progress:

```python
# Cell 1 — crop individual chromosomes from full images
!python crop_chromosomes.py
```

```python
# Cell 2 — augment chromosome X and Y
!python augment_xy.py
```

```python
# Cell 3 — create balanced dataset
!python balanced_dt.py
```

```python
# Cell 4 — verify dataset
import os
total = sum(len(files) for _, _, files in os.walk('data_balanced_200/train'))
print(f"Total training images: {total}")
print(f"Expected: 4800 (200 per class × 24 classes)")
```

---

## Step 8 — Train Model 1 — Basic ViT

```python
!python main_vit.py
```

Expected training time: ~8–10 minutes on T4 GPU.

When finished you will see:
```
Training done. Best Val F1: 0.XXXX
Training complete!
Best F1: 0.XXXX
```

---

## Step 9 — Train Model 2 — Swin-T without HMFO

```python
!python main_swin.py
```

Expected training time: ~8–10 minutes on T4 GPU.

---

## Step 10 — Train Model 3 — Swin-T with HMFO

```python
!python main_hmfo.py
```

Expected training time: **3–4 hours on T4 GPU**. This runs the full HMFO search (20 moths × 30 iterations) before final training. Do not close the browser.

---

## Step 11 — Save results to Google Drive immediately after each model

Run this after EACH model finishes training to prevent losing results if Colab disconnects:

```python
from google.colab import drive
drive.mount('/content/drive')

import shutil, os

# create results folder in Drive if it doesn't exist
os.makedirs('/content/drive/MyDrive/ChromoSwin_results', exist_ok=True)

# copy all results
shutil.copytree(
    '/content/ChromoSwin/results',
    '/content/drive/MyDrive/ChromoSwin_results',
    dirs_exist_ok=True
)

print("Results saved to Google Drive!")
print("Location: My Drive → ChromoSwin_results")
```

---

## Step 12 — Test all 3 models

```python
!python test_vit.py
```

```python
!python test_swin.py
```

```python
!python test_hmfo.py
```

---

## Step 13 — Push results back to GitHub

```python
!git config --global user.email "your_email@gmail.com"
!git config --global user.name "AthisivaSudalai"
!git add results/
!git commit -m "Colab training complete - full dataset results"
!git push origin main
```

If it asks for a password, use a GitHub Personal Access Token (not your GitHub password). Generate one at `github.com → Settings → Developer settings → Personal access tokens`.

---

## Full Colab notebook — complete copy-paste version

If you want to run everything in one go, paste each block below into separate Colab cells in order:

```python
# CELL 1 — setup
!pip install torch torchvision transformers scikit-learn matplotlib roboflow Pillow -q
!git clone https://github.com/AthisivaSudalai/ChromoSwin.git
%cd ChromoSwin

# CELL 2 — dataset
from roboflow import Roboflow
rf = Roboflow(api_key="YOUR_API_KEY")
project = rf.workspace("karyotypezhongxin").project("autokary2022")
version = project.version(1)
dataset = version.download("coco")

# CELL 3 — prepare data
!python crop_chromosomes.py
!python augment_xy.py
!python balanced_dt.py

# CELL 4 — train all 3 models (run one at a time)
!python main_vit.py

# CELL 5
!python main_swin.py

# CELL 6 — this takes 3-4 hours
!python main_hmfo.py

# CELL 7 — test all models
!python test_vit.py
!python test_swin.py
!python test_hmfo.py

# CELL 8 — save to Drive
from google.colab import drive
drive.mount('/content/drive')
import shutil, os
os.makedirs('/content/drive/MyDrive/ChromoSwin_results', exist_ok=True)
shutil.copytree('/content/ChromoSwin/results',
                '/content/drive/MyDrive/ChromoSwin_results',
                dirs_exist_ok=True)
print("All results saved to Google Drive!")
```

---

## Troubleshooting — Google Colab

| Problem | Fix |
|---|---|
| Session disconnected mid-training | Results from completed models are in Google Drive. Re-run only the remaining models. |
| `CUDA out of memory` | Reduce batch size: open main file and change `batch_size=32` to `batch_size=16` |
| `ModuleNotFoundError` | Re-run Cell 1 to reinstall libraries — Colab resets on reconnect |
| Dataset download fails | Check your Roboflow API key is correct and has not expired |
| Push to GitHub fails | Use Personal Access Token instead of password |
| T4 GPU not available | Colab free tier has GPU usage limits — try again after a few hours |

---

## Expected Final Results

After all 3 models are trained and tested, your paper comparison table should show:

| Model | Expected Accuracy | Expected F1 |
|---|---|---|
| Basic ViT | 75 – 82% | 0.74 – 0.81 |
| Swin-T no HMFO | 80 – 87% | 0.79 – 0.86 |
| Swin-T + HMFO | 85 – 92% | 0.84 – 0.91 |

---

## Contact

**Author:** Athisiva Sudalai
**GitHub:** github.com/AthisivaSudalai/ChromoSwin
**Dataset:** AutoKary2022 — universe.roboflow.com/karyotypezhongxin/autokary2022

---

*If you use this code in your research, please cite the ChromoSwin paper.*