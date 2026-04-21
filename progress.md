

## Project Summary
Building ChromoSwin — Automated Chromosomal Karyotyping using 3 models:
- Model 1: Basic ViT (Vision Transformer) — DONE
- Model 2: Swin-T without HMFO — DONE
- Model 3: Swin-T + HMFO (Hybrid Moth-Flame Optimization) — IN PROGRESS

Goal: Prove each model outperforms the previous one.
Final output: Research paper comparing all 3 models on chromosome classification.

---

## My Setup
- Laptop: Windows, PowerShell terminal
- GPU: NVIDIA GeForce RTX 3050 6GB Laptop GPU
- CUDA Version: 12.9
- Python: 3.12 (venv312 environment)
- Project folder: C:\Users\Athi\OneDrive\Documents\ChromaSwin
- GitHub: github.com/AthisivaSudalai/Research_Project

## How to activate environment every time
```powershell
cd C:\Users\Athi\OneDrive\Documents\ChromaSwin
venv312\Scripts\activate
```

---

## Dataset
- Name: AutoKary2022
- Downloaded via: Roboflow (COCO format) into AutoKary2022-1/ folder
- Raw: 848 metaphase images, 37,641 chromosome annotations, 24 classes
- After cropping: Individual chromosome images in data/train/ (26 folders — includes typo folders)
- Balanced dataset: data_balanced/ — 40 images per class, 24 valid classes
- Split: 28 train / 6 val / 6 test per class
- Total: 672 train | 144 val | 144 test
- Classes: chromosome_1 to chromosome_22, chromosome_X, chromosome_Y

## Why 24 classes not binary (Normal/Abnormal):
Stage 1 = classify each chromosome into 24 types (the AI problem)
Stage 2 = count each type and apply rules to detect abnormality (simple logic)
Cannot detect abnormality without first knowing WHICH chromosome it is.

---

## Project Folder Structure
```
ChromaSwin/
├── main_vit.py           ✅ done
├── main_swin.py          ✅ done
├── test_vit.py           ✅ done
├── test_swin.py          ✅ done (just fixed build_swin bug)
├── balanced_dt.py        ✅ done
├── crop_chromosomes.py   ✅ done
├── download_data.py      ✅ done
├── PROGRESS.md           ✅ this file
├── ChromoSwin_Code_Explainer.html  ✅ done
│
├── src/
│   ├── dataset.py        ✅ done (num_workers=0 for Windows)
│   ├── vit_model.py      ✅ done (bug fixed — pretrained=False works)
│   ├── swin_model.py     ✅ done (bug fixed — pretrained=False works)
│   └── train.py          ✅ done (uses torch.amp.GradScaler)
│
├── data/train/           ✅ 37,641 cropped chromosomes (26 folders)
├── data_balanced/        ✅ 40 per class, 24 classes, all splits filled
├── AutoKary2022-1/       ✅ raw downloaded dataset
└── results/
    ├── vit/best_model.pth    ✅ saved
    └── swin/best_model.pth   ✅ saved
```

---

## Results So Far

### Model 1 — Basic ViT
- Val F1:   0.7107
- Accuracy: 68.75%
- F1 Score: 0.6718
- Precision: 0.7030
- Recall:   0.6875
- Confusion matrix: results/vit/confusion_matrix.png
- Notable: chromosome_Y F1 = 0.00 (complete failure — X and Y confused)
- Notable: chromosome_14, 21, 22 all scored F1 = 0.92 (best classes)

### Model 2 — Swin-T (no HMFO)
- Val F1:    0.7033
- Accuracy:  65.28%
- F1 Score:  0.6380
- Precision: 0.6889
- Recall:    0.6528
- Confusion matrix: results/swin/confusion_matrix.png
- Notable: chromosome_Y F1 = 0.00 (same failure as ViT)
- Notable: chromosome_14 F1 = 1.00 (perfect score)
- Notable: Unstable training — loss jumped at epochs 21 and 41
- Conclusion: Better architecture but needs hyperparameter optimization

### Model 3 — Swin-T + HMFO
=======================================================
  MODEL 3 — SWIN-T + HMFO — TEST RESULTS
=======================================================
  Accuracy  : 0.6250  (62.50%)
  F1 Score  : 0.6101
  Precision : 0.6433
  Recall    : 0.6250
=======================================================

---

## Paper Comparison Table (fill in as results come in)
| Model          | Accuracy | F1 Score | Precision | Recall | Status     |
|----------------|----------|----------|-----------|--------|------------|
| Basic ViT      | 68.75%   | 0.6718   | 0.7030    | 0.6875 | ✅ Done    |
| Swin-T no HMFO | TBD      | TBD      | TBD       | TBD    | Testing    |
| Swin-T + HMFO  | TBD      | TBD      | TBD       | TBD    | Not started|

---

## SMALL DATASET RESULTS (40 images/class — proof of concept)
| Model          | Accuracy | F1     | Precision | Recall |
|----------------|----------|--------|-----------|--------|
| Basic ViT      | 68.75%   | 0.6718 | 0.7030    | 0.6875 |
| Swin-T no HMFO | 65.28%   | 0.6380 | 0.6889    | 0.6528 |
| Swin-T + HMFO  | 62.50%   | 0.6101 | 0.6433    | 0.6250 |

## HMFO Best Hyperparameters Found
- learning_rate: 0.000160
- dropout_rate:  0.216742
- weight_decay:  0.005110

## Current Status
ALL 3 MODELS COMPLETE ON SMALL DATASET
NEXT PHASE: Full dataset training on Google Colab

## Key Bugs Fixed So Far
1. num_workers=2 → num_workers=0 (Windows multiprocessing crash)
2. if __name__ == '__main__' guard added to all main files (Windows requirement)
3. torch.cuda.amp.GradScaler() → torch.amp.GradScaler('cuda') (FutureWarning)
4. torch.cuda.amp.autocast() → torch.amp.autocast('cuda') (FutureWarning)
5. num_classes=24 → num_classes=26 then fixed dataset to exactly 24 classes
6. build_vit(pretrained=False) crashed → fixed both vit_model.py and swin_model.py
7. main_swin.py was in src/ folder → moved to root ChromaSwin/
8. from dataset import → from src.dataset import (wrong import path)
9. IMAGES_PER_CLASS=100 → IMAGES_PER_CLASS=40 (chromosome_X and Y only had 50 images)
10. Python 3.13 incompatible with PyTorch CUDA → installed Python 3.12, created venv312

---

## Libraries Installed in venv312
```powershell
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install transformers scikit-learn matplotlib numpy pandas Pillow roboflow
```

---

## Important Notes for Next Session
- Always activate venv312 first: venv312\Scripts\activate
- Dataset is small (40/class) for testing — final paper needs full dataset run on Colab
- Swin-T training was unstable — this is the argument for why HMFO is needed
- chromosome_Y is hardest class — ViT scored 0.00 on it
- All 3 models must use IDENTICAL data splits for fair comparison
- results/vit/best_model.pth and results/swin/best_model.pth are saved locally
- Push to GitHub regularly: git add . → git commit -m "message" → git push origin main

---

## How to continue in next session — say this to Claude:
"I am building ChromoSwin — chromosome karyotyping using ViT, Swin-T, and Swin-T+HMFO.
Model 1 (ViT) done: F1=0.6718. Model 2 (Swin-T) done: testing now.
Need to build Model 3 — hmfo.py and main_hmfo.py. Here is my PROGRESS.md: [paste this file]"






# ChromoSwin Project — Progress Tracker
# Last updated: March 20, 2026 — 2:30 AM
# Paste this file at the start of any new Claude session to continue exactly where you left off

---

## How to continue in next session — say this to Claude:
"I am building ChromoSwin — chromosome karyotyping classification using ViT, Swin-T,
and Swin-T+HMFO. All 3 models are built and tested on small data. Full dataset (26,223
images) is ready. I need to pull code from GitHub to Google Colab and retrain all 3
models on the full dataset for my final paper results. Here is my PROGRESS.md: [paste this file]"

---

## Project Summary
Building ChromoSwin — Automated Chromosomal Karyotyping and Abnormality Detection
comparing 3 models on AutoKary2022 dataset.

Why 24 classes (not binary):
- Stage 1 (AI): Classify each chromosome into 24 types (your 3 models)
- Stage 2 (Rules): Count each type → detect abnormality → name the disease
- Cannot say "Trisomy 21" without first identifying chromosome 21

Models:
- Model 1: Basic ViT (Vision Transformer) — DONE
- Model 2: Swin-T without HMFO — DONE
- Model 3: Swin-T + HMFO (your proposed solution) — DONE
- Stage 2: detect_abnormality.py — written, not yet tested on full results

GitHub: https://github.com/AthisivaSudalai/ChromoSwin

---

## My Setup
- Laptop   : Windows, PowerShell terminal
- GPU      : NVIDIA GeForce RTX 3050 6GB Laptop GPU
- CUDA     : 12.9
- Python   : 3.12 (venv312 environment)
- Folder   : C:\Users\Athi\OneDrive\Documents\ChromaSwin

Activate environment every time:
  cd C:\Users\Athi\OneDrive\Documents\ChromaSwin
  venv312\Scripts\activate

Libraries installed in venv312:
  pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
  pip install transformers scikit-learn matplotlib numpy pandas Pillow roboflow

---

## Dataset
- Name         : AutoKary2022
- Source       : Roboflow (COCO format) → AutoKary2022-1/ folder
- Raw          : 848 metaphase images, 37,641 chromosome annotations, 24 classes
- After crop   : Individual chromosome images in data/train/
- After augment: chromosome_X and chromosome_Y boosted from 50 → 1000 images each
- Full dataset : data_full/ — ALL available images per class, 70/15/15 split

Full dataset size:
  Train : 26,223 images
  Val   : 5,610 images
  Test  : 5,643 images
  Total : 37,476 images

Class weights (for imbalanced training):
  chromosome_1–15, 18–22 : ~0.90 (many images, low weight)
  chromosome_16           : 1.240
  chromosome_17           : 3.458 (fewer images, higher weight)
  chromosome_X            : 1.561 (augmented from 50 to 1000)
  chromosome_Y            : 1.561 (augmented from 50 to 1000)

---

## Complete Project Folder Structure

ChromaSwin/
├── main_vit.py              ✅ uses data_full, class weights, epochs=50
├── main_swin.py             ✅ uses data_full, class weights, epochs=50
├── main_hmfo.py             ✅ uses data_full, class weights, 20 moths, 30 iterations
├── test_vit.py              ✅ done
├── test_swin.py             ✅ done
├── test_hmfo.py             ✅ done
├── balanced_dt.py           ✅ uses ALL images, outputs to data_full/
├── augment_xy.py            ✅ boosts X and Y to 1000 images each
├── crop_chromosomes.py      ✅ crops COCO annotations to individual images
├── download_data.py         ✅ downloads AutoKary2022 from Roboflow
├── detect_abnormality.py    ✅ Stage 2 rule engine written (test after full training)
├── PROGRESS.md              ✅ this file
├── ChromoSwin_Code_Explainer.html  ✅ interactive code explainer
│
├── src/
│   ├── dataset.py           ✅ get_dataloaders + get_class_weights functions
│   ├── vit_model.py         ✅ build_vit, freeze/unfreeze backbone
│   ├── swin_model.py        ✅ build_swin, freeze/unfreeze backbone
│   ├── train.py             ✅ train_model with class_weights parameter
│   └── hmfo.py              ✅ MFO + Levy Flight hybrid optimizer
│
├── data/train/              ✅ 37,641 cropped chromosomes (26 folders)
├── data_full/               ✅ full dataset with augmented X and Y
├── data_balanced/           ✅ old 50/class dataset (keep for reference)
├── AutoKary2022-1/          ✅ raw downloaded COCO dataset
│
└── results/
    ├── vit/best_model.pth          ✅ saved (small dataset run)
    ├── swin/best_model.pth         ✅ saved (small dataset run)
    └── swin_hmfo/
        ├── best_model.pth          ✅ saved (small dataset run)
        └── best_params.json        ✅ HMFO best hyperparameters found

---

## Small Dataset Results (proof of concept — 50 images/class)

| Model          | Accuracy | F1     | Precision | Recall |
|----------------|----------|--------|-----------|--------|
| Basic ViT      | 74.48%   | 0.7300 | 0.7533    | 0.7448 |
| Swin-T no HMFO | 71.35%   | 0.6957 | 0.7180    | 0.7135 |
| Swin-T + HMFO  | 66.15%   | 0.6436 | 0.6630    | 0.6615 |

NOTE: ViT wins on small data because Swin-T needs more data to show
its architectural advantage. Full dataset will flip this order.
chromosome_Y scored 0.00 on all 3 models — severe class imbalance.
Now fixed with augmentation (50 → 1000 images).

## HMFO Best Hyperparameters Found (small data run)
  learning_rate : 0.000160
  dropout_rate  : 0.216742
  weight_decay  : 0.005110

---

## Full Dataset Results (PENDING — train tomorrow on Colab)

| Model          | Accuracy | F1   | Precision | Recall |
|----------------|----------|------|-----------|--------|
| Basic ViT      | TBD      | TBD  | TBD       | TBD    |
| Swin-T no HMFO | TBD      | TBD  | TBD       | TBD    |
| Swin-T + HMFO  | TBD      | TBD  | TBD       | TBD    |

Expected order after full training:
  Swin-T + HMFO > Swin-T no HMFO > Basic ViT

---

## Key Bugs Fixed (for reference)
1.  num_workers=2 → num_workers=0 (Windows multiprocessing crash)
2.  if __name__ == '__main__' guard added to all main files (Windows)
3.  torch.cuda.amp.GradScaler() → torch.amp.GradScaler('cuda')
4.  torch.cuda.amp.autocast() → torch.amp.autocast('cuda')
5.  num_classes=24 → 26 then fixed dataset to exactly 24 classes
6.  build_vit/build_swin(pretrained=False) crashed → fixed both model files
7.  main_swin.py was in src/ folder → moved to root
8.  from dataset import → from src.dataset import
9.  IMAGES_PER_CLASS=100 → 40 → 50 → None (use all)
10. Python 3.13 → installed Python 3.12, created venv312
11. np.math.gamma → import math; math.gamma (NumPy removed np.math)
12. chromosome_23/24 typo folders → cleaned up
13. class imbalance → added get_class_weights + weighted CrossEntropyLoss
14. X and Y only 50 images → augment_xy.py boosts to 1000 each

---

## Current Status (as of 2:30 AM March 20 2026)
DONE LOCALLY:
  ✅ All 3 models built and working
  ✅ Full dataset prepared (26,223 train images)
  ✅ Class weights implemented
  ✅ chromosome_X and Y augmented to 1000 images each
  ✅ 2-epoch test passed — no errors — F1=0.1343 after 2 epochs
  ✅ Code pushed to GitHub

TOMORROW (Colab):
  ⬜ Pull code from GitHub to Google Colab
  ⬜ Download AutoKary2022 dataset on Colab
  ⬜ Run crop_chromosomes.py on Colab
  ⬜ Run augment_xy.py on Colab
  ⬜ Run balanced_dt.py on Colab
  ⬜ Train Model 1 — py main_vit.py (epochs=50)
  ⬜ Train Model 2 — py main_swin.py (epochs=50)
  ⬜ Train Model 3 — py main_hmfo.py (20 moths, 30 iterations, epochs=50)
  ⬜ Test all 3 — py test_vit.py, test_swin.py, test_hmfo.py
  ⬜ Save results to Google Drive

REMAINING TASKS (2 weeks):
  ⬜ Test detect_abnormality.py on real patient cases
  ⬜ Generate GradCAM heatmaps (XAI visualizations)
  ⬜ Plot training curves for all 3 models
  ⬜ Generate normalized confusion matrices
  ⬜ Write methodology section
  ⬜ Write results section
  ⬜ Write introduction + conclusion + abstract
  ⬜ Make all figures publication quality

---

## Google Colab Setup (do this tomorrow)

Step 1 — Open colab.research.google.com
Step 2 — Runtime → Change runtime type → T4 GPU → Save
Step 3 — Run these cells in order:

Cell 1 — Check GPU:
  import torch
  print("GPU:", torch.cuda.is_available())
  print("Name:", torch.cuda.get_device_name(0))

Cell 2 — Install libraries:
  !pip install torch torchvision transformers scikit-learn matplotlib roboflow Pillow -q

Cell 3 — Clone repo:
  !git clone https://github.com/AthisivaSudalai/ChromoSwin.git
  %cd ChromoSwin

Cell 4 — Download dataset:
from roboflow import Roboflow
rf = Roboflow(api_key="YOUR_ROBOFLOW_API_KEY")
project = rf.workspace("karyotypezhongxin").project("autokary2022")
version = project.version(1)
dataset = version.download("coco")

Cell 5 — Prepare data:
  !python crop_chromosomes.py
  !python augment_xy.py
  !python balanced_dt.py

Cell 6 — Verify data:
import os
total = sum(len(files) for _, _, files in os.walk('data_full/train'))
print(f"Total training images: {total}")

Cell 7 — Train all 3 models:
  !python main_vit.py
  !python main_swin.py
  !python main_hmfo.py

Cell 8 — Test all 3 models:
  !python test_vit.py
  !python test_swin.py
  !python test_hmfo.py

Cell 9 — Save to Google Drive:
  from google.colab import drive
  drive.mount('/content/drive')
  import shutil
  shutil.copytree('/content/ChromoSwin/results',
                  '/content/drive/MyDrive/ChromoSwin_results')
  print("Saved to Google Drive!")

Cell 10 — Push results to GitHub:
  !git config --global user.email "your@email.com"
  !git config --global user.name "AthisivaSudalai"
  !git add results/
  !git commit -m "Colab full dataset training results"
  !git push origin main

IMPORTANT: Do not close the Colab browser tab while training.
If session disconnects, results are safe in Google Drive from Cell 9.
main_hmfo.py will take 4-5 hours — run it last and let it run overnight.

---

## 2-Week Schedule
Week 1 (remaining):
  Day 1 — Colab full training (all 3 models)
  Day 2 — Test results + confusion matrices + GradCAM heatmaps
  Day 3 — detect_abnormality.py testing on real cases
  Day 4 — Training curves + publication quality figures

Week 2:
  Day 5-6  — Write methodology section
  Day 7-8  — Write results section
  Day 9-10 — Write introduction + conclusion + abstract
  Day 11-12 — Polish figures and tables
  Day 13-14 — Final review and submission

---

## Important Notes
- Always activate venv312: venv312\Scripts\activate
- All 3 models MUST use identical data splits — already guaranteed by random.seed(42)
- chromosome_Y is hardest class — augmented but still watch its per-class F1
- Colab sessions last max 12 hours — save to Drive regularly
- results/swin_hmfo/best_params.json has HMFO hyperparameters from small run
- On Colab: HMFO will find new better params on full data (20 moths, 30 iterations)
- Paper novelty: Swin-T+HMFO on medical chromosome data — combination is original