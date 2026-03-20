

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