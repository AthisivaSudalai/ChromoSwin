# Running ChromoSwin — Kaggle & Laptop Guide

---

## Option 1: Kaggle (Recommended)

Free P100/T4 GPU, 30 GPU hours/week, no OOM risk.

### Step 1 — Upload your dataset

1. Go to [kaggle.com](https://kaggle.com) → **Datasets** → **New Dataset**
2. Zip your `data_full/` folder and upload it
3. Name it `chromswin-data` → click **Create**

### Step 2 — Create a new Notebook

1. Go to **Code** → **New Notebook**
2. Right panel → **Session options** → set **Accelerator** to `GPU T4 x2` or `P100`
3. Set **Persistence** to `Files`

### Step 3 — Add your dataset to the Notebook

1. Right panel → **Input** → **Add Input** → search `chromswin-data`
2. It will be mounted at `/kaggle/input/chromswin-data/`

### Step 4 — Paste these cells into the Notebook

```python
# Cell 1 — clone repo
!git clone https://github.com/AthisivaSudalai/ChromoSwin /kaggle/working/ChromoSwin
%cd /kaggle/working/ChromoSwin
```

```python
# Cell 2 — install dependencies
!pip install transformers timm einops scikit-learn
```

```python
# Cell 3 — link dataset into expected path
import os

src = '/kaggle/input/chromswin-data/data_full'
dst = '/kaggle/working/ChromoSwin/data_full'

if os.path.lexists(dst):
    os.remove(dst)

os.symlink(src, dst)
print(f"Linked: {src} -> {dst}")
```

```python
# Cell 4 — run training (uncomment the model you want)
!python main_swin.py
# !python main_vit.py
# !python main_hmfo.py
```

```python
# Cell 5 — save results before session ends
import shutil
shutil.copytree('results', '/kaggle/working/results', dirs_exist_ok=True)
```

After training, download results from the **Output** tab on the right panel.

---

## Option 2: Laptop (RTX 3050 6GB)

### Step 1 — Check CUDA is working

```bash
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

If it prints `False`, install PyTorch with CUDA support from [pytorch.org/get-started/locally](https://pytorch.org/get-started/locally/).

### Step 2 — Install dependencies

```bash
pip install transformers timm einops scikit-learn torchvision
```

### Step 3 — Reduce batch size to avoid OOM

In `main_swin.py`, `main_vit.py`, and `main_hmfo.py`, change `batch_size=32` to `batch_size=8`:

```python
train_loader, val_loader, test_loader = get_dataloaders(
    data_dir='data_full', batch_size=8   # was 32
)
```

> If you still get an OOM error, drop to `batch_size=4`.

### Step 4 — Make sure your folder structure is correct

```
ChromoSwin/
├── data_full/
│   ├── train/
│   ├── val/
│   └── test/
├── main_swin.py
├── main_vit.py
├── main_hmfo.py
└── src/
```

### Step 5 — Run training

```bash
cd d:/Github/ChromoSwin
python main_swin.py
```

### Step 6 — Monitor GPU memory (optional)

Open a second terminal and run:

```bash
nvidia-smi -l 2
```

Watch the memory column — if it stays below 5.5 GB you're fine. If it spikes to 6 GB and crashes, reduce batch size further.

---

## Quick Reference

| | Kaggle | Laptop |
|---|---|---|
| GPU | P100 / T4 (free) | RTX 3050 6GB |
| Batch size | 32 (no change needed) | 8 |
| Session limit | 12 hours | No limit |
| GPU hours | 30 hrs/week free | Unlimited |
| Best for | Full 50-epoch runs | Quick experiments |
