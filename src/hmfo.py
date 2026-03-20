# src/hmfo.py
# Hybrid Moth-Flame Optimization
# Combines MFO logarithmic spiral + Levy Flight for global exploration
# Searches for best hyperparameters for Swin-T training

import math

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from sklearn.metrics import f1_score
from src.swin_model import build_swin, freeze_backbone, unfreeze_backbone


# ── SEARCH SPACE ──────────────────────────────────────────────
# These are the hyperparameters HMFO will search through
# Format: [min_value, max_value]
SEARCH_SPACE = {
    'learning_rate': [1e-5, 1e-2],   # how fast model learns
    'dropout_rate':  [0.1,  0.5 ],   # regularization strength
    'weight_decay':  [1e-5, 1e-1],   # L2 regularization
}
N_PARAMS = len(SEARCH_SPACE)         # 3 parameters to optimize


def levy_flight(size, beta=1.5):
    """
    Levy Flight — generates random long-distance jumps
    This is the HYBRID part of HMFO
    Prevents getting stuck in local optima
    """
    # Mantegna's algorithm for Levy distribution
    # CORRECT — use Python's built-in math module instead

    sigma = (
    math.gamma(1 + beta) * np.sin(np.pi * beta / 2) /
    (math.gamma((1 + beta) / 2) * beta * 2 ** ((beta - 1) / 2))
    ) ** (1 / beta)

    u = np.random.normal(0, sigma, size)
    v = np.random.normal(0, 1, size)
    step = u / (np.abs(v) ** (1 / beta))
    return step


def decode_params(moth):
    """
    Converts a moth's position [0,1] range into actual hyperparameter values
    moth = array of 3 numbers between 0 and 1
    returns = dict with actual learning_rate, dropout_rate, weight_decay
    """
    keys  = list(SEARCH_SPACE.keys())
    params = {}
    for i, key in enumerate(keys):
        lo, hi = SEARCH_SPACE[key]
        if key == 'learning_rate' or key == 'weight_decay':
            # log scale for learning rate and weight decay
            # because small differences matter more at small values
            val = 10 ** (np.log10(lo) + moth[i] * (np.log10(hi) - np.log10(lo)))
        else:
            # linear scale for dropout
            val = lo + moth[i] * (hi - lo)
        params[key] = float(val)
    return params


def evaluate_params(params, train_loader, val_loader,
                    num_classes=24, quick_epochs=5):
    """
    Trains Swin-T for a few epochs with given hyperparameters
    Returns the validation F1 score — this is the FITNESS FUNCTION
    Higher F1 = better hyperparameters = better moth position
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # build fresh model for each evaluation
    model = build_swin(num_classes=num_classes, pretrained=True)
    freeze_backbone(model)
    model = model.to(device)

    optimizer = AdamW(
        model.parameters(),
        lr=params['learning_rate'],
        weight_decay=params['weight_decay']
    )
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=quick_epochs)
    criterion = nn.CrossEntropyLoss()

    # apply dropout rate to classifier
    model.classifier = nn.Sequential(
        nn.Dropout(p=params['dropout_rate']),
        nn.Linear(model.config.hidden_size, num_classes)
    ).to(device)

    # quick training loop — just enough to evaluate quality
    model.train()
    for epoch in range(quick_epochs):
        if epoch == 2:
            unfreeze_backbone(model)
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            with torch.amp.autocast('cuda'):
                outputs = model(pixel_values=images).logits
                loss    = criterion(outputs, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        scheduler.step()

    # evaluate on validation set
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for images, labels in val_loader:
            images  = images.to(device)
            preds   = model(pixel_values=images).logits.argmax(dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())

    f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)

    # clean up GPU memory
    del model
    torch.cuda.empty_cache()

    return f1


def run_hmfo(train_loader, val_loader,
             n_moths=10, n_iterations=20, num_classes=24):
    """
    Main HMFO algorithm
    n_moths     = number of hyperparameter combinations to try at once
    n_iterations = how many rounds of optimization to run
    Returns the best hyperparameters found
    """

    print("\n" + "="*55)
    print("  HMFO — Hybrid Moth-Flame Optimization")
    print(f"  Moths: {n_moths} | Iterations: {n_iterations}")
    print(f"  Searching: learning_rate, dropout_rate, weight_decay")
    print("="*55)

    # ── Step 1: Initialize moths randomly ─────────────────────
    # Each moth = one combination of hyperparameters
    # Position values are between 0 and 1 (decoded later)
    moths = np.random.uniform(0, 1, (n_moths, N_PARAMS))

    best_fitness  = -1.0
    best_moth     = None
    best_params   = None
    history       = []

    for iteration in range(n_iterations):

        print(f"\nIteration {iteration+1}/{n_iterations}")

        # ── Step 2: Evaluate each moth ─────────────────────────
        fitness = np.zeros(n_moths)
        for i in range(n_moths):
            params = decode_params(moths[i])
            f1     = evaluate_params(
                params, train_loader, val_loader,
                num_classes=num_classes, quick_epochs=5
            )
            fitness[i] = f1
            print(f"  Moth {i+1:2d} | lr={params['learning_rate']:.6f} "
                  f"dropout={params['dropout_rate']:.3f} "
                  f"wd={params['weight_decay']:.6f} | F1={f1:.4f}")

        # ── Step 3: Sort moths by fitness ──────────────────────
        sorted_idx = np.argsort(fitness)[::-1]  # best first
        moths      = moths[sorted_idx]
        fitness    = fitness[sorted_idx]

        # ── Step 4: Update best solution ──────────────────────
        if fitness[0] > best_fitness:
            best_fitness = fitness[0]
            best_moth    = moths[0].copy()
            best_params  = decode_params(best_moth)
            print(f"  --> New best! F1={best_fitness:.4f} | {best_params}")

        history.append(best_fitness)

        # ── Step 5: Number of flames decreases over time ──────
        # Early iterations = many flames (explore broadly)
        # Later iterations = fewer flames (exploit best regions)
        n_flames = max(1, int(n_moths - iteration * (n_moths - 1) / n_iterations))

        # ── Step 6: Update moth positions ─────────────────────
        for i in range(n_moths):
            # pick which flame this moth spirals toward
            flame_idx = min(i, n_flames - 1)
            flame     = moths[flame_idx]

            # MFO logarithmic spiral update
            t = np.random.uniform(-1, 1, N_PARAMS)
            b = 1.0
            D = np.abs(flame - moths[i])
            moths[i] = D * np.exp(b * t) * np.cos(2 * np.pi * t) + flame

            # Levy Flight perturbation (the HYBRID component)
            # Applied with 30% probability — occasional big jumps
            if np.random.random() < 0.3:
                levy    = levy_flight(N_PARAMS)
                step    = 0.01 * levy * (moths[i] - best_moth)
                moths[i] = moths[i] + step

            # keep positions within [0, 1] bounds
            moths[i] = np.clip(moths[i], 0, 1)

    print("\n" + "="*55)
    print(f"  HMFO COMPLETE")
    print(f"  Best F1 found : {best_fitness:.4f}")
    print(f"  Best params   : {best_params}")
    print("="*55)

    return best_params, best_fitness, history