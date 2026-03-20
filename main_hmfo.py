# main_hmfo.py
from src.dataset    import get_dataloaders
from src.swin_model import build_swin
from src.train      import train_model
from src.hmfo       import run_hmfo
import os
import json

if __name__ == '__main__':

    os.makedirs('results/swin_hmfo', exist_ok=True)

    # same dataset — fair comparison
    train_loader, val_loader, test_loader = get_dataloaders(
        data_dir='data_balanced', batch_size=32
    )

    # ── Phase 1: HMFO finds best hyperparameters ──────────────
    print("Phase 1: Running HMFO to find best hyperparameters...")
    best_params, best_f1, history = run_hmfo(
        train_loader, val_loader,
        n_moths=8,        # 8 hyperparameter combinations per iteration
        n_iterations=10,  # 10 rounds of optimization
        num_classes=24
    )

    # save best params so you don't lose them
    with open('results/swin_hmfo/best_params.json', 'w') as f:
        json.dump(best_params, f, indent=2)
    print(f"\nBest params saved to results/swin_hmfo/best_params.json")
    print(f"Best params: {best_params}")

    # ── Phase 2: Full training with best hyperparameters ──────
    print("\nPhase 2: Full training with HMFO-optimized hyperparameters...")
    model = build_swin(num_classes=24, pretrained=True)

    history = train_model(
        model, train_loader, val_loader,
        lr=best_params['learning_rate'],
        epochs=50,
        model_name="swin_hmfo"
    )

    print('Training complete!')
    print(f'Best F1: {max(history["val_f1"]):.4f}')