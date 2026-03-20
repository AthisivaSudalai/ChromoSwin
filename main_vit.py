# main_vit.py

from src.dataset   import get_dataloaders
from src.vit_model import build_vit
from src.train     import train_model
import os

# THIS LINE IS THE FIX — required on Windows when using num_workers > 0
if __name__ == '__main__':

    os.makedirs('results/vit', exist_ok=True)

    # Step 1 — load data
    train_loader, val_loader, test_loader = get_dataloaders(
        data_dir='data_balanced', batch_size=32
    )

    # Step 2 — build model
    model = build_vit(num_classes=24, pretrained=True)

    # Step 3 — train
    history = train_model(
        model, train_loader, val_loader,
        lr=1e-4, epochs=50, model_name="vit"
    )

    print('Training complete!')
    print(f'Best F1: {max(history["val_f1"]):.4f}')