# main_vit.py

from src.dataset   import get_dataloaders, get_class_weights
from src.vit_model import build_vit
from src.train     import train_model
import os

if __name__ == '__main__':

    os.makedirs('results/vit', exist_ok=True)

    # Step 1 — load data from full dataset
    train_loader, val_loader, test_loader = get_dataloaders(
        data_dir='data_balanced_200', batch_size=32
    )

    # Step 2 — compute class weights for imbalanced data
    # chromosome_X and Y have fewer images so they get higher weight
    class_weights = get_class_weights('data_balanced_200')

    # Step 3 — build model
    model = build_vit(num_classes=24, pretrained=True)

    # Step 4 — train with class weights
    history = train_model(
        model, train_loader, val_loader,
        lr=1e-4, epochs=50, model_name="vit",
        class_weights=class_weights
    )

    print('Training complete!')
    print(f'Best F1: {max(history["val_f1"]):.4f}')
