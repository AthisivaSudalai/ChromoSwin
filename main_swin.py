# main_swin.py

from src.dataset    import get_dataloaders, get_class_weights
from src.swin_model import build_swin
from src.train      import train_model
import os

if __name__ == '__main__':

    os.makedirs('results/swin', exist_ok=True)

    # same dataset as ViT — fair comparison
    train_loader, val_loader, test_loader = get_dataloaders(
        data_dir='data_balanced_200', batch_size=32
    )

    # same class weights — identical conditions for fair comparison
    class_weights = get_class_weights('data_balanced_200')

    model = build_swin(num_classes=24, pretrained=True)

    history = train_model(
        model, train_loader, val_loader,
        lr=1e-4, epochs=50, model_name="swin",
        class_weights=class_weights
    )

    print('Training complete!')
    print(f'Best F1: {max(history["val_f1"]):.4f}')
