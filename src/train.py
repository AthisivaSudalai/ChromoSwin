import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from sklearn.metrics import f1_score
import numpy as np

def train_model(model, train_loader, val_loader,
                lr=1e-4, epochs=50, model_name="vit", class_weights=None):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on: {device}")
    model = model.to(device)

    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=10)
    # use class weights if provided — handles imbalanced dataset
    if class_weights is not None:
        criterion = nn.CrossEntropyLoss(
            weight=class_weights.to(device)
    )
    else:
        criterion = nn.CrossEntropyLoss()

    # mixed precision for speed
    scaler = torch.amp.GradScaler('cuda')

    best_f1   = 0.0
    patience  = 15
    no_improve = 0
    history   = {'train_loss': [], 'val_f1': []}

    # ── Stage 1: freeze backbone for first 5 epochs ──
    from src.vit_model import freeze_backbone, unfreeze_backbone
    freeze_backbone(model)

    for epoch in range(epochs):

        # unfreeze at epoch 5
        if epoch == 5:
            unfreeze_backbone(model)

        # ── training ──
        model.train()
        total_loss = 0
        for batch in train_loader:
            images, labels = batch
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            with torch.amp.autocast('cuda'):  # FP16 mixed precision
                outputs = model(pixel_values=images).logits
                loss = criterion(outputs, labels)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            total_loss += loss.item()

        scheduler.step()

        # ── validation ──
        model.eval()
        all_preds, all_labels = [], []
        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(device)
                preds = model(pixel_values=images).logits.argmax(dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.numpy())

        val_f1 = f1_score(all_labels, all_preds, average='macro')
        avg_loss = total_loss / len(train_loader)
        history['train_loss'].append(avg_loss)
        history['val_f1'].append(val_f1)

        print(f"Epoch {epoch+1}/{epochs} | Loss: {avg_loss:.4f} | Val F1: {val_f1:.4f}")

        # save best model
        if val_f1 > best_f1:
            best_f1 = val_f1
            torch.save(model.state_dict(), f'results/{model_name}/best_model.pth')
            no_improve = 0
            print(f"  --> New best saved! F1={best_f1:.4f}")
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break

    print(f"\nTraining done. Best Val F1: {best_f1:.4f}")
    return history