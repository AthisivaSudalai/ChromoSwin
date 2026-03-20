# test_swin.py
import torch
import os
import matplotlib.pyplot as plt
from sklearn.metrics import (
    f1_score, accuracy_score,
    precision_score, recall_score,
    confusion_matrix, classification_report
)
from src.dataset    import get_dataloaders
from src.swin_model import build_swin

if __name__ == '__main__':

    MODEL_PATH  = 'results/swin/best_model.pth'
    DATA_DIR    = 'data_balanced'
    NUM_CLASSES = 24
    BATCH_SIZE  = 32

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Testing on: {device}")

    _, _, test_loader = get_dataloaders(DATA_DIR, BATCH_SIZE)
    class_names = test_loader.dataset.classes
    print(f"Test images : {len(test_loader.dataset)}")

    model = build_swin(num_classes=NUM_CLASSES, pretrained=False)
    model.load_state_dict(torch.load(
        MODEL_PATH, map_location=device, weights_only=True
    ))
    model = model.to(device)
    model.eval()
    print(f"Model loaded from: {MODEL_PATH}")

    all_preds  = []
    all_labels = []

    with torch.no_grad():
        for images, labels in test_loader:
            images  = images.to(device)
            outputs = model(pixel_values=images).logits
            preds   = outputs.argmax(dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())

    accuracy  = accuracy_score(all_labels, all_preds)
    f1        = f1_score(all_labels, all_preds, average='macro')
    precision = precision_score(all_labels, all_preds,
                                average='macro', zero_division=0)
    recall    = recall_score(all_labels, all_preds,
                             average='macro', zero_division=0)

    print("\n" + "="*50)
    print("  MODEL 2 — SWIN-T (no HMFO) — TEST RESULTS")
    print("="*50)
    print(f"  Accuracy  : {accuracy:.4f}  ({accuracy*100:.2f}%)")
    print(f"  F1 Score  : {f1:.4f}")
    print(f"  Precision : {precision:.4f}")
    print(f"  Recall    : {recall:.4f}")
    print("="*50)

    print("\nPer-class breakdown:")
    print(classification_report(
        all_labels, all_preds,
        target_names=class_names,
        zero_division=0
    ))

    # confusion matrix
    os.makedirs('results/swin', exist_ok=True)
    cm      = confusion_matrix(all_labels, all_preds)
    cm_norm = cm.astype('float') / cm.sum(axis=1, keepdims=True)

    plt.figure(figsize=(14, 12))
    plt.imshow(cm_norm, interpolation='nearest', cmap='Greens')
    plt.colorbar()
    plt.title('Swin-T Confusion Matrix (normalised)', fontsize=14)
    plt.xlabel('Predicted Class', fontsize=12)
    plt.ylabel('True Class', fontsize=12)
    short_names = [c.replace('chromosome_', 'chr') for c in class_names]
    plt.xticks(range(NUM_CLASSES), short_names, rotation=90, fontsize=9)
    plt.yticks(range(NUM_CLASSES), short_names, fontsize=9)
    plt.tight_layout()
    plt.savefig('results/swin/confusion_matrix.png', dpi=150)
    plt.show()
    print("Confusion matrix saved to results/swin/confusion_matrix.png")

    print("\n" + "="*50)
    print("  PAPER TABLE — COPY THESE NUMBERS")
    print("="*50)
    print(f"  Model     : Swin-T (no HMFO)")
    print(f"  Accuracy  : {accuracy*100:.2f}%")
    print(f"  F1 Score  : {f1:.4f}")
    print(f"  Precision : {precision:.4f}")
    print(f"  Recall    : {recall:.4f}")
    print("="*50)
    print("\nNext: build hmfo.py for Model 3")