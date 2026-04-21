# test_hmfo.py
import torch
import os
import json
import matplotlib.pyplot as plt
from sklearn.metrics import (
    f1_score, accuracy_score,
    precision_score, recall_score,
    confusion_matrix, classification_report
)
from src.dataset    import get_dataloaders
from src.swin_model import build_swin

if __name__ == '__main__':

    MODEL_PATH  = 'results/swin_hmfo/best_model.pth'
    DATA_DIR    = 'data_balanced'
    NUM_CLASSES = 24
    BATCH_SIZE  = 32

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Testing on: {device}")

    _, _, test_loader = get_dataloaders(DATA_DIR, BATCH_SIZE)
    class_names = test_loader.dataset.classes

    model = build_swin(num_classes=NUM_CLASSES, pretrained=False)
    model.load_state_dict(torch.load(
        MODEL_PATH, map_location=device, weights_only=True
    ))
    model = model.to(device)
    model.eval()
    print(f"Model loaded from: {MODEL_PATH}")

    all_preds, all_labels = [], []
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

    print("\n" + "="*55)
    print("  MODEL 3 — SWIN-T + HMFO — TEST RESULTS")
    print("="*55)
    print(f"  Accuracy  : {accuracy:.4f}  ({accuracy*100:.2f}%)")
    print(f"  F1 Score  : {f1:.4f}")
    print(f"  Precision : {precision:.4f}")
    print(f"  Recall    : {recall:.4f}")
    print("="*55)

    print("\nPer-class breakdown:")
    print(classification_report(
        all_labels, all_preds,
        target_names=class_names,
        zero_division=0
    ))

    # load best HMFO params found
    params_path = 'results/swin_hmfo/best_params.json'
    if os.path.exists(params_path):
        with open(params_path) as f:
            best_params = json.load(f)
        print("\nBest hyperparameters found by HMFO:")
        for k, v in best_params.items():
            print(f"  {k}: {v:.6f}")

    # confusion matrix
    os.makedirs('results/swin_hmfo', exist_ok=True)
    cm      = confusion_matrix(all_labels, all_preds)
    cm_norm = cm.astype('float') / cm.sum(axis=1, keepdims=True)

    plt.figure(figsize=(14, 12))
    plt.imshow(cm_norm, interpolation='nearest', cmap='Purples')
    plt.colorbar()
    plt.title('Swin-T + HMFO Confusion Matrix (normalised)', fontsize=14)
    plt.xlabel('Predicted Class', fontsize=12)
    plt.ylabel('True Class', fontsize=12)
    short_names = [c.replace('chromosome_', 'chr') for c in class_names]
    plt.xticks(range(NUM_CLASSES), short_names, rotation=90, fontsize=9)
    plt.yticks(range(NUM_CLASSES), short_names, fontsize=9)
    plt.tight_layout()
    plt.savefig('results/swin_hmfo/confusion_matrix.png', dpi=150)
    plt.show()
    print("Confusion matrix saved to results/swin_hmfo/confusion_matrix.png")

    print("\n" + "="*55)
    print("  FINAL PAPER TABLE — ALL 3 MODELS")
    print("="*55)
    print(f"  Basic ViT      : Acc=68.75%  F1=0.6718")
    print(f"  Swin-T no HMFO : Acc=65.28%  F1=0.6380")
    print(f"  Swin-T + HMFO  : Acc={accuracy*100:.2f}%  F1={f1:.4f}  ← your model")
    print("="*55)
    print("\nIMPORTANT: These are small-dataset results (40 images/class)")
    print("Run on full dataset (Colab) for final paper numbers")
