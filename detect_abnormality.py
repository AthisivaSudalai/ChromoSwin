# detect_abnormality.py
# Stage 2 — Takes classified chromosomes from one patient
# and determines if the karyotype is normal or abnormal

import torch
import os
from PIL import Image
from torchvision import transforms
from src.swin_model import build_swin

# ── known abnormalities and their rules ───────────────────────
ABNORMALITY_RULES = {
    'Trisomy 21 (Down Syndrome)':    lambda c: c.get('chromosome_21', 0) == 3,
    'Trisomy 18 (Edwards Syndrome)': lambda c: c.get('chromosome_18', 0) == 3,
    'Trisomy 13 (Patau Syndrome)':   lambda c: c.get('chromosome_13', 0) == 3,
    'Turner Syndrome (45,X)':        lambda c: c.get('chromosome_X', 0) == 1
                                               and c.get('chromosome_Y', 0) == 0,
    'Klinefelter Syndrome (47,XXY)': lambda c: c.get('chromosome_X', 0) == 2
                                               and c.get('chromosome_Y', 0) == 1,
    'Triple X Syndrome (47,XXX)':    lambda c: c.get('chromosome_X', 0) == 3,
    'XYY Syndrome (47,XYY)':         lambda c: c.get('chromosome_X', 0) == 1
                                               and c.get('chromosome_Y', 0) == 2,
}

# chromosome class names in the same order as your model output
CLASS_NAMES = [
    'chromosome_1',  'chromosome_10', 'chromosome_11',
    'chromosome_12', 'chromosome_13', 'chromosome_14',
    'chromosome_15', 'chromosome_16', 'chromosome_17',
    'chromosome_18', 'chromosome_19', 'chromosome_2',
    'chromosome_20', 'chromosome_21', 'chromosome_22',
    'chromosome_3',  'chromosome_4',  'chromosome_5',
    'chromosome_6',  'chromosome_7',  'chromosome_8',
    'chromosome_9',  'chromosome_X',  'chromosome_Y'
]


def load_model(model_path, num_classes=24):
    """Loads your best trained Swin-T+HMFO model"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model  = build_swin(num_classes=num_classes, pretrained=False)
    model.load_state_dict(torch.load(
        model_path, map_location=device, weights_only=True
    ))
    model = model.to(device)
    model.eval()
    return model, device


def classify_chromosome(model, device, image_path):
    """
    Takes one chromosome image
    Returns the predicted class name and confidence score
    """
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])

    img    = Image.open(image_path).convert('RGB')
    tensor = transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs     = model(pixel_values=tensor).logits
        probs       = torch.softmax(outputs, dim=1)
        pred_idx    = probs.argmax(dim=1).item()
        confidence  = probs[0][pred_idx].item()

    predicted_class = CLASS_NAMES[pred_idx]
    return predicted_class, confidence


def analyze_karyotype(chromosome_folder, model, device):
    """
    Takes a folder of 46 cropped chromosome images from ONE patient
    Classifies each one and detects abnormalities
    """
    print(f"\nAnalyzing karyotype from: {chromosome_folder}")
    print("-" * 50)

    # classify every chromosome image in the folder
    counts     = {}
    all_preds  = []

    image_files = [
        f for f in os.listdir(chromosome_folder)
        if f.lower().endswith(('.jpg', '.jpeg', '.png'))
    ]

    for img_file in sorted(image_files):
        img_path    = os.path.join(chromosome_folder, img_file)
        pred, conf  = classify_chromosome(model, device, img_path)

        all_preds.append((img_file, pred, conf))
        counts[pred] = counts.get(pred, 0) + 1

        print(f"  {img_file:30s} → {pred:20s} ({conf*100:.1f}%)")

    # Stage 2 — apply abnormality rules
    print("\n" + "="*50)
    print("KARYOTYPE ANALYSIS REPORT")
    print("="*50)
    print(f"Total chromosomes classified: {len(image_files)}")
    print(f"\nChromosome counts:")

    for cls in CLASS_NAMES:
        count  = counts.get(cls, 0)
        normal = 2 if 'X' not in cls and 'Y' not in cls else None
        marker = ""
        if normal and count != normal:
            marker = f"  ← ABNORMAL (expected {normal}, found {count})"
        print(f"  {cls:20s}: {count}{marker}")

    # check each abnormality rule
    print(f"\nAbnormality screening:")
    found_abnormality = False

    for disease, rule in ABNORMALITY_RULES.items():
        if rule(counts):
            print(f"  DETECTED: {disease}")
            found_abnormality = True

    if not found_abnormality:
        # check for any autosomal trisomies not in the rules
        for cls in CLASS_NAMES:
            if 'X' not in cls and 'Y' not in cls:
                if counts.get(cls, 0) != 2:
                    print(f"  DETECTED: Abnormal count for {cls} "
                          f"(found {counts.get(cls,0)}, expected 2)")
                    found_abnormality = True

    print("\n" + "="*50)
    if found_abnormality:
        print("  RESULT: ABNORMAL KARYOTYPE DETECTED")
        print("  Action: Refer to clinical cytogeneticist")
    else:
        print("  RESULT: NORMAL KARYOTYPE")
        print("  Action: No chromosomal abnormality detected")
    print("="*50)

    return counts, found_abnormality


if __name__ == '__main__':

    # load your best model — Swin-T + HMFO
    MODEL_PATH = 'results/swin_hmfo/best_model.pth'
    model, device = load_model(MODEL_PATH)
    print(f"Model loaded — running on {device}")

    # point this to a folder containing chromosome images from one patient
    # for testing, use any folder from your test set
    TEST_FOLDER = 'data_full/test/chromosome_21'

    counts, is_abnormal = analyze_karyotype(TEST_FOLDER, model, device)