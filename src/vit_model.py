import torch
import torch.nn as nn
from transformers import ViTForImageClassification, ViTConfig

def build_vit(num_classes=24, pretrained=True):

    if pretrained:
        model = ViTForImageClassification.from_pretrained(
            'google/vit-base-patch16-224-in21k',
            num_labels=num_classes,
            ignore_mismatched_sizes=True
        )
        print("Loaded pretrained ViT-Base/16 weights")
    else:
        model = ViTForImageClassification.from_pretrained(
            'google/vit-base-patch16-224-in21k',
            num_labels=num_classes,
            ignore_mismatched_sizes=True
        )
        print("Built ViT architecture (weights loaded separately)")

    model.classifier = nn.Sequential(
        nn.Dropout(p=0.3),
        nn.Linear(model.config.hidden_size, num_classes)
    )

    return model


# ── two-stage fine tuning ──────────────────────────────
def freeze_backbone(model):
    """Stage 1: freeze everything except the head"""
    for name, param in model.named_parameters():
        if 'classifier' not in name:
            param.requires_grad = False
    print("Backbone frozen — only training classifier head")

def unfreeze_backbone(model):
    """Stage 2: unfreeze all layers for full fine-tuning"""
    for param in model.parameters():
        param.requires_grad = True
    print("Full model unfrozen — fine-tuning all layers")


if __name__ == "__main__":
    model = build_vit(num_classes=24, pretrained=True)

    # quick sanity check
    dummy = torch.randn(2, 3, 224, 224)  # batch of 2 images
    out = model(pixel_values=dummy).logits
    print(f"Output shape: {out.shape}")  # should be [2, 24]