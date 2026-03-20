import torch
import torch.nn as nn
from transformers import ViTForImageClassification, ViTConfig

def build_vit(num_classes=24, pretrained=True):

    if pretrained:
        # load pretrained ViT-Base/16 from Hugging Face
        # pretrained on ImageNet-21k — gives it a head start
        model = ViTForImageClassification.from_pretrained(
            'google/vit-base-patch16-224-in21k',
            num_labels=num_classes,
            ignore_mismatched_sizes=True  # needed when changing num_classes
        )
        print("Loaded pretrained ViT-Base/16 weights")

    else:
        # build from scratch — useful to show WHY pretrained matters
        config = ViTConfig(
            image_size=224,
            patch_size=16,
            num_channels=3,
            num_labels=num_classes,
            hidden_size=768,
            num_hidden_layers=12,
            num_attention_heads=12,
            intermediate_size=3072,
        )
        model = ViTForImageClassification(config)
        print("Built ViT from scratch — no pretrained weights")

    # replace classifier head with dropout + linear
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