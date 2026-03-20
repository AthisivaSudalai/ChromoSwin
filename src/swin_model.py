# src/swin_model.py
import torch
import torch.nn as nn
from transformers import SwinForImageClassification

def build_swin(num_classes=24, pretrained=True):

    if pretrained:
        model = SwinForImageClassification.from_pretrained(
            'microsoft/swin-tiny-patch4-window7-224',
            num_labels=num_classes,
            ignore_mismatched_sizes=True
        )
        print("Loaded pretrained Swin-Tiny weights")
    else:
        # load architecture only — no pretrained weights
        # used when loading our own saved weights
        model = SwinForImageClassification.from_pretrained(
            'microsoft/swin-tiny-patch4-window7-224',
            num_labels=num_classes,
            ignore_mismatched_sizes=True
        )
        print("Built Swin-T architecture (weights loaded separately)")

    model.classifier = nn.Sequential(
        nn.Dropout(p=0.3),
        nn.Linear(model.config.hidden_size, num_classes)
    )

    return model


def freeze_backbone(model):
    for name, param in model.named_parameters():
        if 'classifier' not in name:
            param.requires_grad = False
    print("Backbone frozen — only training classifier head")


def unfreeze_backbone(model):
    for param in model.parameters():
        param.requires_grad = True
    print("Full model unfrozen — fine-tuning all layers")


if __name__ == "__main__":
    model = build_swin(num_classes=24, pretrained=True)
    dummy = torch.randn(2, 3, 224, 224)
    out   = model(pixel_values=dummy).logits
    print(f"Output shape: {out.shape}")  # should be [2, 24]