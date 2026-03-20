# augment_xy.py
# Creates extra chromosome_X and chromosome_Y images
# using transformations — each real image generates 20 new ones
# bringing X and Y from 50 images to ~1000 images each

import os
import random
from PIL import Image, ImageFilter, ImageEnhance
import numpy as np

random.seed(42)

# ── SETTINGS ──────────────────────────────────────────────────
SOURCE_CLASSES = ['chromosome_X', 'chromosome_Y']
SOURCE_DIR     = 'data/train'
TARGET_COUNT   = 1000   # how many images we want per class after augmentation
# ──────────────────────────────────────────────────────────────


def augment_image(img):
    """
    Applies a random combination of transformations to one image.
    Every transformation keeps the chromosome visually valid.
    """
    # 1 — Random rotation (0 to 360 degrees)
    # chromosomes can appear at any angle in a karyogram
    angle = random.uniform(0, 360)
    img   = img.rotate(angle, expand=True, fillcolor=(255, 255, 255))

    # 2 — Random horizontal flip (50% chance)
    if random.random() > 0.5:
        img = img.transpose(Image.FLIP_LEFT_RIGHT)

    # 3 — Random vertical flip (50% chance)
    if random.random() > 0.5:
        img = img.transpose(Image.FLIP_TOP_BOTTOM)

    # 4 — Random brightness adjustment (±20%)
    factor = random.uniform(0.8, 1.2)
    img    = ImageEnhance.Brightness(img).enhance(factor)

    # 5 — Random contrast adjustment (±20%)
    factor = random.uniform(0.8, 1.2)
    img    = ImageEnhance.Contrast(img).enhance(factor)

    # 6 — Random slight zoom (crop center 80-100% of image)
    zoom = random.uniform(0.80, 1.00)
    w, h = img.size
    new_w, new_h = int(w * zoom), int(h * zoom)
    left   = (w - new_w) // 2
    top    = (h - new_h) // 2
    img    = img.crop((left, top, left + new_w, top + new_h))

    # 7 — Slight Gaussian blur (30% chance) — simulates focus variation
    if random.random() > 0.7:
        img = img.filter(ImageFilter.GaussianBlur(radius=0.5))

    # always resize back to 224x224 for model compatibility
    img = img.resize((224, 224), Image.BICUBIC)

    return img


def augment_class(cls_name, source_dir, target_count):

    source_path = os.path.join(source_dir, cls_name)

    if not os.path.exists(source_path):
        print(f"  ERROR: {source_path} not found")
        return

    # get all existing images
    existing = [
        f for f in os.listdir(source_path)
        if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tif'))
    ]

    current_count = len(existing)
    needed        = target_count - current_count

    if needed <= 0:
        print(f"  {cls_name}: already has {current_count} images — no augmentation needed")
        return

    print(f"  {cls_name}: {current_count} real images → generating {needed} augmented images...")

    generated = 0
    while generated < needed:
        # pick a random source image to augment
        source_img_name = random.choice(existing)
        source_img_path = os.path.join(source_path, source_img_name)

        try:
            img = Image.open(source_img_path).convert('RGB')
            aug = augment_image(img)

            # save with a clear augmented filename
            save_name = f"aug_{cls_name}_{generated:05d}.jpg"
            aug.save(os.path.join(source_path, save_name), quality=90)
            generated += 1

            if generated % 100 == 0:
                print(f"    Generated {generated}/{needed}...")

        except Exception as e:
            print(f"    Warning: could not process {source_img_name}: {e}")
            continue

    final_count = len([
        f for f in os.listdir(source_path)
        if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tif'))
    ])

    print(f"  {cls_name}: done! {current_count} → {final_count} images")


if __name__ == '__main__':

    print("="*55)
    print("Augmenting chromosome_X and chromosome_Y")
    print(f"Target: {TARGET_COUNT} images per class")
    print("="*55 + "\n")

    for cls in SOURCE_CLASSES:
        augment_class(cls, SOURCE_DIR, TARGET_COUNT)

    print("\n" + "="*55)
    print("DONE! Now run:")
    print("  1. Remove-Item -Recurse -Force data_full")
    print("  2. python balanced_dt.py")
    print("  3. py main_vit.py")
    print("="*55)