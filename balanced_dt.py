# balanced_dt.py
import os
import shutil
import random

random.seed(42)

IMAGES_PER_CLASS = 40
SOURCE_DIR       = 'data/train'
OUTPUT_DIR       = 'data_balanced'

VALID_CLASSES = [
    'chromosome_1',  'chromosome_2',  'chromosome_3',
    'chromosome_4',  'chromosome_5',  'chromosome_6',
    'chromosome_7',  'chromosome_8',  'chromosome_9',
    'chromosome_10', 'chromosome_11', 'chromosome_12',
    'chromosome_13', 'chromosome_14', 'chromosome_15',
    'chromosome_16', 'chromosome_17', 'chromosome_18',
    'chromosome_19', 'chromosome_20', 'chromosome_21',
    'chromosome_22', 'chromosome_X',  'chromosome_Y'
]

def copy_balanced(source_dir, output_dir, n_per_class):

    n_train = int(n_per_class * 0.70)
    n_val   = int(n_per_class * 0.15)
    n_test  = n_per_class - n_train - n_val

    print(f"Creating balanced dataset — {n_per_class} images per class")
    print(f"Split: {n_train} train | {n_val} val | {n_test} test\n")

    skipped = []

    for cls in VALID_CLASSES:
        cls_path = os.path.join(source_dir, cls)

        # ── check folder exists ──────────────────────────────
        if not os.path.exists(cls_path):
            print(f"  SKIP: {cls} — folder not found")
            skipped.append(cls)
            continue

        # ── get all images in this class ─────────────────────
        all_images = [
            f for f in os.listdir(cls_path)
            if f.lower().endswith(('.jpg','.jpeg','.png','.bmp','.tif'))
        ]

        # ── check at least 10 images exist ───────────────────
        if len(all_images) == 0:
            print(f"  SKIP: {cls} — folder is empty")
            skipped.append(cls)
            continue

        if len(all_images) < n_per_class:
            print(f"  WARNING: {cls} only has {len(all_images)} images "
                  f"(wanted {n_per_class}) — using all {len(all_images)}")
            selected = all_images.copy()
        else:
            selected = random.sample(all_images, n_per_class)

        # ── shuffle and split ────────────────────────────────
        random.shuffle(selected)

        buckets = {
            'train': selected[:n_train],
            'val':   selected[n_train:n_train + n_val],
            'test':  selected[n_train + n_val:]
        }

        # ── copy to output folders ───────────────────────────
        for split, images in buckets.items():
            dest = os.path.join(output_dir, split, cls)
            os.makedirs(dest, exist_ok=True)
            for img in images:
                shutil.copy(
                    os.path.join(cls_path, img),
                    os.path.join(dest, img)
                )

        print(f"  {cls:20s} → {len(buckets['train'])} train | "
              f"{len(buckets['val'])} val | "
              f"{len(buckets['test'])} test")

    # ── summary ──────────────────────────────────────────────
    processed = len(VALID_CLASSES) - len(skipped)
    print(f"\n{'='*50}")
    print(f"DONE!")
    print(f"Classes processed : {processed} / {len(VALID_CLASSES)}")
    print(f"Images per class  : {n_per_class}")
    print(f"Total images      : ~{processed * n_per_class}")
    print(f"Saved to          : {output_dir}/")
    if skipped:
        print(f"Skipped           : {skipped}")
    print(f"{'='*50}")
    print(f"\nNext — run: py main_vit.py")


if __name__ == '__main__':
    copy_balanced(SOURCE_DIR, OUTPUT_DIR, IMAGES_PER_CLASS)