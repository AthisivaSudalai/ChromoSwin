# balanced_dt.py
import os
import shutil
import random

random.seed(42)

# ── SETTINGS ──────────────────────────────────────────────────
SOURCE_DIR = 'data/train'
OUTPUT_DIR = 'data_balanced_200'

# exact fixed counts per class
N_TRAIN = 200
N_VAL   = 40
N_TEST  = 20
N_TOTAL = N_TRAIN + N_VAL + N_TEST  # 260 images needed per class

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
# ──────────────────────────────────────────────────────────────


def copy_balanced(source_dir, output_dir):

    print(f"Creating balanced dataset — {N_TOTAL} images per class")
    print(f"Split: {N_TRAIN} train | {N_VAL} val | {N_TEST} test per class\n")

    skipped = []

    for cls in VALID_CLASSES:
        cls_path = os.path.join(source_dir, cls)

        # check folder exists
        if not os.path.exists(cls_path):
            print(f"  SKIP: {cls} — folder not found")
            skipped.append(cls)
            continue

        # get all images
        all_images = [
            f for f in os.listdir(cls_path)
            if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tif'))
        ]

        if len(all_images) == 0:
            print(f"  SKIP: {cls} — folder is empty")
            skipped.append(cls)
            continue

        # check enough images exist for the required total
        if len(all_images) < N_TOTAL:
            print(f"  WARNING: {cls} only has {len(all_images)} images "
                  f"(need {N_TOTAL}) — using all available")
            selected = all_images.copy()
            # recalculate splits proportionally from what's available
            n = len(selected)
            actual_train = int(n * 0.70)
            actual_val   = int(n * 0.20)
            actual_test  = n - actual_train - actual_val
        else:
            # randomly pick exactly N_TOTAL images
            selected     = random.sample(all_images, N_TOTAL)
            actual_train = N_TRAIN
            actual_val   = N_VAL
            actual_test  = N_TEST

        random.shuffle(selected)

        buckets = {
            'train': selected[:actual_train],
            'val':   selected[actual_train:actual_train + actual_val],
            'test':  selected[actual_train + actual_val:actual_train + actual_val + actual_test]
        }

        # copy to output folders
        for split, images in buckets.items():
            dest = os.path.join(output_dir, split, cls)
            os.makedirs(dest, exist_ok=True)
            for img in images:
                shutil.copy(
                    os.path.join(cls_path, img),
                    os.path.join(dest, img)
                )

        print(f"  {cls:20s} → "
              f"{len(buckets['train'])} train | "
              f"{len(buckets['val'])} val | "
              f"{len(buckets['test'])} test")

    # summary
    processed = len(VALID_CLASSES) - len(skipped)
    total_train = processed * N_TRAIN
    total_val   = processed * N_VAL
    total_test  = processed * N_TEST

    print(f"\n{'='*55}")
    print(f"DONE!")
    print(f"Classes processed : {processed} / {len(VALID_CLASSES)}")
    print(f"Total train images: {total_train}")
    print(f"Total val images  : {total_val}")
    print(f"Total test images : {total_test}")
    print(f"Grand total       : {total_train + total_val + total_test}")
    if skipped:
        print(f"Skipped           : {skipped}")
    print(f"Saved to          : {output_dir}/")
    print(f"{'='*55}")
    print(f"\nNext steps:")
    print(f"  1. Update data_dir in main files to '{output_dir}'")
    print(f"  2. Run: py main_vit.py")


if __name__ == '__main__':
    copy_balanced(SOURCE_DIR, OUTPUT_DIR)