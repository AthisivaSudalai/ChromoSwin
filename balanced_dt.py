# balanced_dt.py
import os
import shutil
import random

random.seed(42)

# ── SETTINGS ──────────────────────────────────────────────────
SOURCE_DIR = 'data/train'
OUTPUT_DIR = 'data_full'

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

    print(f"Creating full dataset — using ALL available images per class")
    print(f"Split: 70% train | 15% val | 15% test\n")

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

        # use ALL images — no limit
        selected = all_images.copy()
        random.shuffle(selected)

        # calculate splits from actual count
        n_total = len(selected)
        n_train = int(n_total * 0.70)
        n_val   = int(n_total * 0.15)
        n_test  = n_total - n_train - n_val  # remainder goes to test

        buckets = {
            'train': selected[:n_train],
            'val':   selected[n_train:n_train + n_val],
            'test':  selected[n_train + n_val:]
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

        print(f"  {cls:20s} — total: {n_total:4d} "
              f"→ {len(buckets['train'])} train | "
              f"{len(buckets['val'])} val | "
              f"{len(buckets['test'])} test")

    # summary
    processed = len(VALID_CLASSES) - len(skipped)
    print(f"\n{'='*55}")
    print(f"DONE!")
    print(f"Classes processed : {processed} / {len(VALID_CLASSES)}")
    if skipped:
        print(f"Skipped           : {skipped}")
    print(f"Saved to          : {output_dir}/")
    print(f"{'='*55}")
    print(f"\nNext — run: py main_vit.py")


if __name__ == '__main__':
    copy_balanced(SOURCE_DIR, OUTPUT_DIR)