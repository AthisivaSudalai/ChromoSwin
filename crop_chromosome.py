# crop_chromosomes.py
# ─────────────────────────────────────────────────────────────
# This script reads the downloaded AutoKary2022 dataset,
# cuts out each individual chromosome from the full images,
# and saves them into folders PyTorch can read.
# ─────────────────────────────────────────────────────────────

import os
import json
from PIL import Image

# ── ONLY EDIT THIS LINE if your folder name is different ──────
ROBOFLOW_DIR = "AutoKary2022-1"
# ─────────────────────────────────────────────────────────────


def get_class_name(category_name):
    name = str(category_name).strip().replace(" ", "_").lower()

    # fix typo in AutoKary2022 — 'chromosomes24' should be 'chromosome_24'
    if name == "chromosomes24":
        return "chromosome_24"

    if not name.startswith("chromosome"):
        name = "chromosome_" + name

    return name


def crop_one_split(coco_json_path, images_folder, output_folder):
    print(f"\nProcessing: {coco_json_path}")

    # check the json file actually exists
    if not os.path.exists(coco_json_path):
        print(f"  WARNING: Could not find {coco_json_path} — skipping")
        return 0

    with open(coco_json_path, 'r') as f:
        coco = json.load(f)

    # build lookup tables from the JSON
    id_to_image    = {img['id']: img    for img in coco['images']}
    id_to_category = {cat['id']: cat['name'] for cat in coco['categories']}

    print(f"  Images found      : {len(coco['images'])}")
    print(f"  Annotations found : {len(coco['annotations'])}")
    print(f"  Classes           : {sorted(id_to_category.values())}")

    saved   = 0
    skipped = 0

    for ann in coco['annotations']:

        # get the image this annotation belongs to
        image_info  = id_to_image[ann['image_id']]
        class_name  = get_class_name(id_to_category[ann['category_id']])
        image_path  = os.path.join(images_folder, image_info['file_name'])

        # skip if image file is missing
        if not os.path.exists(image_path):
            skipped += 1
            continue

        # COCO bounding box format: [x_top_left, y_top_left, width, height]
        x, y, w, h = ann['bbox']
        x, y, w, h = int(x), int(y), int(w), int(h)

        # skip if the crop is too tiny — likely a bad annotation
        if w < 10 or h < 10:
            skipped += 1
            continue

        # open image and crop out the chromosome
        try:
            img       = Image.open(image_path).convert('RGB')
            img_w, img_h = img.size

            # add a small 4px border around the chromosome
            pad = 4
            x1 = max(0, x - pad)
            y1 = max(0, y - pad)
            x2 = min(img_w, x + w + pad)
            y2 = min(img_h, y + h + pad)

            crop = img.crop((x1, y1, x2, y2))

            # resize to 224x224 — required by ViT and Swin-T
            crop = crop.resize((224, 224), Image.BICUBIC)

        except Exception as e:
            print(f"  Could not process {image_path}: {e}")
            skipped += 1
            continue

        # create output folder for this class if it doesn't exist
        dest_folder = os.path.join(output_folder, class_name)
        os.makedirs(dest_folder, exist_ok=True)

        # save the cropped chromosome image
        filename = f"{class_name}_{ann['id']:06d}.jpg"
        crop.save(os.path.join(dest_folder, filename), quality=95)
        saved += 1

    print(f"  Saved : {saved}")
    print(f"  Skipped : {skipped}")
    return saved


# ── MAIN ──────────────────────────────────────────────────────
if __name__ == "__main__":

    total = 0

    # AutoKary2022 has 3 splits: train, valid, test
    # We rename "valid" → "val" to match our project structure
    splits = [
        ("train", "train"),
        ("valid", "val"),
        ("test",  "test"),
    ]

    for roboflow_split, our_split in splits:
        json_path     = os.path.join(ROBOFLOW_DIR, roboflow_split,
                                     "_annotations.coco.json")
        images_folder = os.path.join(ROBOFLOW_DIR, roboflow_split)
        output_folder = os.path.join("data", our_split)

        total += crop_one_split(json_path, images_folder, output_folder)

    print("\n" + "="*50)
    print(f"DONE! Total chromosomes saved: {total}")
    print("="*50)
    print("\nYour data/ folder now looks like:")
    print("  data/train/chromosome_1/  ← training images")
    print("  data/train/chromosome_2/")
    print("  ...up to chromosome_Y/")
    print("\nNext step — run your training:")
    print("  py main_vit.py")