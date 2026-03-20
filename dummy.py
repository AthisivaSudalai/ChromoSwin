# create_dummy_data.py
# run this ONCE to create fake images so you can test your code
# delete this later when you have real data

import os
from PIL import Image
import random

# 24 chromosome classes
classes = [f'chromosome_{i}' for i in range(1, 23)] + ['chromosome_X', 'chromosome_Y']

splits = {'train': 50, 'val': 10, 'test': 10}  # images per class per split

for split, count in splits.items():
    for cls in classes:
        folder = f'data/{split}/{cls}'
        os.makedirs(folder, exist_ok=True)

        for i in range(count):
            # create a random 224x224 image
            img = Image.fromarray(
                __import__('numpy').random.randint(0, 255, (224, 224, 3),
                dtype='uint8')
            )
            img.save(f'{folder}/img_{i:03d}.jpg')

print("Dummy dataset created!")
print("data/train  — 24 classes × 50 images = 1200 images")
print("data/val    — 24 classes × 10 images = 240 images")
print("data/test   — 24 classes × 10 images = 240 images")