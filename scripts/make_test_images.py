"""Generate test images so nobody needs to download anything.
   images/sizes/img_<S>.png   -> one image each of 256, 512, 1024, 2048, 4096 px (square)
   images/batch/img_XXX.png   -> 100 images of 512x512 (use --batch N to pick the batch size)
Run from the project root:  python scripts/make_test_images.py
"""
import os
import numpy as np
from PIL import Image


def make(size, seed):
    rng = np.random.default_rng(seed)
    y, x = np.mgrid[0:size, 0:size].astype(np.float32)
    f = size / 8.0
    img = np.stack([
        127 + 127 * np.sin(x / f + seed),
        127 + 127 * np.sin(y / f + 2 * seed),
        127 + 127 * np.sin((x + y) / (2 * f)),
    ], axis=-1).astype(np.uint8)
    for _ in range(25):                                  # random rectangles -> sharp edges for Sobel
        x0, y0 = rng.integers(0, size - 2, 2)
        x1 = min(size, x0 + rng.integers(2, max(3, size // 4)))
        y1 = min(size, y0 + rng.integers(2, max(3, size // 4)))
        img[y0:y1, x0:x1] = rng.integers(0, 256, 3)
    return img


os.makedirs("images/sizes", exist_ok=True)
os.makedirs("images/batch", exist_ok=True)
for s in [256, 512, 1024, 2048, 4096]:
    Image.fromarray(make(s, s)).save(f"images/sizes/img_{s}.png")
    print("saved size", s)
for i in range(100):
    Image.fromarray(make(512, i)).save(f"images/batch/img_{i:03d}.png")
print("saved 100 batch images")
