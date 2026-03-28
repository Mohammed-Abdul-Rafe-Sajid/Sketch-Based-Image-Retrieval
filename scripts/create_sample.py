import os
import shutil
from torchvision.datasets import CIFAR10
import numpy as np
import cv2

# Create folders
os.makedirs("data/sample_edges", exist_ok=True)
os.makedirs("data/sample_original", exist_ok=True)

dataset = CIFAR10(root="data/raw", train=True, download=False)

for i in range(500):
    # Copy edge image
    edge_path = f"data/edges/img_{i}.png"
    shutil.copy(edge_path, f"data/sample_edges/img_{i}.png")

    # Save original image
    img, _ = dataset[i]
    img = np.array(img)

    cv2.imwrite(f"data/sample_original/img_{i}.png", img)

print("Sample dataset created (edges + original)")