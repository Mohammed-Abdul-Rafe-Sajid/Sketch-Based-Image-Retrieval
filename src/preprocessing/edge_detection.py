import cv2
import os
import numpy as np
from torchvision.datasets import CIFAR10

# Create output directory
os.makedirs("data/edges", exist_ok=True)

# Load dataset (no transform needed here)
dataset = CIFAR10(root="data/raw", train=True, download=False)

# Loop through images
for i in range(len(dataset)):
    img, label = dataset[i]

    # Convert PIL image to numpy array
    img = np.array(img)

    # Convert to grayscale (required for edge detection)
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    # Apply Canny Edge Detection
    edges = cv2.Canny(gray, threshold1=100, threshold2=200)

    # Save image
    cv2.imwrite(f"data/edges/img_{i}.png", edges)

    # Limit for testing (remove later)
    if i == 500:
        break

print("Edge images created successfully!")