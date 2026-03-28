import numpy as np
from torchvision.datasets import CIFAR10


import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from src.retrieval.retrieve import retrieve

# Load dataset (for labels)
dataset = CIFAR10(root="data/raw", train=True, download=False)

# Load filenames
filenames = np.load("features/filenames.npy", allow_pickle=True)


# --- FUNCTION: get label from filename ---
def get_label_from_filename(file):
    idx = int(file.split("_")[1].split(".")[0])
    _, label = dataset[idx]
    return label


# --- Precision@K ---
def precision_at_k(query_idx, k=5):
    query_file = f"img_{query_idx}.png"

    # Get true label
    true_label = get_label_from_filename(query_file)

    # Retrieve results
    results = retrieve(f"data/edges/{query_file}", top_k=k)

    correct = 0

    for file, _ in results:
        if get_label_from_filename(file) == true_label:
            correct += 1

    return correct / k


# --- TEST MULTIPLE QUERIES ---
def evaluate(num_samples=50, k=5):
    scores = []

    for i in range(num_samples):
        score = precision_at_k(i, k)
        scores.append(score)

    avg_score = sum(scores) / len(scores)

    return avg_score


# --- RUN ---
if __name__ == "__main__":
    evaluate(num_samples=50, k=5)