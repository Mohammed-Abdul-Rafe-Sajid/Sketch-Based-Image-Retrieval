import numpy as np
import cv2
import torch
import torchvision.transforms as transforms
from torchvision.models import resnet18, ResNet18_Weights
import os
import sys
from pathlib import Path

# Add parent directory to path to import notebooks
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from notebooks.show_results import show_results
# Load model
model = resnet18(weights=ResNet18_Weights.DEFAULT)
model = torch.nn.Sequential(*list(model.children())[:-1])
model.eval()

# Transform
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# Load saved data
filenames = np.load("features/filenames.npy", allow_pickle=True)
embeddings = np.load("features/embeddings.npy")

# --- FUNCTION: Extract embedding from query image ---
def extract_query_embedding(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    img = transform(img).unsqueeze(0)

    with torch.no_grad():
        embedding = model(img)

    return embedding.view(-1).numpy()


# --- FUNCTION: Cosine similarity ---
def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


# --- MAIN RETRIEVAL ---
def retrieve_similar(query_path, top_k=5):
    query_embedding = extract_query_embedding(query_path)

    similarities = []

    for i, emb in enumerate(embeddings):
        sim = cosine_similarity(query_embedding, emb)
        similarities.append((filenames[i], sim))

    # Sort by similarity (descending)
    similarities.sort(key=lambda x: x[1], reverse=True)

    return similarities[:top_k]


# Alias for compatibility
retrieve = retrieve_similar


# --- TEST ---
if __name__ == "__main__":
    query_image = "data/edges/img_100.png"

    results = retrieve_similar(query_image, top_k=5)

    print("Top matches:")
    for file, score in results:
        print(file, "->", score)


    show_results(query_image, results)