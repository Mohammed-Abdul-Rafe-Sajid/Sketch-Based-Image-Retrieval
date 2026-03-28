import streamlit as st
import numpy as np
import cv2
import torch
import torchvision.transforms as transforms
from torchvision.models import resnet18, ResNet18_Weights

# ---------------- MODEL ----------------
model = resnet18(weights=ResNet18_Weights.DEFAULT)
model = torch.nn.Sequential(*list(model.children())[:-1])
model.eval()

transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# ---------------- LOAD DATA ----------------
filenames = np.load("features/filenames.npy", allow_pickle=True)
embeddings = np.load("features/embeddings.npy")

# ---------------- FUNCTIONS ----------------
def extract_embedding(img):
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    img = transform(img).unsqueeze(0)

    with torch.no_grad():
        emb = model(img)

    return emb.view(-1).numpy()


def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


def retrieve(query_img, top_k=5):
    query_emb = extract_embedding(query_img)

    sims = []
    for i, emb in enumerate(embeddings):
        sim = cosine_similarity(query_emb, emb)
        sims.append((filenames[i], sim))

    sims.sort(key=lambda x: x[1], reverse=True)
    return sims[:top_k]


# ---------------- UI ----------------
st.title("🖊️ Sketch-Based Image Retrieval")
st.markdown("Upload a sketch or edge image to retrieve visually similar images using deep learning and Spark.")




st.info("Note: This demo uses a small sample dataset for fast performance.")
uploaded_file = st.file_uploader("Upload a sketch (or edge image)", type=["png", "jpg"])

if uploaded_file:
    # Read image
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_GRAYSCALE)

    st.subheader("Query Image")
    st.image(img, channels="GRAY")

    results = retrieve(img)

    st.subheader("Top Matches")

    cols = st.columns(len(results))

    for i, (file, score) in enumerate(results):
        # Load from sample dataset (NOT full dataset)
        result_img = cv2.imread(f"data/sample_original/{file}")
        result_img = cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB)

        with cols[i]:
            st.image(result_img, caption=f"Similarity: {score:.2f}", width=150)

    # ---------------- EVALUATION ----------------
    st.markdown("---")
    st.subheader("📊 System Evaluation")

    if st.button("Show Evaluation (Precision@5)"):
        try:
            from src.evaluation.evaluate import evaluate
            score = evaluate(num_samples=20, k=5)
            st.write(f"Average Precision@5: {score:.2f}")
        except Exception:
            st.warning("Evaluation module not available in deployed version.")

with st.expander("ℹ️ How this works"):
    st.write("""
    - Images are converted to edge representations
    - A CNN (ResNet18) extracts feature embeddings
    - Cosine similarity is used to find similar images
    - Apache Spark enables scalable processing
    """)
st.markdown("---")
st.markdown("Built by Mohammed Abdul Rafe Sajid 🚀")
# import streamlit as st
# import numpy as np
# import cv2
# import torch
# import torchvision.transforms as transforms
# from torchvision.models import resnet18, ResNet18_Weights
# from torchvision.datasets import CIFAR10

# # ---------------- MODEL ----------------
# model = resnet18(weights=ResNet18_Weights.DEFAULT)
# model = torch.nn.Sequential(*list(model.children())[:-1])
# model.eval()

# transform = transforms.Compose([
#     transforms.ToPILImage(),
#     transforms.Resize((224, 224)),
#     transforms.ToTensor(),
# ])

# # ---------------- LOAD DATA ----------------
# filenames = np.load("features/filenames.npy", allow_pickle=True)
# embeddings = np.load("features/embeddings.npy")

# # ---------------- FUNCTIONS ----------------
# def extract_embedding(img):
#     img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
#     img = transform(img).unsqueeze(0)

#     with torch.no_grad():
#         emb = model(img)

#     return emb.view(-1).numpy()


# def cosine_similarity(a, b):
#     return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


# def retrieve(query_img, top_k=5):
#     query_emb = extract_embedding(query_img)
#     sims = []

#     for i, emb in enumerate(embeddings):
#         sim = cosine_similarity(query_emb, emb)
#         sims.append((filenames[i], sim))

#     sims.sort(key=lambda x: x[1], reverse=True)
#     return sims[:top_k]


# # ---------------- UI ----------------
# st.title("🖊️ Sketch-Based Image Retrieval")

# uploaded_file = st.file_uploader(
#     "Upload a sketch (or edge image)",
#     type=["png", "jpg"]
# )

# if uploaded_file:
#     # Read image
#     file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
#     img = cv2.imdecode(file_bytes, cv2.IMREAD_GRAYSCALE)

#     st.subheader("Query Image")
#     st.image(img, channels="GRAY")

#     results = retrieve(img)

#     st.subheader("Top Matches")
#     cols = st.columns(len(results))

#     for i, (file, score) in enumerate(results):
#         idx = int(file.split("_")[1].split(".")[0])

#         dataset = CIFAR10(root="data/raw", train=True, download=False)
#         img, _ = dataset[idx]
#         img = np.array(img)

#         with cols[i]:
#             st.image(img, caption=f"{score:.2f}", width=150)

#     st.markdown("---")

#     st.subheader("📊 System Evaluation")
#     if st.button("Show Evaluation (Precision@5)"):
#         from src.evaluation.evaluate import evaluate

#         score = evaluate(num_samples=20, k=5)
#         st.write(f"Average Precision@5: {score:.2f}")