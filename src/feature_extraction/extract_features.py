import torch
import torchvision.models as models
import torchvision.transforms as transforms
import cv2
import os
import numpy as np

# Create output folder
os.makedirs("features", exist_ok=True)

# Load pretrained ResNet18
model = models.resnet18(pretrained=True)

# Remove final classification layer
model = torch.nn.Sequential(*list(model.children())[:-1])

model.eval()

# Transform for input images
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# Path to edge images
edge_dir = "data/sample_edges" # Change to "data/edges" for full dataset

features = []

# Loop through edge images
for i, file in enumerate(os.listdir(edge_dir)):
    img_path = os.path.join(edge_dir, file)

    # Read image
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

    # Convert to 3 channels (required for ResNet)
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

    # Apply transformations
    img = transform(img).unsqueeze(0)

    # Extract features
    with torch.no_grad():
        embedding = model(img)

    # Flatten to vector
    embedding = embedding.view(-1).numpy()

    features.append((file, embedding))

    # Limit for testing
    if i == 500:
        break

# Save features
filenames = [f[0] for f in features]
embeddings = [f[1] for f in features]

# Convert to proper numpy arrays
embeddings = np.array(embeddings)

# Save separately
np.save("features/filenames.npy", filenames)
np.save("features/embeddings.npy", embeddings)

print("Features saved successfully!")
print("Embeddings shape:", embeddings.shape)

print(embedding.shape)
print("Feature extraction complete!")