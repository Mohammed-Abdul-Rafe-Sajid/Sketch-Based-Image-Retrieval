import torchvision
import torchvision.transforms as transforms
import os

# Create data directory
os.makedirs("data/raw", exist_ok=True)

transform = transforms.Compose([
    transforms.ToTensor()
])

dataset = torchvision.datasets.CIFAR10(
    root="data/raw",
    train=True,
    download=True,
    transform=transform
)

print("Dataset downloaded successfully!")
print(f"Total images: {len(dataset)}")