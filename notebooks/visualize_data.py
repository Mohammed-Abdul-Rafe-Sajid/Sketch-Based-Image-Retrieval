import matplotlib.pyplot as plt
import torchvision

dataset = torchvision.datasets.CIFAR10(root="data/raw", train=True, download=False)

fig, axes = plt.subplots(1, 5, figsize=(10,2))

for i in range(5):
    img, label = dataset[i]
    axes[i].imshow(img)
    axes[i].axis("off")

plt.savefig("reports/figures/sample_images.png")
plt.show()