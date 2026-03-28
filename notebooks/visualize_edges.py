import cv2
import matplotlib.pyplot as plt

img = cv2.imread("data/raw/cifar-10-batches-py/../..", cv2.IMREAD_COLOR)

# Instead, load one processed image
edge_img = cv2.imread("data/edges/img_0.png", cv2.IMREAD_GRAYSCALE)

plt.imshow(edge_img, cmap='gray')
plt.title("Edge Image (Sketch-like)")
plt.axis("off")

plt.savefig("reports/figures/edge_sample.png")
plt.show()