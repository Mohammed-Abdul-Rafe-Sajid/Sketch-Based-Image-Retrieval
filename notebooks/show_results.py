import cv2
import matplotlib.pyplot as plt

def show_results(query, results):
    plt.figure(figsize=(10,3))

    # Show query
    img = cv2.imread(query, cv2.IMREAD_GRAYSCALE)
    plt.subplot(1,6,1)
    plt.imshow(img, cmap='gray')
    plt.title("Query")
    plt.axis("off")

    # Show results
    for i, (file, _) in enumerate(results):
        img = cv2.imread(f"data/edges/{file}", cv2.IMREAD_GRAYSCALE)
        plt.subplot(1,6,i+2)
        plt.imshow(img, cmap='gray')
        plt.title(f"Top {i+1}")
        plt.axis("off")

    plt.savefig("reports/figures/retrieval_result.png")
    plt.close()