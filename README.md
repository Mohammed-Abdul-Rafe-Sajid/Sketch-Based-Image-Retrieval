# 🖊️ Sketch-Based Image Retrieval using Apache Spark

## 🚀 Live Demo

👉 <<https://sketch-based-image-retrieval.streamlit.app/>>

---

## 📌 Overview

This project implements a **Sketch-Based Image Retrieval (SBIR)** system that retrieves visually similar images from a dataset based on a user-provided sketch or edge representation.

Unlike traditional text-based search, this system leverages **visual content (edges and shapes)** to perform retrieval, making it suitable for applications where textual descriptions are insufficient.

The system integrates:

* Computer Vision (Edge Detection)
* Deep Learning (Feature Embeddings)
* Big Data Processing (Apache Spark)

---

## 🎯 Objectives

* Enable image retrieval using sketch-based input
* Extract meaningful visual features using deep learning
* Perform similarity search efficiently
* Demonstrate scalability using Apache Spark
* Evaluate system performance using quantitative metrics

---

## 🧠 Key Concept

Instead of comparing raw images, the system converts images into **feature embeddings**:

Image → Edge Detection → CNN → Feature Vector → Similarity Matching

This ensures efficient and meaningful comparison between query sketches and dataset images.

---

## ⚙️ Tech Stack

* Python
* PyTorch (Deep Learning)
* OpenCV (Image Processing)
* Apache Spark (PySpark)
* Streamlit (UI for demo)
* NumPy

---

## 🧩 System Architecture

1. **Dataset Loading**

   * CIFAR-10 dataset used for initial implementation
   * Provides labeled images for evaluation

2. **Preprocessing (Edge Detection)**

   * Convert images to grayscale
   * Apply Canny Edge Detection
   * Generate sketch-like representations

3. **Feature Extraction**

   * Use pretrained ResNet18
   * Remove classification layer
   * Extract 512-dimensional embeddings

4. **Similarity Search**

   * Compute cosine similarity
   * Retrieve top-K similar images

5. **Spark Integration**

   * Parallelize image processing using RDDs
   * Demonstrate scalability for large datasets

6. **User Interface**

   * Upload sketch/image
   * Display top matches
   * Show similarity scores

---

## 📂 Project Structure

```
sketch-cbir/
│
├── data/
│   ├── raw/
│   └── edges/
│
├── features/
│   ├── filenames.npy
│   └── embeddings.npy
│
├── src/
│   ├── preprocessing/
│   ├── feature_extraction/
│   ├── retrieval/
│   ├── evaluation/
│   └── spark_pipeline/
│
├── notebooks/
├── reports/
├── app.py
├── requirements.txt
└── README.md
```

---

## 📦 Dataset

### CIFAR-10

* 50,000 training images
* 10 classes (airplane, car, bird, etc.)
* Image size: 32×32

### Why CIFAR-10?

* Lightweight and fast for prototyping
* Provides labeled data for evaluation
* Easy integration with PyTorch

> Note: For higher-quality results, datasets like STL-10 or ImageNet can be used.

---

## 🔍 Feature Representation

* Model: ResNet18 (pretrained on ImageNet)
* Output: 512-dimensional embedding vector

### Storage Format

* `filenames.npy` → image identifiers
* `embeddings.npy` → feature vectors

This separation ensures efficient computation and scalability.

---

## 📐 Similarity Metric

### Cosine Similarity

Measures similarity between two vectors:

* Range: [-1, 1]
* Higher value = more similar

Used to retrieve top-K matching images.

---

## 📊 Evaluation

### Metric: Precision@K

Measures the proportion of relevant images among the top-K retrieved results.

#### Formula:

Precision@K = (Number of relevant images in top K) / K

### Example:

* Query: airplane
* Top 5 results: 3 airplanes
* Precision@5 = 0.6

### Result:

> Average Precision@5: ~0.5 – 0.7 (varies)

---

## ⚡ Spark Integration

Apache Spark is used to enhance scalability:

* Distributed image processing
* Parallel computation using RDDs
* Efficient handling of large datasets

### Role of Spark

* Data loading
* Parallel preprocessing
* Feature handling

> Deep learning is handled using PyTorch, while Spark manages data scalability.

---

## 🎨 User Interface

Built using Streamlit:

* Upload sketch or image
* View query image
* Display top-K similar results
* Show similarity scores
* Optional evaluation display

---

## 🧪 How to Run

### 1. Install dependencies

```
pip install -r requirements.txt
```

### 2. Run feature extraction

```
python src/feature_extraction/extract_features.py
```

### 3. Run retrieval (test)

```
python src/retrieval/retrieve.py
```

### 4. Launch UI

```
streamlit run app.py
```

---

## 🧠 Key Insights

* Edge-based representation enables sketch matching
* Deep features capture semantic similarity
* Spark improves scalability for large datasets
* Evaluation metrics validate system performance

---

## ⚠️ Limitations

* CIFAR-10 images are low resolution (32×32)
* Edge detection may lose fine details
* Retrieval accuracy depends on feature quality

---

## 🚀 Future Work

* Use higher-resolution datasets (STL-10, ImageNet)
* Implement sketch-specific datasets (TU-Berlin, Sketchy)
* Add multimodal retrieval (text + image)
* Improve embeddings using fine-tuning
* Deploy on cloud for large-scale usage

---

## 📄 Research Potential

This project can be extended into a research paper focusing on:

* Sketch-based retrieval techniques
* Feature optimization
* Scalable CBIR systems using Spark

---

## 🤝 Contribution

Feel free to fork, improve, and experiment with the project.

---

## 📬 Author

Mohammed Abdul Rafe Sajid

---


