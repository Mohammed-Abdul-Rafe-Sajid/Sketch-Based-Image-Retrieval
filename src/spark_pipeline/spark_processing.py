from pyspark.sql import SparkSession
import cv2
import numpy as np
import os

# Initialize Spark
spark = SparkSession.builder \
    .appName("SketchCBIR") \
    .getOrCreate()

sc = spark.sparkContext

# Path
image_dir = "data/edges"

# Get list of image paths
image_paths = [os.path.join(image_dir, f) for f in os.listdir(image_dir)]

# Parallelize
rdd = sc.parallelize(image_paths)


# --- FUNCTION: Load and process image ---
def process_image(path):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)

    # Simple feature: flatten image (for demo)
    feature = img.flatten()

    return (path, feature)


# Apply in parallel
features_rdd = rdd.map(process_image)

# Collect results
results = features_rdd.collect()

print("Processed images using Spark:", len(results))