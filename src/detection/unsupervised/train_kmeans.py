# src/detection/unsupervised/train_kmeans.py
"""
===============================================================================
Train KMeans on Temporal Behavioral Features
===============================================================================

This script applies KMeans clustering to temporal behavioral features extracted
from sliding windows over system events.

The objective is to:
- identify dominant behavioral patterns
- compute distance-to-centroid as an unsupervised anomaly score

The number of clusters k is selected using a combination of:
- Elbow method (inertia)
- Silhouette score

------------------------------------------------------------------------------
Input:
- X_train.npy
- X_val.npy
- X_test.npy

------------------------------------------------------------------------------
Output:
- kmeans_model.pkl
- cluster labels and distances for train / val / test

Author: Ngueyep Ulrich
Date: 2025-12-29
===============================================================================
"""

import os
import numpy as np
import joblib
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from config.config import Config

# ------------------------------
# Initialization
# ------------------------------
config = Config()

INPUT_DIR = os.path.join(
    config.PROCESSED_DYNAMIC_DIR,
    "temporal"
)

OUTPUT_DIR = os.path.join(
    config.MODELS_DIR,
    "kmeans"
)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ------------------------------
# Load data
# ------------------------------
print("📥 Loading temporal features...")

X_train = np.load(os.path.join(INPUT_DIR, "X_train.npy"))
X_val   = np.load(os.path.join(INPUT_DIR, "X_val.npy"))
X_test  = np.load(os.path.join(INPUT_DIR, "X_test.npy"))

print(f"✅ Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")

# ------------------------------
# K selection (Elbow + Silhouette)
# ------------------------------
print("🔍 Selecting optimal k...")

K_RANGE = range(2, 15)
inertias = []
silhouettes = []

for k in K_RANGE:
    kmeans = KMeans(
        n_clusters=k,
        random_state=42,
        n_init=10
    )
    labels = kmeans.fit_predict(X_train)
    inertias.append(kmeans.inertia_)
    silhouettes.append(silhouette_score(X_train, labels))

# Plot k selection
plt.figure()
plt.plot(K_RANGE, inertias, marker='o', label="Inertia")
plt.plot(K_RANGE, silhouettes, marker='s', label="Silhouette")
plt.xlabel("Number of clusters (k)")
plt.legend()
plt.title("KMeans k selection (Elbow + Silhouette)")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "kmeans_k_selection.png"))
plt.close()

# Choose k maximizing silhouette
optimal_k = K_RANGE[np.argmax(silhouettes)]
print(f"✅ Selected k = {optimal_k}")

# ------------------------------
# Train final KMeans
# ------------------------------
print("🎯 Training final KMeans...")

kmeans = KMeans(
    n_clusters=optimal_k,
    random_state=42,
    n_init=20
)
kmeans.fit(X_train)

# ------------------------------
# Inference & distances
# ------------------------------
def compute_labels_and_distances(X):
    labels = kmeans.predict(X)
    centers = kmeans.cluster_centers_[labels]
    distances = np.linalg.norm(X - centers, axis=1)
    return labels, distances

labels_train, dist_train = compute_labels_and_distances(X_train)
labels_val,   dist_val   = compute_labels_and_distances(X_val)
labels_test,  dist_test  = compute_labels_and_distances(X_test)

# ------------------------------
# Save outputs
# ------------------------------
joblib.dump(kmeans, os.path.join(OUTPUT_DIR, "kmeans_model.pkl"))

np.save(os.path.join(OUTPUT_DIR, "kmeans_labels_train.npy"), labels_train)
np.save(os.path.join(OUTPUT_DIR, "kmeans_labels_val.npy"), labels_val)
np.save(os.path.join(OUTPUT_DIR, "kmeans_labels_test.npy"), labels_test)

np.save(os.path.join(OUTPUT_DIR, "kmeans_dist_train.npy"), dist_train)
np.save(os.path.join(OUTPUT_DIR, "kmeans_dist_val.npy"), dist_val)
np.save(os.path.join(OUTPUT_DIR, "kmeans_dist_test.npy"), dist_test)

print("💾 KMeans model and outputs saved")
print(f"   - Dist train: {dist_train.shape}")
print(f"   - Dist val  : {dist_val.shape}")
print(f"   - Dist test : {dist_test.shape}")
