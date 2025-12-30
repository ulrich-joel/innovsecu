# src/detection/meta/build_meta_features.py
"""
===============================================================================
Build Meta-Features from Unsupervised Models (with Indices)
===============================================================================

This script aggregates the outputs of multiple unsupervised anomaly detection
models (Isolation Forest, KMeans, VAE) into a unified meta-feature space.

It also saves the indices of the windows used, so labels can be aligned correctly
for the supervised meta-classifier (Random Forest).

------------------------------------------------------------------------------
Input:
- Isolation Forest scores
- KMeans distances and cluster IDs
- VAE reconstruction errors

------------------------------------------------------------------------------
Output:
- X_meta_train.npy
- X_meta_val.npy
- X_meta_test.npy
- idx_train.npy
- idx_val.npy
- idx_test.npy
- meta_feature_names.pkl

Author: Ngueyep Ulrich
Date: 2025-12-30
===============================================================================
"""

import os
import numpy as np
import joblib
from config.config import Config

# ------------------------------
# Initialization
# ------------------------------
config = Config()

MODELS_DIR = config.MODELS_DIR
OUTPUT_DIR = os.path.join(config.OUTPUT_DIR, "meta")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ------------------------------
# Load model outputs
# ------------------------------
def load_split(split):
    """
    Load outputs of all unsupervised models for a given split.
    """
    if_scores = np.load(os.path.join(MODELS_DIR, "isolation_forest", f"if_scores_{split}.npy"))
    kmeans_dist = np.load(os.path.join(MODELS_DIR, "kmeans", f"kmeans_dist_{split}.npy"))
    kmeans_labels = np.load(os.path.join(MODELS_DIR, "kmeans", f"kmeans_labels_{split}.npy"))
    vae_scores = np.load(os.path.join(MODELS_DIR, "vae", f"vae_scores_{split}.npy"))
    return if_scores, kmeans_dist, kmeans_labels, vae_scores

# ------------------------------
# Build meta-feature matrix and indices
# ------------------------------
def build_meta_X(split):
    """
    Build meta-feature matrix and corresponding indices for a split.
    """
    if_s, km_d, km_l, vae_s = load_split(split)
    
    # Combine features into a meta-feature matrix
    X_meta = np.column_stack([if_s, km_d, vae_s, km_l])
    
    # Indices of rows in X_meta
    idx = np.arange(X_meta.shape[0])
    
    return X_meta, idx

# ------------------------------
# Main
# ------------------------------
print("🔗 Building meta-features...")

X_meta_train, idx_train = build_meta_X("train")
X_meta_val,   idx_val   = build_meta_X("val")
X_meta_test,  idx_test  = build_meta_X("test")

print(f"✅ Meta Train: {X_meta_train.shape}")
print(f"✅ Meta Val  : {X_meta_val.shape}")
print(f"✅ Meta Test : {X_meta_test.shape}")

# ------------------------------
# Save meta-features and indices
# ------------------------------
np.save(os.path.join(OUTPUT_DIR, "X_meta_train.npy"), X_meta_train)
np.save(os.path.join(OUTPUT_DIR, "X_meta_val.npy"), X_meta_val)
np.save(os.path.join(OUTPUT_DIR, "X_meta_test.npy"), X_meta_test)

np.save(os.path.join(OUTPUT_DIR, "idx_train.npy"), idx_train)
np.save(os.path.join(OUTPUT_DIR, "idx_val.npy"), idx_val)
np.save(os.path.join(OUTPUT_DIR, "idx_test.npy"), idx_test)

# ------------------------------
# Save meta-feature names
# ------------------------------
meta_feature_names = [
    "if_score",
    "kmeans_distance",
    "vae_reconstruction_error",
    "kmeans_cluster_id"
]

joblib.dump(meta_feature_names, os.path.join(OUTPUT_DIR, "meta_feature_names.pkl"))

print("💾 Meta-features and indices saved successfully")
