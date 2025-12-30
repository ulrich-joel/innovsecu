"""
===============================================================================
Train Isolation Forest on Temporal Behavioral Features
===============================================================================

This script trains an Isolation Forest model on temporal features derived
from sliding windows over event-level system activity.

The model learns normal behavioral patterns in an unsupervised manner
and produces anomaly scores used later as meta-features.

------------------------------------------------------------------------------
Input:
- X_train.npy : temporal training features
- X_val.npy   : temporal validation features
- X_test.npy  : temporal testing features

------------------------------------------------------------------------------
Output:
- if_model.pkl        : trained Isolation Forest model
- if_scores_train.npy : anomaly scores for training set
- if_scores_val.npy   : anomaly scores for validation set
- if_scores_test.npy  : anomaly scores for testing set

------------------------------------------------------------------------------
Notes:
- Higher scores indicate higher anomaly likelihood
- Model is trained ONLY on training data
- Scores are computed consistently across all splits

Author: Ngueyep Ulrich
Date: 2025-12-29
===============================================================================
"""

import os
import numpy as np
import joblib
from sklearn.ensemble import IsolationForest
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
    "isolation_forest"
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
# Train Isolation Forest
# ------------------------------
print("🌲 Training Isolation Forest...")

if_model = IsolationForest(
    n_estimators=200,
    max_samples="auto",
    contamination="auto",
    random_state=42,
    n_jobs=-1
)

if_model.fit(X_train)

# ------------------------------
# Compute anomaly scores
# ------------------------------
print("📊 Computing anomaly scores...")

# We use negative score_samples so that:
# higher score = more anomalous
scores_train = -if_model.score_samples(X_train)
scores_val   = -if_model.score_samples(X_val)
scores_test  = -if_model.score_samples(X_test)

# ------------------------------
# Save outputs
# ------------------------------
joblib.dump(if_model, os.path.join(OUTPUT_DIR, "if_model.pkl"))

np.save(os.path.join(OUTPUT_DIR, "if_scores_train.npy"), scores_train)
np.save(os.path.join(OUTPUT_DIR, "if_scores_val.npy"), scores_val)
np.save(os.path.join(OUTPUT_DIR, "if_scores_test.npy"), scores_test)

print(f"💾 Isolation Forest model and scores saved in {OUTPUT_DIR}")
print(f"   - Scores train: {scores_train.shape}")
print(f"   - Scores val  : {scores_val.shape}")
print(f"   - Scores test : {scores_test.shape}")
