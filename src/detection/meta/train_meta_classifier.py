# src/detection/meta/train_meta_classifier.py
"""
===============================================================================
Train Random Forest Meta-Classifier with Correct Label Alignment and SMOTE
===============================================================================

This script trains a supervised Random Forest classifier on meta-features
constructed from multiple unsupervised anomaly detection models.

It ensures that labels are aligned with meta-features using saved indices
and applies SMOTE to balance the training set.

------------------------------------------------------------------------------
Input:
- X_meta_train.npy, X_meta_val.npy, X_meta_test.npy
- y_train.npy, y_val.npy, y_test.npy
- idx_train.npy, idx_val.npy, idx_test.npy

------------------------------------------------------------------------------
Output:
- rf_meta_model.pkl
- y_pred_test.npy
- y_score_test.npy
- feature_importance.png

Author: Ngueyep Ulrich
Date: 2025-12-30
===============================================================================
"""

import os
import numpy as np
import joblib
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score
from imblearn.over_sampling import SMOTE
from config.config import Config

# ------------------------------
# Initialization
# ------------------------------
config = Config()

META_DIR = os.path.join(config.OUTPUT_DIR, "meta")
LABEL_DIR = os.path.join(config.OUTPUT_DIR, "processed", "processed_dynamic", "temporal")
OUTPUT_DIR = os.path.join(config.MODELS_DIR, "meta_classifier")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ------------------------------
# Load meta-features and indices
# ------------------------------
print("📥 Loading meta-features and indices...")
X_train = np.load(os.path.join(META_DIR, "X_meta_train.npy"))
X_val   = np.load(os.path.join(META_DIR, "X_meta_val.npy"))
X_test  = np.load(os.path.join(META_DIR, "X_meta_test.npy"))

idx_train = np.load(os.path.join(META_DIR, "idx_train.npy"))
idx_val   = np.load(os.path.join(META_DIR, "idx_val.npy"))
idx_test  = np.load(os.path.join(META_DIR, "idx_test.npy"))

# ------------------------------
# Load full labels
# ------------------------------
y_train_full = np.load(os.path.join(LABEL_DIR, "y_train.npy"))
y_val_full   = np.load(os.path.join(LABEL_DIR, "y_val.npy"))
y_test_full  = np.load(os.path.join(LABEL_DIR, "y_test.npy"))

# ------------------------------
# Align labels with meta-features using indices
# ------------------------------
y_train = y_train_full[idx_train]
y_val   = y_val_full[idx_val]
y_test  = y_test_full[idx_test]

print("✅ Data shapes after alignment:")
print(f"   X_train: {X_train.shape}, y_train: {y_train.shape}")
print(f"   X_val:   {X_val.shape},   y_val:   {y_val.shape}")
print(f"   X_test:  {X_test.shape},  y_test:  {y_test.shape}")

# ------------------------------
# Apply SMOTE to balance classes
# ------------------------------
print("🌱 Applying SMOTE to balance classes in training set...")
smote = SMOTE(random_state=42)
X_train_res, y_train_res = smote.fit_resample(X_train, y_train)
print(f"   Resampled X_train: {X_train_res.shape}, y_train: {y_train_res.shape}")

# ------------------------------
# Train Random Forest Meta-Classifier
# ------------------------------
print("🌲 Training Random Forest meta-classifier...")
rf = RandomForestClassifier(
    n_estimators=300,
    max_depth=None,
    random_state=42,
    class_weight="balanced",
    n_jobs=-1
)
rf.fit(X_train_res, y_train_res)

# ------------------------------
# Evaluate on test set
# ------------------------------
print("📊 Evaluating on test set...")
y_pred = rf.predict(X_test)
y_score = rf.predict_proba(X_test)[:, 1]

print(classification_report(y_test, y_pred))
print("ROC-AUC:", roc_auc_score(y_test, y_score))

# ------------------------------
# Feature importance visualization
# ------------------------------
importances = rf.feature_importances_
plt.figure(figsize=(8, 5))
plt.bar(range(len(importances)), importances)
plt.xlabel("Meta-feature index")
plt.ylabel("Importance")
plt.title("Meta-feature importance (Random Forest)")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "feature_importance.png"))
plt.close()

# ------------------------------
# Save trained model and outputs
# ------------------------------
joblib.dump(rf, os.path.join(OUTPUT_DIR, "rf_meta_model.pkl"))
np.save(os.path.join(OUTPUT_DIR, "y_pred_test.npy"), y_pred)
np.save(os.path.join(OUTPUT_DIR, "y_score_test.npy"), y_score)

# ------------------------------
# Save aligned labels
# ------------------------------
np.save(os.path.join(OUTPUT_DIR, "y_train.npy"), y_train)
np.save(os.path.join(OUTPUT_DIR, "y_val.npy"), y_val)
np.save(os.path.join(OUTPUT_DIR, "y_test.npy"), y_test)

print



# # src/detection/meta/train_meta_classifier.py
# """
# ===============================================================================
# Train Meta-Classifier (Random Forest) with Correct Label Alignment
# ===============================================================================

# This script trains a supervised Random Forest classifier on meta-features
# constructed from multiple unsupervised anomaly detection models.

# It ensures that labels are aligned with meta-features using saved indices,
# avoiding dimension mismatches and enabling proper supervised training.

# ------------------------------------------------------------------------------
# Input:
# - X_meta_train.npy, X_meta_val.npy, X_meta_test.npy
# - y_train.npy, y_val.npy, y_test.npy
# - idx_train.npy, idx_val.npy, idx_test.npy

# ------------------------------------------------------------------------------
# Output:
# - rf_meta_model.pkl               : Saved Random Forest model
# - y_pred_test.npy                 : Predictions on the test set
# - y_score_test.npy                : Prediction probabilities (scores)
# - feature_importance.png          : Visualization of feature importance

# Author: Ngueyep Ulrich
# Date: 2025-12-30
# ===============================================================================
# """

# import os
# import numpy as np
# import joblib
# import matplotlib.pyplot as plt
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.metrics import classification_report, roc_auc_score
# from config.config import Config

# # ------------------------------
# # Initialization
# # ------------------------------
# config = Config()

# # Directory containing meta-features
# META_DIR = os.path.join(config.OUTPUT_DIR, "meta")

# # Directory containing labels and indices
# LABEL_DIR = os.path.join(config.OUTPUT_DIR, "processed", "processed_dynamic", "temporal")

# # Directory to save trained model and outputs
# OUTPUT_DIR = os.path.join(config.MODELS_DIR, "meta_classifier")
# os.makedirs(OUTPUT_DIR, exist_ok=True)

# # ------------------------------
# # Load meta-features
# # ------------------------------
# print("📥 Loading meta-features...")

# X_train = np.load(os.path.join(META_DIR, "X_meta_train.npy"))
# X_val   = np.load(os.path.join(META_DIR, "X_meta_val.npy"))
# X_test  = np.load(os.path.join(META_DIR, "X_meta_test.npy"))

# # ------------------------------
# # Load labels and indices
# # ------------------------------
# print("📥 Loading labels and indices...")

# y_train_full = np.load(os.path.join(LABEL_DIR, "y_train.npy"))
# y_val_full   = np.load(os.path.join(LABEL_DIR, "y_val.npy"))
# y_test_full  = np.load(os.path.join(LABEL_DIR, "y_test.npy"))

# idx_train = np.load(os.path.join(META_DIR, "idx_train.npy"))
# idx_val   = np.load(os.path.join(META_DIR, "idx_val.npy"))
# idx_test  = np.load(os.path.join(META_DIR, "idx_test.npy"))

# # ------------------------------
# # Align labels with meta-features
# # ------------------------------
# y_train = y_train_full[idx_train]
# y_val   = y_val_full[idx_val]
# y_test  = y_test_full[idx_test]

# print(f"✅ Data shapes after alignment:")
# print(f"   X_train: {X_train.shape}, y_train: {y_train.shape}")
# print(f"   X_val:   {X_val.shape},   y_val:   {y_val.shape}")
# print(f"   X_test:  {X_test.shape},  y_test:  {y_test.shape}")

# # ------------------------------
# # Train Random Forest Meta-Classifier
# # ------------------------------
# print("🌲 Training Random Forest meta-classifier...")

# rf = RandomForestClassifier(
#     n_estimators=300,
#     max_depth=None,
#     random_state=42,
#     n_jobs=-1,
#     class_weight="balanced"
# )

# rf.fit(X_train, y_train)

# # ------------------------------
# # Evaluate on test set
# # ------------------------------
# print("📊 Evaluating on test set...")

# y_pred = rf.predict(X_test)
# y_score = rf.predict_proba(X_test)[:, 1]

# print(classification_report(y_test, y_pred))
# print("ROC-AUC:", roc_auc_score(y_test, y_score))

# # ------------------------------
# # Feature importance visualization
# # ------------------------------
# importances = rf.feature_importances_

# plt.figure(figsize=(8, 5))
# plt.bar(range(len(importances)), importances)
# plt.xlabel("Meta-feature index")
# plt.ylabel("Importance")
# plt.title("Meta-feature importance (Random Forest)")
# plt.tight_layout()
# plt.savefig(os.path.join(OUTPUT_DIR, "feature_importance.png"))
# plt.close()

# # ------------------------------
# # Save trained model and outputs
# # ------------------------------
# joblib.dump(rf, os.path.join(OUTPUT_DIR, "rf_meta_model.pkl"))
# np.save(os.path.join(OUTPUT_DIR, "y_pred_test.npy"), y_pred)
# np.save(os.path.join(OUTPUT_DIR, "y_score_test.npy"), y_score)

# print("💾 Meta-classifier trained and saved successfully!")
