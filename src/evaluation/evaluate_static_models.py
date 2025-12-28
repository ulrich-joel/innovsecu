# src/evaluation/evaluate_static_models.py
"""
====================================================================================
Static Anomaly Detection Model Evaluation Script
====================================================================================

This script evaluates static (unsupervised) anomaly detection models 
on preprocessed feature data. It supports evaluation of:

    - Isolation Forest models trained with different contamination levels (1%, 5%, 10%, 15%)
    - A KMeans model using distance-to-centroid thresholding

Outputs:
    - Console summary of model performance
    - ROC and Precision-Recall curves
    - Confusion matrices

Author  : Ngueyep Ulrich
Date    : 2025-10-14
====================================================================================
"""

import numpy as np
import joblib
import os
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest
from sklearn.cluster import KMeans
from sklearn.metrics import (
    precision_score, recall_score, f1_score, roc_auc_score,
    RocCurveDisplay, PrecisionRecallDisplay, ConfusionMatrixDisplay
)
from config.config import Config

# Load configuration
config = Config()

# Load test data
X_test = np.load(config.X_TEST_FILE)
y_test = (np.load(config.Y_TEST_FILE) > 0).astype(int)

print(f"Test data: {X_test.shape[0]} samples, {X_test.shape[1]} features")

# Helper function to load model or fail
def load_model_or_fail(model_path):
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}")
    return joblib.load(model_path)

# === Isolation Forest ===
contamination_levels = [1, 5, 10, 15]
models = {}

print("🔍 Evaluating Isolation Forest models:\n")
for cont in contamination_levels:
    model_name = f"IF_cont={cont}"
    model_path = f"{config.ISOLATION_FOREST_MODEL_PATH}_{cont}.pkl"
    model = load_model_or_fail(model_path)
    models[model_name] = model

results_summary = []

# Evaluate Isolation Forest models
for model_name, model in models.items():
    preds = np.where(model.predict(X_test) == -1, 1, 0)
    precision = precision_score(y_test, preds, zero_division=0)
    recall = recall_score(y_test, preds, zero_division=0)
    f1 = f1_score(y_test, preds, zero_division=0)
    try:
        auc = roc_auc_score(y_test, preds)
    except ValueError:
        auc = 0.5

    print(f"== {model_name} ==")
    print(f"Precision: {precision:.3f}, Recall: {recall:.3f}, F1: {f1:.3f}, AUC: {auc:.3f}\n")

    results_summary.append({
        'name': model_name,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'auc': auc,
        'preds': preds
    })

# === KMeans ===
print(" Evaluating KMeans model:\n")
kmeans = load_model_or_fail(config.KMEANS_MODEL_PATH)
kmeans_labels = kmeans.predict(X_test)
distances = np.linalg.norm(X_test - kmeans.cluster_centers_[kmeans_labels], axis=1)
threshold = np.percentile(distances, 95)
kmeans_preds = (distances > threshold).astype(int)

precision = precision_score(y_test, kmeans_preds, zero_division=0)
recall = recall_score(y_test, kmeans_preds, zero_division=0)
f1 = f1_score(y_test, kmeans_preds, zero_division=0)
try:
    auc = roc_auc_score(y_test, kmeans_preds)
except ValueError:
    auc = 0.5

print("== KMeans (95th percentile) ==")
print(f"Precision: {precision:.3f}, Recall: {recall:.3f}, F1: {f1:.3f}, AUC: {auc:.3f}\n")

results_summary.append({
    'name': "KMeans (95th percentile)",
    'precision': precision,
    'recall': recall,
    'f1': f1,
    'auc': auc,
    'preds': kmeans_preds,
    'distances': distances
})

# === Summary Table ===
print(" Model Performance Summary:")
print(f"{'name':<25} {'precision':<10} {'recall':<10} {'f1':<10} {'auc':<10}")
for res in results_summary:
    print(f"{res['name']:<25} {res['precision']:<10.3f} {res['recall']:<10.3f} {res['f1']:<10.3f} {res['auc']:<10.3f}")

# === Visualization ===
os.makedirs(config.EVALUATION_DIR, exist_ok=True)

# Best IF model for plots (cont=15)
try:
    print("\n Generating ROC & PR curves...")
    best_if_model = models['IF_cont=15']
    best_preds = np.where(best_if_model.predict(X_test) == -1, 1, 0)

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    RocCurveDisplay.from_predictions(y_test, best_preds, ax=axes[0, 0])
    axes[0, 0].set_title("ROC - Isolation Forest")

    PrecisionRecallDisplay.from_predictions(y_test, best_preds, ax=axes[0, 1])
    axes[0, 1].set_title("PR - Isolation Forest")

    RocCurveDisplay.from_predictions(y_test, results_summary[-1]['distances'], ax=axes[1, 0])
    axes[1, 0].set_title("ROC - KMeans")

    PrecisionRecallDisplay.from_predictions(y_test, results_summary[-1]['distances'], ax=axes[1, 1])
    axes[1, 1].set_title("PR - KMeans")

    for ax in axes.flat:
        ax.grid(True)

    plt.tight_layout()
    roc_path = os.path.join(config.EVALUATION_DIR, 'static_model_curves.png')
    plt.savefig(roc_path, dpi=300)
    print(f"✓ Saved ROC & PR curves to: {roc_path}")
    plt.close()

except Exception as e:
    print(f" Failed to generate ROC/PR curves: {e}")

# === Confusion Matrices ===
try:
    print(" Generating confusion matrices...")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    ConfusionMatrixDisplay.from_predictions(y_test, best_preds, ax=ax1, cmap='Blues')
    ax1.set_title("Confusion - Isolation Forest")

    ConfusionMatrixDisplay.from_predictions(y_test, results_summary[-1]['preds'], ax=ax2, cmap='Blues')
    ax2.set_title("Confusion - KMeans")

    plt.tight_layout()
    cm_path = os.path.join(config.EVALUATION_DIR, 'static_confusion_matrices.png')
    plt.savefig(cm_path, dpi=300)
    print(f"✓ Saved confusion matrices to: {cm_path}")
    plt.close()

except Exception as e:
    print(f" Failed to generate confusion matrices: {e}")

print("\n✓ Static model evaluation completed.")

# # src/evaluation/evaluate_static_models.py
# """
# ====================================================================================
# Static Anomaly Detection Model Evaluation Script
# ====================================================================================

# This script evaluates static (unsupervised) anomaly detection models 
# on preprocessed feature data. It supports evaluation of:

#     - Isolation Forest models trained with different contamination levels (1%, 5%, 10%, 15%)
#     - A KMeans model using distance-to-centroid thresholding

# It loads pre-trained models from disk, applies them to test data, and computes
# a range of performance metrics including precision, recall, F1-score, and ROC AUC.
# Visualizations such as ROC/PR curves and confusion matrices are generated and saved.

# ------------------------------------------------------------------------------------

# Usage:
#     - Ensure that the following `.npy` files exist:
#         - `x_test.npy`, `y_test.npy` (and optionally `x_train.npy`)
#     - Trained models should exist at paths configured in `config.py`:
#         - e.g., `isolation_forest_1.joblib`, `kmeans_model.joblib`, etc.

# Outputs:
#     - Console summary of model performance
#     - ROC and Precision-Recall curves saved as `static_model_curves.png`
#     - Confusion matrices saved as `static_confusion_matrices.png`
#     - All saved files are placed in the directory defined by `config.EVALUATION_DIR`

# ------------------------------------------------------------------------------------

# Author  : Ngueyep Ulrich
# Date    : 2025-10-14
# Version : 1.0

# ====================================================================================
# """


# import numpy as np
# import joblib
# from sklearn.ensemble import IsolationForest
# from sklearn.cluster import KMeans
# from config.config import Config
# from sklearn.metrics import (
#     precision_score, recall_score, f1_score, roc_auc_score,
#     RocCurveDisplay, PrecisionRecallDisplay, ConfusionMatrixDisplay
# )
# import matplotlib.pyplot as plt
# import os

# # Load configuration
# config = Config()

# # Load test data
# X_test = np.load(config.X_TEST_FILE)
# y_test = (np.load(config.Y_TEST_FILE) > 0).astype(int)

# print(f"Test data: {X_test.shape[0]} samples, {X_test.shape[1]} features")

# # Load model helper
# def load_model_or_fail(model_path):
#     if not os.path.exists(model_path):
#         raise FileNotFoundError(f"Model not found: {model_path}")
#     return joblib.load(model_path)

# # Load Isolation Forest models
# contamination_levels = [0.01, 0.05, 0.10, 0.15]
# models = {}

# print("\nLoading Isolation Forest models:")
# for cont, model_path in config.ISOLATION_FOREST_MODELS.items():
#     model_name = f"IF_cont={cont}"
#     models[model_name] = load_model_or_fail(model_path)
#     print(f"  ✓ Loaded {model_name}")


# # Load KMeans model
# print("\nLoading KMeans model:")
# kmeans = load_model_or_fail(config.KMEANS_MODEL_PATH)
# print("  ✓ Loaded KMeans model")

# results_summary = []

# # Evaluate Isolation Forest models
# print("\nEvaluating Isolation Forest models:")
# for model_name, model in models.items():
#     preds = np.where(model.predict(X_test) == -1, 1, 0)
#     precision = precision_score(y_test, preds, zero_division=0)
#     recall = recall_score(y_test, preds, zero_division=0)
#     f1 = f1_score(y_test, preds, zero_division=0)
#     try:
#         auc = roc_auc_score(y_test, preds)
#     except ValueError:
#         auc = 0.5
#     print(f"{model_name}: Precision={precision:.3f}, Recall={recall:.3f}, F1={f1:.3f}, AUC={auc:.3f}")
#     results_summary.append({
#         'name': model_name, 'precision': precision,
#         'recall': recall, 'f1': f1, 'auc': auc,
#         'detections': np.sum(preds)
#     })

# # Evaluate KMeans
# print("\nEvaluating KMeans model:")
# kmeans_labels = kmeans.predict(X_test)
# distances = np.linalg.norm(X_test - kmeans.cluster_centers_[kmeans_labels], axis=1)
# threshold = np.percentile(distances, 95)
# kmeans_preds = (distances > threshold).astype(int)
# precision = precision_score(y_test, kmeans_preds, zero_division=0)
# recall = recall_score(y_test, kmeans_preds, zero_division=0)
# f1 = f1_score(y_test, kmeans_preds, zero_division=0)
# try:
#     auc = roc_auc_score(y_test, kmeans_preds)
# except ValueError:
#     auc = 0.5
# print(f"KMeans (95th percentile): Precision={precision:.3f}, Recall={recall:.3f}, F1={f1:.3f}, AUC={auc:.3f}")
# results_summary.append({
#     'name': 'KMeans (95th percentile)', 'precision': precision,
#     'recall': recall, 'f1': f1, 'auc': auc,
#     'detections': np.sum(kmeans_preds)
# })

# # Summary table
# print("\nModel Performance Summary:")
# print(f"{'Name':<25} {'Precision':<10} {'Recall':<10} {'F1':<10} {'AUC':<10}")
# print("-" * 65)
# for res in results_summary:
#     print(f"{res['name']:<25} {res['precision']:<10.3f} {res['recall']:<10.3f} {res['f1']:<10.3f} {res['auc']:<10.3f}")

# # === Visualization ===
# os.makedirs(config.EVALUATION_DIR, exist_ok=True)

# # ROC & PR curves
# try:
#     print("\nGenerating ROC & PR curves...")
#     fig, axes = plt.subplots(2, 2, figsize=(12, 10))

#     best_model = models['IF_cont=15']
#     best_preds = np.where(best_model.predict(X_test) == -1, 1, 0)

#     RocCurveDisplay.from_predictions(y_test, best_preds, ax=axes[0, 0])
#     axes[0, 0].set_title("ROC - Isolation Forest")
#     PrecisionRecallDisplay.from_predictions(y_test, best_preds, ax=axes[0, 1])
#     axes[0, 1].set_title("PR - Isolation Forest")

#     RocCurveDisplay.from_predictions(y_test, distances, ax=axes[1, 0])
#     axes[1, 0].set_title("ROC - KMeans")
#     PrecisionRecallDisplay.from_predictions(y_test, distances, ax=axes[1, 1])
#     axes[1, 1].set_title("PR - KMeans")

#     for ax in axes.flat:
#         ax.grid(True)

#     plt.tight_layout()
#     roc_path = os.path.join(config.EVALUATION_DIR, 'static_model_curves.png')
#     plt.savefig(roc_path, dpi=300, bbox_inches='tight')
#     print(f"✓ Saved ROC & PR curves to: {roc_path}")
#     plt.close()
# except Exception as e:
#     print(f" Failed to generate curves: {e}")

# # Confusion matrices
# try:
#     print("Generating confusion matrices...")
#     fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
#     ConfusionMatrixDisplay.from_predictions(y_test, best_preds, ax=ax1, cmap='Blues')
#     ax1.set_title("Confusion - Isolation Forest")
#     ConfusionMatrixDisplay.from_predictions(y_test, kmeans_preds, ax=ax2, cmap='Blues')
#     ax2.set_title("Confusion - KMeans")
#     plt.tight_layout()
#     cm_path = os.path.join(config.EVALUATION_DIR, 'static_confusion_matrices.png')
#     plt.savefig(cm_path, dpi=300, bbox_inches='tight')
#     print(f"✓ Saved confusion matrices to: {cm_path}")
#     plt.close()
# except Exception as e:
#     print(f" Failed to generate confusion matrices: {e}")

# print("\n Static model evaluation completed.")
